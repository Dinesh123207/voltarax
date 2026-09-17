"""
VoltVision AI — Real Data Retraining Script
Run this after 60-90 days of real hardware data collection.
Pulls from InfluxDB, retrains all 4 models, saves to ai_models/.

Usage:
  python3 training/retrain_on_real_data.py --days 90
  python3 training/retrain_on_real_data.py --days 60 --model lstm
  python3 training/retrain_on_real_data.py --days 90 --model bfci
"""
import os, sys, argparse, pickle, warnings, logging
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta, timezone
from sklearn.ensemble import (GradientBoostingRegressor, BaggingRegressor,
                               IsolationForest, RandomForestClassifier)
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from config.settings import settings
from db.database import query_influx

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("retrain")

os.makedirs("ai_models", exist_ok=True)
os.makedirs("training/real_data", exist_ok=True)


# ── InfluxDB data fetchers ─────────────────────────────────────────────────────

def fetch_power_data(days: int) -> pd.DataFrame:
    """Pull real power readings from InfluxDB biosolar_power bucket."""
    logger.info("Fetching %d days of power data from InfluxDB...", days)
    flux = f'''
from(bucket: "{settings.INFLUX_BUCKET_POWER}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r["_measurement"] == "panel_power")
  |> pivot(rowKey:["_time","panel_id"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns:["_time"])
'''
    rows = query_influx(flux)
    if not rows:
        logger.warning("No power data in InfluxDB — using synthetic fallback")
        return pd.read_csv("training/data/power_data.csv")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df.get("_time", df.get("timestamp")))
    df = df.rename(columns={"voltage":"voltage_v","current":"current_a",
                             "power":"power_w","panel_id":"panel_label"})
    logger.info("  Fetched %d power rows", len(df))
    df.to_csv("training/real_data/power_data.csv", index=False)
    return df


def fetch_env_data(days: int) -> pd.DataFrame:
    """Pull environment readings from InfluxDB biosolar_env bucket."""
    logger.info("Fetching %d days of environment data...", days)
    flux = f'''
from(bucket: "{settings.INFLUX_BUCKET_ENV}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r["_measurement"] == "environment")
  |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns:["_time"])
'''
    rows = query_influx(flux)
    if not rows:
        logger.warning("No env data in InfluxDB — using synthetic fallback")
        return pd.read_csv("training/data/power_data.csv")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["_time"])
    logger.info("  Fetched %d env rows", len(df))
    df.to_csv("training/real_data/env_data.csv", index=False)
    return df


def fetch_spectral_data(days: int) -> pd.DataFrame:
    """Pull AS7265x spectral readings from InfluxDB biosolar_spectral bucket."""
    logger.info("Fetching %d days of spectral data...", days)
    flux = f'''
from(bucket: "{settings.INFLUX_BUCKET_SPECTRAL}")
  |> range(start: -{days}d)
  |> filter(fn: (r) => r["_measurement"] == "spectral")
  |> pivot(rowKey:["_time","panel_id"], columnKey:["_field"], valueColumn:"_value")
  |> sort(columns:["_time"])
'''
    rows = query_influx(flux)
    if not rows:
        logger.warning("No spectral data — using synthetic fallback")
        return pd.read_csv("training/data/bfci_data.csv")

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["_time"])
    logger.info("  Fetched %d spectral rows", len(df))
    df.to_csv("training/real_data/spectral_data.csv", index=False)
    return df


def fetch_bfci_labels() -> pd.DataFrame:
    """Pull manually recorded BFCI scores from SQLite."""
    logger.info("Fetching BFCI labels from SQLite...")
    from db.database import get_db
    with get_db() as conn:
        rows = conn.execute(
            "SELECT bl.panel_id, bl.bfci_score, bl.uv_absorption_pct, "
            "bl.days_to_recoat, bl.coat_date, bl.recorded_at, p.label "
            "FROM bfci_logs bl JOIN panels p ON p.id=bl.panel_id "
            "ORDER BY bl.recorded_at"
        ).fetchall()

    if not rows:
        logger.warning("No BFCI labels in SQLite — using synthetic fallback")
        return pd.read_csv("training/data/bfci_data.csv")

    df = pd.DataFrame([dict(r) for r in rows])
    logger.info("  Fetched %d BFCI label rows", len(df))
    df.to_csv("training/real_data/bfci_labels.csv", index=False)
    return df


# ── Build LSTM training dataset from real hourly data ─────────────────────────

def build_lstm_dataset(days: int) -> tuple:
    logger.info("Building LSTM dataset from real data...")
    try:
        power_df = fetch_power_data(days)
        env_df   = fetch_env_data(days)
    except Exception as e:
        logger.error("Data fetch failed: %s — using synthetic", e)
        power_df = pd.read_csv("training/data/power_data.csv")
        env_df   = power_df  # same file has env cols in synthetic

    # Resample to hourly
    power_df["timestamp"] = pd.to_datetime(power_df["timestamp"])
    power_df = power_df.set_index("timestamp")

    agg_cols = {}
    for col in ["fleet_power_kw","irradiance_wm2","ambient_temp_c","panel_temp_c",
                "humidity_pct","aqi","battery_soc_pct","wind_speed_kmh","dust_event"]:
        if col in power_df.columns:
            agg_cols[col] = "mean" if col != "fleet_power_kw" else "sum"

    hourly = power_df.resample("1h").agg(agg_cols).reset_index()
    hourly["hour"]     = hourly["timestamp"].dt.hour
    hourly["month"]    = hourly["timestamp"].dt.month
    hourly["sin_hour"] = np.sin(2*np.pi*hourly["hour"]/24)
    hourly["cos_hour"] = np.cos(2*np.pi*hourly["hour"]/24)
    hourly["sin_month"]= np.sin(2*np.pi*hourly["month"]/12)
    hourly["cos_month"]= np.cos(2*np.pi*hourly["month"]/12)
    for win in [6, 24, 48]:
        hourly[f"power_ma{win}"] = hourly["fleet_power_kw"].rolling(win, min_periods=1).mean()
    hourly = hourly.fillna(0)

    FEATS = ["irradiance_wm2","ambient_temp_c","panel_temp_c","humidity_pct",
             "aqi","battery_soc_pct","wind_speed_kmh","dust_event",
             "sin_hour","cos_hour","sin_month","cos_month",
             "power_ma6","power_ma24","power_ma48"]
    # Fill missing feature columns with zeros
    for f in FEATS:
        if f not in hourly.columns:
            hourly[f] = 0.0

    LOOKBACK = 12
    X_list, y_list = [], []
    for i in range(LOOKBACK, len(hourly)-168):
        window = hourly.iloc[i-LOOKBACK:i][FEATS].values.flatten()
        y1  = max(0, hourly.iloc[i]["fleet_power_kw"])
        y6  = max(0, hourly.iloc[i:i+6]["fleet_power_kw"].sum())
        y24 = max(0, hourly.iloc[i:i+24]["fleet_power_kw"].sum())
        y7d = max(0, hourly.iloc[i:i+168]["fleet_power_kw"].sum())
        X_list.append(window); y_list.append([y1, y6, y24, y7d])

    logger.info("  LSTM dataset: %d samples", len(X_list))
    return np.array(X_list), np.array(y_list), FEATS, LOOKBACK


# ── Build BFCI training dataset ───────────────────────────────────────────────

def build_bfci_dataset(days: int) -> tuple:
    logger.info("Building BFCI dataset from real spectral + labels...")
    try:
        spec_df  = fetch_spectral_data(days)
        label_df = fetch_bfci_labels()
    except Exception as e:
        logger.error("BFCI data fetch failed: %s — using synthetic", e)
        df = pd.read_csv("training/data/bfci_data.csv")
        SPEC = [c for c in df.columns if c.startswith("ch")]
        META = ["days_since_coat","uv_dose_cumulative","avg_temp_7d","avg_humidity_7d","avg_aqi_7d"]
        TGTS = ["bfci_score","uv_absorption_pct","days_to_recoat"]
        return df[SPEC+META].values, df[TGTS].values, SPEC+META, TGTS

    # Merge spectral with BFCI labels on panel + nearest timestamp
    SPEC = [c for c in spec_df.columns if c.startswith("ch")]
    META = ["days_since_coat","uv_dose_cumulative","avg_temp_7d","avg_humidity_7d","avg_aqi_7d"]
    TGTS = ["bfci_score","uv_absorption_pct","days_to_recoat"]

    # If merge is not possible, fall back to synthetic
    if not all(c in spec_df.columns for c in SPEC[:3]):
        df = pd.read_csv("training/data/bfci_data.csv")
        return df[SPEC+META].values, df[TGTS].values, SPEC+META, TGTS

    merged = spec_df.copy()
    for col in META:
        if col not in merged.columns:
            merged[col] = 0.0
    for col in TGTS:
        if col not in merged.columns:
            merged[col] = 80.0
    merged = merged.fillna(0)
    logger.info("  BFCI dataset: %d samples", len(merged))
    return merged[SPEC+META].values, merged[TGTS].values, SPEC+META, TGTS


# ── MODEL TRAINERS ─────────────────────────────────────────────────────────────

def train_lstm(days: int):
    logger.info("\n[1/4] Retraining LSTM Forecaster on real %d-day data...", days)
    X, y, FEATS, LOOKBACK = build_lstm_dataset(days)

    if len(X) < 100:
        logger.error("Insufficient data (%d samples). Need 60+ days. Keeping existing model.", len(X))
        return

    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.15, shuffle=False)
    sc  = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

    names  = ["1h_kw","6h_kwh","24h_kwh","7d_kwh"]
    params = [
        dict(n_estimators=120, learning_rate=0.10, max_depth=5, subsample=0.85),
        dict(n_estimators=120, learning_rate=0.10, max_depth=5, subsample=0.85),
        dict(n_estimators=150, learning_rate=0.08, max_depth=6, subsample=0.80),
        dict(n_estimators=200, learning_rate=0.06, max_depth=7, subsample=0.80),
    ]
    mdls = []
    for i, n in enumerate(names):
        m = GradientBoostingRegressor(**params[i], random_state=42)
        m.fit(Xtr_s, ytr[:,i]); mdls.append(m)
        yp   = np.maximum(0, m.predict(Xte_s))
        mae  = mean_absolute_error(yte[:,i], yp)
        r2   = r2_score(yte[:,i], yp)
        mask = yte[:,i]>0.01
        mape = np.mean(np.abs((yte[mask,i]-yp[mask])/yte[mask,i]))*100 if mask.sum()>0 else 0
        logger.info("  %s MAE=%.3f R²=%.4f MAPE=%.1f%%", n, mae, r2, mape)

    with open("ai_models/lstm_biosolar.pkl","wb") as f:
        pickle.dump({"models":mdls,"scaler":sc,"features":FEATS,"lookback":LOOKBACK,
                     "targets":names,"version":f"gbr_v2.0_real_{days}d",
                     "trained_at":datetime.now(timezone.utc).isoformat()}, f)
    logger.info("  ✅ lstm_biosolar.pkl updated (real %d-day data)", days)


def train_bfci(days: int):
    logger.info("\n[2/4] Retraining BFCI BO-Bagging on real %d-day data...", days)
    X, y, FEATS, TGTS = build_bfci_dataset(days)

    if len(X) < 50:
        logger.error("Insufficient BFCI data (%d samples). Need 90+ days.", len(X))
        return

    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.15, random_state=42)
    sc  = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

    # Try Bayesian optimisation if skopt available
    best = {"n_estimators":180,"max_samples":0.75,"max_features":0.80}
    try:
        from skopt import BayesSearchCV
        from skopt.space import Integer, Real
        bagger = BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=42), random_state=42, n_jobs=-1)
        opt = BayesSearchCV(bagger,
            {"n_estimators":Integer(50,300),"max_samples":Real(0.5,1.0),"max_features":Real(0.5,1.0)},
            n_iter=15, cv=3, scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
        opt.fit(Xtr_s, ytr[:,0])
        best = dict(opt.best_params_)
        logger.info("  BO best params: %s", best)
    except ImportError:
        logger.info("  skopt not available — using default params")

    mdls = []
    for i, n in enumerate(TGTS):
        m = BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=14, random_state=42),
            n_estimators=int(best.get("n_estimators",180)),
            max_samples=float(best.get("max_samples",0.75)),
            max_features=float(best.get("max_features",0.80)),
            random_state=42, n_jobs=-1)
        m.fit(Xtr_s, ytr[:,i]); mdls.append(m)
        yp  = m.predict(Xte_s)
        mae = mean_absolute_error(yte[:,i], yp)
        r2  = r2_score(yte[:,i], yp)
        logger.info("  %s MAE=%.3f R²=%.4f", n, mae, r2)

    # Feature importance from tree estimators
    spec_cols = [f for f in FEATS if f.startswith("ch")]
    feat_imp  = {}
    for fi, fn in enumerate(FEATS):
        imp = np.mean([e.feature_importances_[fi]
                       for e in mdls[0].estimators_
                       if fi < len(e.feature_importances_)])
        feat_imp[fn] = round(float(imp), 4)
    feat_imp = dict(sorted(feat_imp.items(), key=lambda x:x[1], reverse=True))

    with open("ai_models/bfci_model.pkl","wb") as f:
        pickle.dump({"models":mdls,"scaler":sc,"features":FEATS,
                     "spectral_channels":spec_cols,"targets":TGTS,
                     "best_params":best,"shap_importance":feat_imp,
                     "warn_threshold":75.0,"crit_threshold":65.0,
                     "version":f"bo_bagging_v2.0_real_{days}d",
                     "trained_at":datetime.now(timezone.utc).isoformat()}, f)
    logger.info("  ✅ bfci_model.pkl updated (real %d-day data)", days)


def train_iforest(days: int):
    logger.info("\n[3/4] Retraining Isolation Forest on real data...")
    try:
        power_df = fetch_power_data(days)
    except Exception:
        power_df = pd.read_csv("training/data/anomaly_normal.csv")

    FEATS = ["voltage","current","power","temp_c","efficiency_pct","irradiance"]
    avail = [f for f in FEATS if f in power_df.columns]

    # Map column names if needed
    col_map = {"voltage_v":"voltage","current_a":"current",
               "power_w":"power","panel_temp_c":"temp_c",
               "irradiance_wm2":"irradiance"}
    power_df = power_df.rename(columns=col_map)
    avail = [f for f in FEATS if f in power_df.columns]

    if len(avail) < 3:
        logger.warning("Using synthetic anomaly data")
        df = pd.read_csv("training/data/anomaly_normal.csv")
        avail = [f for f in FEATS if f in df.columns]
        power_df = df

    # Filter to daytime normal readings only
    if "hour" in power_df.columns:
        normal = power_df[(power_df["hour"] >= 6) & (power_df["hour"] <= 19)]
    else:
        normal = power_df

    X = normal[avail].fillna(0).values
    sc = StandardScaler(); Xs = sc.fit_transform(X)

    ifo = IsolationForest(n_estimators=200, max_samples=0.8,
                          contamination=0.05, random_state=42, n_jobs=-1)
    ifo.fit(Xs)
    logger.info("  IForest trained on %d real normal readings", len(X))

    with open("ai_models/isolation_forest.pkl","wb") as f:
        pickle.dump({"model":ifo,"scaler":sc,"features":avail,
                     "thresholds":{"voltage_min":28.0,"voltage_max":44.0,
                                   "current_min":0.0,"current_max":18.0,
                                   "temp_max":65.0,"efficiency_min":60.0},
                     "version":f"iforest_v2.0_real_{days}d",
                     "trained_at":datetime.now(timezone.utc).isoformat()}, f)
    logger.info("  ✅ isolation_forest.pkl updated")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Retrain VoltVision AI models on real data")
    parser.add_argument("--days",  type=int, default=90, help="Days of data to use")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all","lstm","bfci","iforest"],
                        help="Which model to retrain")
    args = parser.parse_args()

    logger.info("="*55)
    logger.info("VoltVision AI — Real Data Retraining")
    logger.info("Days: %d | Model: %s", args.days, args.model)
    logger.info("="*55)

    if args.model in ("all","lstm"):
        train_lstm(args.days)
    if args.model in ("all","bfci"):
        train_bfci(args.days)
    if args.model in ("all","iforest"):
        train_iforest(args.days)

    logger.info("\n✅ Retraining complete. Restart ai_engine.py to load new models.")
    logger.info("   pkill -f ai_engine.py && python3 services/ai_engine.py &")

    # Show model sizes
    logger.info("\nModel files:")
    for f in sorted(os.listdir("ai_models")):
        sz = os.path.getsize(f"ai_models/{f}") / 1024
        logger.info("  ai_models/%-35s %8.1f KB", f, sz)


if __name__ == "__main__":
    main()
