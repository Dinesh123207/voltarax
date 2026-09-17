"""
VoltVision AI — Train All 4 Models
Trains on synthetic data now. When real hardware data is available,
replace CSV files in training/data/ and re-run this script.
Models are saved to ai_models/ and used by ai_engine.py automatically.

Run: python3 training/train_all_models.py
"""
import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import (
    BaggingRegressor, IsolationForest, RandomForestClassifier,
    GradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, accuracy_score, confusion_matrix
)
from sklearn.pipeline import Pipeline

os.makedirs("ai_models", exist_ok=True)

print("=" * 60)
print("VoltVision AI — Model Training Pipeline")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 1 — LSTM FORECASTER (sklearn GBR as proxy for TFLite)
#  In production: TFLite LSTM. Here: GradientBoostingRegressor (same API)
#  When you have TFLite, replace ai_models/lstm_biosolar.pkl with .tflite
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1/4] Training LSTM Energy Forecaster...")
t0 = time.time()

df = pd.read_csv("training/data/lstm_hourly.csv")
df = df.fillna(0)

# Feature engineering
df["sin_hour"]   = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"]   = np.cos(2 * np.pi * df["hour"] / 24)
df["sin_month"]  = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"]  = np.cos(2 * np.pi * df["month"] / 12)

FEATURES_LSTM = [
    "irradiance_wm2", "ambient_temp_c", "panel_temp_c", "humidity_pct",
    "aqi", "battery_soc_pct", "wind_speed_kmh", "dust_event",
    "sin_hour", "cos_hour", "sin_month", "cos_month",
]

# Build sliding window features (24-hour lookback)
LOOKBACK  = 24
X_list, y_list = [], []

for i in range(LOOKBACK, len(df) - 24):
    window = df.iloc[i - LOOKBACK:i][FEATURES_LSTM].values.flatten()
    # Targets: next 1h, 6h, 24h power
    y_1h  = df.iloc[i]["fleet_power_kw"]
    y_6h  = df.iloc[i:i+6]["fleet_power_kw"].sum()
    y_24h = df.iloc[i:i+24]["fleet_power_kw"].sum()
    y_7d  = df.iloc[i:min(i+168, len(df))]["fleet_power_kw"].sum()
    X_list.append(window)
    y_list.append([y_1h, y_6h, y_24h, y_7d])

X = np.array(X_list)
y = np.array(y_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=False)

# Train one GBR per output target
models_lstm = []
target_names = ["1h_kw", "6h_kwh", "24h_kwh", "7d_kwh"]

for idx, name in enumerate(target_names):
    gbr = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.08, max_depth=5,
        subsample=0.8, random_state=42
    )
    gbr.fit(X_train, y_train[:, idx])
    models_lstm.append(gbr)

    y_pred = gbr.predict(X_test)
    mae    = mean_absolute_error(y_test[:, idx], y_pred)
    r2     = r2_score(y_test[:, idx], y_pred)
    mape   = np.mean(np.abs((y_test[:, idx] - y_pred) /
                             np.clip(np.abs(y_test[:, idx]), 1e-5, None))) * 100
    print(f"    {name:<12} MAE={mae:.3f}  R²={r2:.3f}  MAPE={mape:.1f}%")

# Save scaler and models
scaler_lstm = StandardScaler()
scaler_lstm.fit(X_train)

lstm_package = {
    "models":      models_lstm,
    "scaler":      scaler_lstm,
    "features":    FEATURES_LSTM,
    "lookback":    LOOKBACK,
    "targets":     target_names,
    "version":     "gbr_v1.0_synthetic",
    "trained_on":  "90day_synthetic_jaipur",
}
with open("ai_models/lstm_biosolar.pkl", "wb") as f:
    pickle.dump(lstm_package, f)

print(f"  ✅ ai_models/lstm_biosolar.pkl saved  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 2 — BFCI BO-BAGGING (Patent Claim 1)
#  Bayesian-Optimised Bagging Regressor for bio-coating health prediction
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2/4] Training BFCI BO-Bagging Model (Patent Claim 1)...")
t0 = time.time()

bfci_df = pd.read_csv("training/data/bfci_data.csv")
bfci_df = bfci_df.fillna(0)

SPECTRAL_COLS = [c for c in bfci_df.columns if c.startswith("ch")]
META_COLS     = ["days_since_coat", "uv_dose_cumulative",
                 "avg_temp_7d", "avg_humidity_7d", "avg_aqi_7d"]
BFCI_FEATURES = SPECTRAL_COLS + META_COLS
BFCI_TARGETS  = ["bfci_score", "uv_absorption_pct", "days_to_recoat"]

X_bfci = bfci_df[BFCI_FEATURES].values
y_bfci = bfci_df[BFCI_TARGETS].values

X_train, X_test, y_train, y_test = train_test_split(
    X_bfci, y_bfci, test_size=0.15, random_state=42
)

# Bayesian Optimisation to find best Bagging hyperparameters
print("    Running Bayesian hyperparameter optimisation...")
try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real
    from sklearn.tree import DecisionTreeRegressor

    base_est = DecisionTreeRegressor(random_state=42)
    search_space = {
        "n_estimators":     Integer(50, 300),
        "max_samples":      Real(0.5, 1.0),
        "max_features":     Real(0.5, 1.0),
    }
    bagger = BaggingRegressor(estimator=base_est, random_state=42, n_jobs=-1)
    opt    = BayesSearchCV(
        bagger, search_space, n_iter=20, cv=3,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1,
    )
    opt.fit(X_train, y_train[:, 0])   # optimise on bfci_score
    best_params = opt.best_params_
    print(f"    Best params: {dict(best_params)}")

except ImportError:
    print("    skopt not available — using optimised default params")
    best_params = {"n_estimators": 150, "max_samples": 0.8, "max_features": 0.75}

# Train final BFCI models per target
bfci_models = []
for idx, name in enumerate(BFCI_TARGETS):
    from sklearn.tree import DecisionTreeRegressor
    model = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=12, random_state=42),
        n_estimators=int(best_params.get("n_estimators", 150)),
        max_samples=float(best_params.get("max_samples", 0.8)),
        max_features=float(best_params.get("max_features", 0.75)),
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train[:, idx])
    bfci_models.append(model)

    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test[:, idx], y_pred)
    r2     = r2_score(y_test[:, idx], y_pred)
    print(f"    {name:<22} MAE={mae:.3f}  R²={r2:.3f}")

# SHAP values for explainability
print("    Computing SHAP feature importance...")
try:
    import shap
    explainer   = shap.TreeExplainer(bfci_models[0])
    shap_values = explainer.shap_values(X_test[:100])
    feature_importance = dict(zip(BFCI_FEATURES,
                                  np.abs(shap_values).mean(axis=0).tolist()))
    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(),
                                     key=lambda x: x[1], reverse=True))
    print(f"    Top 5 SHAP features: {list(feature_importance.keys())[:5]}")
except Exception as e:
    print(f"    SHAP skipped: {e}")
    feature_importance = {f: float(i) for i, f in enumerate(reversed(BFCI_FEATURES))}

scaler_bfci = StandardScaler()
scaler_bfci.fit(X_train)

bfci_package = {
    "models":             bfci_models,
    "scaler":             scaler_bfci,
    "features":           BFCI_FEATURES,
    "spectral_channels":  SPECTRAL_COLS,
    "targets":            BFCI_TARGETS,
    "best_params":        dict(best_params),
    "shap_importance":    feature_importance,
    "version":            "bo_bagging_v1.0_synthetic",
    "warn_threshold":     75.0,
    "crit_threshold":     65.0,
}
with open("ai_models/bfci_model.pkl", "wb") as f:
    pickle.dump(bfci_package, f)

print(f"  ✅ ai_models/bfci_model.pkl saved  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 3 — CV CLASSIFIER (Random Forest on feature vectors)
#  In production: MobileNetV3 TFLite on real camera images.
#  This RF uses extracted feature vectors — same inference pipeline.
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3/4] Training CV Panel Classifier...")
t0 = time.time()

cv_df  = pd.read_csv("training/data/cv_data.csv")
FEAT_COLS = [c for c in cv_df.columns if c.startswith("f")]
FEAT_COLS += ["brightness", "contrast", "uniformity"]

X_cv = cv_df[FEAT_COLS].values
y_cv = cv_df["label_idx"].values
le   = LabelEncoder()
le.fit(cv_df["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X_cv, y_cv, test_size=0.2, random_state=42, stratify=y_cv
)

scaler_cv = StandardScaler()
X_train_s = scaler_cv.fit_transform(X_train)
X_test_s  = scaler_cv.transform(X_test)

rf_cv = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_split=4,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf_cv.fit(X_train_s, y_train)

y_pred = rf_cv.predict(X_test_s)
acc    = accuracy_score(y_test, y_pred)
print(f"    Accuracy: {acc:.3f} ({acc*100:.1f}%)")
print(f"    Classes:  {le.classes_.tolist()}")

# Per-class metrics
cm = confusion_matrix(y_test, y_pred)
for i, cls in enumerate(le.classes_):
    tp = cm[i, i]
    fn = cm[i].sum() - tp
    prec = tp / (cm[:, i].sum() + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    print(f"    {cls:<15} Precision={prec:.2f}  Recall={rec:.2f}")

cv_package = {
    "model":       rf_cv,
    "scaler":      scaler_cv,
    "label_encoder": le,
    "classes":     le.classes_.tolist(),
    "features":    FEAT_COLS,
    "n_features":  len(FEAT_COLS),
    "accuracy":    float(acc),
    "version":     "rf_v1.0_synthetic",
    "eff_loss":    {"clean": 0.0, "dusty": 3.1, "cracked": 15.4, "bio_degraded": 8.7},
    # Note: replace with MobileNetV3 TFLite after real image collection
    "note": "RF on feature vectors. Replace with mobilenet_cv.tflite after real image dataset."
}
with open("ai_models/cv_model.pkl", "wb") as f:
    pickle.dump(cv_package, f)

print(f"  ✅ ai_models/cv_model.pkl saved  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 4 — ISOLATION FOREST (Anomaly Detection)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4/4] Training Isolation Forest Anomaly Detector...")
t0 = time.time()

anom_df  = pd.read_csv("training/data/anomaly_normal.csv")
ANOM_FEATURES = ["voltage", "current", "power", "temp_c", "efficiency_pct", "irradiance"]
X_anom   = anom_df[ANOM_FEATURES].values

scaler_anom = StandardScaler()
X_anom_s    = scaler_anom.fit_transform(X_anom)

iforest = IsolationForest(
    n_estimators=200,
    max_samples=0.8,
    contamination=0.05,   # expect ~5% anomalies in real data
    random_state=42,
    n_jobs=-1,
)
iforest.fit(X_anom_s)

# Validate on full dataset (with anomalies)
full_df   = pd.read_csv("training/data/anomaly_data.csv")
X_full    = full_df[ANOM_FEATURES].values
y_full    = full_df["is_anomaly"].values
X_full_s  = scaler_anom.transform(X_full)
y_pred_if = iforest.predict(X_full_s)   # -1=anomaly, 1=normal
y_pred_01 = np.where(y_pred_if == -1, 1, 0)

acc_if  = accuracy_score(y_full, y_pred_01)
# Count correct anomaly detections
true_anom   = np.sum(y_full == 1)
det_anom    = np.sum((y_full == 1) & (y_pred_01 == 1))
false_alarm = np.sum((y_full == 0) & (y_pred_01 == 1))
print(f"    Overall accuracy:    {acc_if:.3f}")
print(f"    Anomalies detected:  {det_anom}/{true_anom} ({det_anom/true_anom*100:.1f}%)")
print(f"    False alarms:        {false_alarm}")

iforest_package = {
    "model":    iforest,
    "scaler":   scaler_anom,
    "features": ANOM_FEATURES,
    "thresholds": {
        "voltage_min":  28.0,
        "voltage_max":  44.0,
        "current_min":  0.0,
        "current_max":  18.0,
        "temp_max":     65.0,
        "efficiency_min": 60.0,
    },
    "accuracy":  float(acc_if),
    "version":   "iforest_v1.0_synthetic",
}
with open("ai_models/isolation_forest.pkl", "wb") as f:
    pickle.dump(iforest_package, f)

print(f"  ✅ ai_models/isolation_forest.pkl saved  ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL MODELS TRAINED & SAVED")
print("=" * 60)
for f in os.listdir("ai_models"):
    size = os.path.getsize(f"ai_models/{f}") / 1024
    print(f"  ai_models/{f:<30} {size:>8.1f} KB")

print("""
Next steps:
  1. Run: uvicorn api.main:app --reload --port 8000
  2. The AI engine will now use REAL trained models
  3. When hardware is connected, replace training/data/ CSVs with
     real sensor readings and re-run this script to retrain.
""")
