"""
VoltVision AI — Synthetic Dataset Generator
Generates 90 days of realistic BioSolar sensor data for training all 4 models.
Based on real Jaipur, Rajasthan solar physics + bio-coating degradation science.

Outputs:
  training/data/power_data.csv       — for LSTM forecasting
  training/data/env_data.csv         — environment readings
  training/data/spectral_data.csv    — AS7265x spectral for BFCI
  training/data/bfci_labels.csv      — BFCI target labels
  training/data/cv_features.csv      — CV feature vectors
  training/data/cv_labels.csv        — CV class labels
  training/data/anomaly_normal.csv   — normal readings for IForest

Run: python3 training/generate_data.py
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

np.random.seed(42)
os.makedirs("training/data", exist_ok=True)

print("=" * 55)
print("VoltVision AI — Synthetic Data Generator")
print("=" * 55)

# ── Site constants (Jaipur, Rajasthan) ───────────────────────────────────────
N_PANELS   = 32
N_DAYS     = 90
INTERVAL_S = 30          # sensor reading every 30 seconds
N_PER_DAY  = 86400 // INTERVAL_S   # 2880 readings/day
N_TOTAL    = N_DAYS * N_PER_DAY    # 259,200 total readings

# Jaipur monthly average irradiance (W/m²)
MONTHLY_IRRADIANCE = {
    1: 580, 2: 650, 3: 720, 4: 750, 5: 710,
    6: 620, 7: 580, 8: 600, 9: 650, 10: 700,
    11: 640, 12: 570
}

# Panel degradation profiles
PANEL_PROFILES = {
    "B4": {"type": "cracked",      "eff_base": 0.72, "bfci_start": 85, "bfci_drop": 1.8},
    "A6": {"type": "dusty",        "eff_base": 0.84, "bfci_start": 80, "bfci_drop": 1.5},
    "D2": {"type": "bio_degraded", "eff_base": 0.80, "bfci_start": 78, "bfci_drop": 1.6},
    "C1": {"type": "dusty",        "eff_base": 0.86, "bfci_start": 88, "bfci_drop": 1.2},
    "C3": {"type": "dusty",        "eff_base": 0.87, "bfci_start": 87, "bfci_drop": 1.1},
    "D8": {"type": "offline",      "eff_base": 0.00, "bfci_start": 60, "bfci_drop": 0.5},
}
DEFAULT_PROFILE = {"type": "clean", "eff_base": 0.924, "bfci_start": 95, "bfci_drop": 0.8}

# All 32 panel labels
PANELS = [f"{chr(65+r)}{c}" for r in range(4) for c in range(1, 9)]

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: solar irradiance at given hour/month
# ─────────────────────────────────────────────────────────────────────────────
def solar_irradiance(hour: int, month: int, noise: float = 0.05) -> float:
    """Gaussian solar bell curve centred at solar noon (13:00 IST for Jaipur)."""
    peak = MONTHLY_IRRADIANCE.get(month, 650)
    if hour < 5 or hour > 19:
        return 0.0
    angle = (hour - 13) / 5.5
    irr   = peak * np.exp(-0.5 * angle ** 2)
    irr   = max(0, irr * (1 + np.random.normal(0, noise)))
    return round(irr, 1)


def panel_power(irr: float, profile: dict, dust_factor: float = 1.0) -> dict:
    """Compute panel electrical output from irradiance and panel state."""
    if irr <= 0 or profile["type"] == "offline":
        return {"voltage": 0.0, "current": 0.0, "power": 0.0, "energy_wh": 0.0}

    eff    = profile["eff_base"] * dust_factor
    stc    = 500.0            # panel STC rating (W)
    voc    = 45.2             # open circuit voltage
    vmpp   = 36.8             # max power point voltage
    impp   = stc / vmpp       # max power point current at STC

    pwr    = irr / 1000.0 * stc * eff * (1 + np.random.normal(0, 0.02))
    pwr    = max(0, pwr)
    volt   = vmpp * (1 + np.random.normal(0, 0.01)) if pwr > 0 else 0
    curr   = pwr / volt if volt > 0 else 0
    energy = pwr * (30 / 3600)   # Wh per 30s interval

    return {
        "voltage":   round(volt, 2),
        "current":   round(curr, 2),
        "power":     round(pwr,  1),
        "energy_wh": round(energy, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  1. POWER + ENVIRONMENT DATA (for LSTM)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Generating power + environment data...")

start_dt  = datetime(2025, 6, 1, 0, 0, 0)
timestamps = [start_dt + timedelta(seconds=i * INTERVAL_S) for i in range(N_TOTAL)]

# Use panel A1 as reference (clean, optimal) + fleet aggregated
records = []
for i, ts in enumerate(timestamps):
    hour  = ts.hour
    month = ts.month
    day   = (ts - start_dt).days

    irr   = solar_irradiance(hour, month)
    lux   = irr * 93.0 + np.random.normal(0, 200)
    lux   = max(0, lux)

    # Ambient conditions — Jaipur summer/monsoon
    if month in [6, 7, 8]:   # monsoon
        temp_amb = np.random.normal(35, 3)
        humidity = np.random.normal(68, 10)
        aqi      = np.random.normal(85, 15)
    else:                      # pre-monsoon / post-monsoon
        temp_amb = np.random.normal(40, 4)
        humidity = np.random.normal(32, 8)
        aqi      = np.random.normal(95, 20)

    # Dust storm events (random, ~5 in 90 days)
    dust_event = 1.0
    if np.random.random() < 0.002:  # 0.2% chance per reading ≈ 5 events
        dust_event = np.random.uniform(0.3, 0.6)

    # Panel surface temp (higher than ambient due to solar heating)
    panel_temp = temp_amb + irr / 100 * 8 + np.random.normal(0, 1.5)
    panel_temp = max(temp_amb, panel_temp)

    # Fleet power (sum of all 32 panels)
    fleet_power = 0
    for label in PANELS:
        prof  = PANEL_PROFILES.get(label, DEFAULT_PROFILE)
        day_dust = max(0.85, 1.0 - day * 0.001) if prof["type"] == "dusty" else 1.0
        pw    = panel_power(irr, prof, dust_factor=day_dust * dust_event)
        fleet_power += pw["power"]

    # Battery SoC (simplified model)
    solar_gen = fleet_power / 1000   # kW
    load      = np.random.uniform(1.5, 3.5)
    net       = solar_gen - load
    soc_base  = 75 + net * 5 + np.random.normal(0, 3)
    soc       = np.clip(soc_base, 20, 100)

    records.append({
        "timestamp":      ts.isoformat(),
        "hour":           hour,
        "day_of_year":    ts.timetuple().tm_yday,
        "month":          month,
        "irradiance_wm2": irr,
        "lux":            round(max(0, lux), 0),
        "ambient_temp_c": round(np.clip(temp_amb, 20, 50), 1),
        "panel_temp_c":   round(np.clip(panel_temp, 20, 75), 1),
        "humidity_pct":   round(np.clip(humidity, 10, 95), 1),
        "aqi":            round(np.clip(aqi, 30, 300), 0),
        "wind_speed_kmh": round(max(0, np.random.normal(12, 5)), 1),
        "battery_soc_pct":round(soc, 1),
        "fleet_power_kw": round(fleet_power / 1000, 3),
        "fleet_energy_kwh":round(fleet_power / 1000 * (30/3600), 5),
        "dust_event":     round(dust_event, 3),
    })

power_df = pd.DataFrame(records)
power_df.to_csv("training/data/power_data.csv", index=False)
print(f"  ✅ power_data.csv — {len(power_df):,} rows")


# ─────────────────────────────────────────────────────────────────────────────
#  2. SPECTRAL + BFCI DATA (for BFCI BO-Bagging model)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Generating spectral + BFCI data...")

# AS7265x channels (wavelengths in nm)
SPECTRAL_CHANNELS = {
    "ch1_410nm":  (410, "violet"),
    "ch2_445nm":  (445, "violet"),
    "ch3_480nm":  (480, "blue"),
    "ch4_515nm":  (515, "blue_green"),
    "ch5_555nm":  (555, "green"),
    "ch6_590nm":  (590, "yellow"),
    "ch7_630nm":  (630, "orange"),
    "ch8_680nm":  (680, "red"),
    "ch9_720nm":  (720, "deep_red"),
    "ch10_760nm": (760, "NIR"),
    "ch11_810nm": (810, "NIR"),
    "ch12_860nm": (860, "NIR"),
    "ch13_900nm": (900, "NIR"),
    "ch14_940nm": (940, "NIR"),
    "ch15_UV1":   (365, "UV"),
    "ch16_UV2":   (340, "UV"),
    "ch17_UV3":   (320, "UV"),
    "ch18_UV4":   (300, "UV"),
}

bfci_records = []
# One reading every 6 hours per panel for 90 days
BFCI_INTERVAL_H = 6
bfci_times = [start_dt + timedelta(hours=i * BFCI_INTERVAL_H)
              for i in range(int(N_DAYS * 24 / BFCI_INTERVAL_H))]

for panel_label in PANELS:
    prof        = PANEL_PROFILES.get(panel_label, DEFAULT_PROFILE)
    bfci_score  = prof["bfci_start"]
    bfci_drop   = prof["bfci_drop"]
    coat_date   = start_dt - timedelta(days=np.random.randint(10, 30))
    days_coated = (start_dt - coat_date).days

    for ts in bfci_times:
        day = (ts - start_dt).days
        days_since_coat = days_coated + day

        # BFCI degrades over time — accelerated by heat, AQI, humidity
        month = ts.month
        heat_factor = 1.2 if month in [4, 5, 6] else 1.0
        aqi_factor  = np.random.uniform(0.9, 1.3)
        daily_drop  = bfci_drop * heat_factor * aqi_factor / (24 / BFCI_INTERVAL_H)
        bfci_score  = max(50, bfci_score - daily_drop + np.random.normal(0, 0.1))

        # Recoating event (simulate every ~75-80 days for low panels)
        if bfci_score < 65 and np.random.random() < 0.05:
            bfci_score  = 95 + np.random.normal(0, 2)
            coat_date   = ts
            days_since_coat = 0

        # UV absorption % — correlated with BFCI score
        uv_absorption = bfci_score * np.random.uniform(0.86, 0.92)

        # Days to recoat prediction
        days_to_recoat = max(0, int((bfci_score - 65) / bfci_drop * (24 / BFCI_INTERVAL_H)))

        # Spectral readings — UV channels degrade as coating degrades
        irr_factor = max(0, solar_irradiance(ts.hour, ts.month) / 800)
        spectral   = {}
        for ch, (wl, band) in SPECTRAL_CHANNELS.items():
            base = irr_factor * np.random.uniform(0.3, 0.9)
            # UV channels absorb more when coating is fresh (high BFCI)
            if "UV" in band:
                base *= (bfci_score / 100) * np.random.uniform(0.8, 1.2)
            # Visible channels relatively stable
            base += np.random.normal(0, 0.02)
            spectral[ch] = round(np.clip(base, 0, 1), 4)

        row = {
            "timestamp":        ts.isoformat(),
            "panel_id":         panel_label,
            "days_since_coat":  days_since_coat,
            "uv_dose_cumulative": round(days_since_coat * 6.4, 1),
            "avg_temp_7d":      round(36 + np.random.normal(0, 3), 1),
            "avg_humidity_7d":  round(45 + np.random.normal(0, 10), 1),
            "avg_aqi_7d":       round(90 + np.random.normal(0, 20), 0),
            **spectral,
            # Labels
            "bfci_score":       round(bfci_score, 2),
            "uv_absorption_pct":round(np.clip(uv_absorption, 0, 100), 2),
            "days_to_recoat":   days_to_recoat,
        }
        bfci_records.append(row)

bfci_df = pd.DataFrame(bfci_records)
bfci_df.to_csv("training/data/bfci_data.csv", index=False)
print(f"  ✅ bfci_data.csv — {len(bfci_df):,} rows × {len(bfci_df.columns)} cols")


# ─────────────────────────────────────────────────────────────────────────────
#  3. CV FEATURE DATA (for MobileNetV3 simulation via feature extraction)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Generating CV feature data...")

CV_CLASSES    = ["clean", "dusty", "cracked", "bio_degraded"]
CV_CLASS_IDX  = {c: i for i, c in enumerate(CV_CLASSES)}
N_CV_SAMPLES  = 2000   # per class

cv_records = []
for cls in CV_CLASSES:
    for _ in range(N_CV_SAMPLES):
        # Simulate image feature vectors (CNN penultimate layer — 128 dims)
        # Each class has a distinct statistical signature
        if cls == "clean":
            base    = np.random.normal(0.72, 0.08, 128)
            # High uniform reflectance across visible spectrum
            base[:32] += 0.15   # visible channels bright
            base[96:]  -= 0.05  # UV channels normal
        elif cls == "dusty":
            base    = np.random.normal(0.55, 0.12, 128)
            # Reduced reflectance, dust scatters UV
            base[:32] -= 0.12   # visible reduced
            base[96:]  -= 0.18  # UV absorbed by dust particles
        elif cls == "cracked":
            base    = np.random.normal(0.48, 0.18, 128)
            # Irregular reflectance pattern, dark shadows from cracks
            base[:32] -= 0.20
            base[32:64] += np.random.normal(0, 0.25, 32)  # high variance = cracks
            base[64:96] -= 0.15
        elif cls == "bio_degraded":
            base    = np.random.normal(0.60, 0.10, 128)
            # Green/yellow tint from algae, specific UV signature
            base[16:32] += 0.18  # green channel boost (algae pigment)
            base[96:]   += 0.12  # UV channels boosted (fluorescence)

        base = np.clip(base, 0, 1)
        row  = {f"f{i:03d}": round(float(v), 4) for i, v in enumerate(base)}
        row["label"]     = cls
        row["label_idx"] = CV_CLASS_IDX[cls]
        # Add simulated confidence/quality metrics
        row["brightness"] = round(float(np.mean(base[:32])), 4)
        row["contrast"]   = round(float(np.std(base)), 4)
        row["uniformity"] = round(float(1 - np.std(base[:64])), 4)
        cv_records.append(row)

cv_df = pd.DataFrame(cv_records).sample(frac=1, random_state=42).reset_index(drop=True)
cv_df.to_csv("training/data/cv_data.csv", index=False)
print(f"  ✅ cv_data.csv — {len(cv_df):,} rows × {len(cv_df.columns)} cols ({N_CV_SAMPLES} per class)")


# ─────────────────────────────────────────────────────────────────────────────
#  4. ANOMALY DATA (for Isolation Forest)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Generating anomaly detection data...")

normal_records  = []
anomaly_records = []

for i in range(20000):
    hour = np.random.randint(0, 24)
    irr  = solar_irradiance(hour, 7)

    # Normal operating range
    if irr > 0:
        v    = np.random.normal(36.8, 0.8)
        curr = np.random.normal(13.2, 0.5)
        temp = np.random.normal(44.0, 2.5)
        eff  = np.random.normal(92.0, 1.5)
    else:
        v = curr = 0.0
        temp = np.random.normal(28.0, 3.0)
        eff  = 0.0

    normal_records.append({
        "voltage": round(np.clip(v, 0, 50), 2),
        "current": round(np.clip(curr, 0, 20), 2),
        "power":   round(max(0, v * curr), 1),
        "temp_c":  round(np.clip(temp, 15, 75), 1),
        "efficiency_pct": round(np.clip(eff, 0, 100), 1),
        "irradiance": irr,
        "is_anomaly": 0,
    })

# Inject anomaly types
for i in range(2000):
    anomaly_type = np.random.choice(["voltage_drop", "overtemp", "open_circuit", "shading"])
    hour = np.random.randint(7, 17)  # anomalies during daylight
    irr  = solar_irradiance(hour, 7)

    if anomaly_type == "voltage_drop":
        v    = np.random.uniform(18, 28)   # abnormally low
        curr = np.random.normal(13.2, 0.5)
        temp = np.random.normal(44.0, 2.5)
        eff  = v * curr / (irr / 1000 * 500) * 100 if irr > 0 else 0
    elif anomaly_type == "overtemp":
        v    = np.random.normal(34.0, 1.0)  # slightly reduced due to heat
        curr = np.random.normal(12.8, 0.5)
        temp = np.random.uniform(65, 85)    # dangerously hot
        eff  = np.random.uniform(60, 75)
    elif anomaly_type == "open_circuit":
        v    = np.random.uniform(40, 46)   # high voltage, no current
        curr = np.random.uniform(0, 0.5)
        temp = np.random.normal(44.0, 2.5)
        eff  = 0.0
    else:  # shading
        v    = np.random.normal(36.8, 0.8)
        curr = np.random.uniform(2, 7)     # very low current
        temp = np.random.normal(38.0, 2.0)
        eff  = np.random.uniform(20, 45)

    anomaly_records.append({
        "voltage":  round(max(0, v), 2),
        "current":  round(max(0, curr), 2),
        "power":    round(max(0, v * curr), 1),
        "temp_c":   round(np.clip(temp, 15, 90), 1),
        "efficiency_pct": round(np.clip(eff, 0, 100), 1),
        "irradiance": irr,
        "is_anomaly": 1,
    })

anomaly_df = pd.DataFrame(normal_records + anomaly_records).sample(frac=1, random_state=42)
anomaly_df.to_csv("training/data/anomaly_data.csv", index=False)
normal_only = anomaly_df[anomaly_df["is_anomaly"] == 0].drop(columns=["is_anomaly"])
normal_only.to_csv("training/data/anomaly_normal.csv", index=False)
print(f"  ✅ anomaly_data.csv — {len(anomaly_df):,} rows (normal + anomaly)")
print(f"  ✅ anomaly_normal.csv — {len(normal_only):,} normal rows (for IForest training)")


# ─────────────────────────────────────────────────────────────────────────────
#  5. LSTM SEQUENCE DATA (sliding window for time-series forecasting)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Building LSTM sequence dataset...")

# Resample power_df to hourly for LSTM (one row per hour)
power_df["timestamp"] = pd.to_datetime(power_df["timestamp"])
hourly = power_df.set_index("timestamp").resample("1h").agg({
    "irradiance_wm2":  "mean",
    "ambient_temp_c":  "mean",
    "panel_temp_c":    "mean",
    "humidity_pct":    "mean",
    "aqi":             "mean",
    "battery_soc_pct": "mean",
    "fleet_power_kw":  "sum",
    "fleet_energy_kwh":"sum",
    "hour":            "first",
    "month":           "first",
    "wind_speed_kmh":  "mean",
    "dust_event":      "min",
}).reset_index()

hourly.to_csv("training/data/lstm_hourly.csv", index=False)
print(f"  ✅ lstm_hourly.csv — {len(hourly):,} hourly rows ({N_DAYS} days × 24h)")

# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("GENERATION COMPLETE")
print("=" * 55)
for f in os.listdir("training/data"):
    size = os.path.getsize(f"training/data/{f}") / 1024
    rows = sum(1 for _ in open(f"training/data/{f}")) - 1
    print(f"  {f:<30} {rows:>8,} rows   {size:>8.1f} KB")
print("\nNext: python3 training/train_all_models.py")
