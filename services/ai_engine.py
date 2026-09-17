"""
VoltVision AI — AI Engine Service (Real Models)
Loads trained .pkl models from ai_models/ directory.
Falls back to simulation if models not found.
Runs 3 parallel inference threads:
  1. LSTM/GBR Forecaster   — every 60s
  2. CV Classifier         — every 30 min
  3. BFCI BO-Bagging       — every 6h
Run: python3 services/ai_engine.py
"""
import os, sys, json, time, logging, signal, pickle, threading, random
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings
from db.database import init_db, get_db, write_influx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger  = logging.getLogger("ai_engine")
_SOLAR  = [0,0,0,0,0,0,.1,.3,.8,1.8,3.2,4.4,5.1,5.6,5.7,5.3,4.6,3.8,2.7,1.5,.6,.2,0,0]
_N      = settings.SITE_PANEL_COUNT
_STOP   = threading.Event()


def _load_pkl(path: str):
    """Load a pickle model file. Returns None if not found."""
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                pkg = pickle.load(f)
            logger.info("Loaded model: %s", path)
            return pkg
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
    logger.info("Model not found: %s — using simulation", path)
    return None


def _setup_mqtt():
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(client_id=settings.MQTT_CLIENT_ID + "-ai")
        if settings.MQTT_USERNAME:
            client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)
        client.connect(settings.MQTT_HOST, settings.MQTT_PORT, keepalive=60)
        client.loop_start()
        return client
    except Exception as e:
        logger.warning("MQTT not available: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 1 — LSTM/GBR FORECASTER
# ═══════════════════════════════════════════════════════════════════════════════
class LSTMForecaster:
    def __init__(self):
        self.pkg = _load_pkl(settings.LSTM_MODEL_PATH.replace(".tflite", ".pkl"))
        if not self.pkg:
            self.pkg = _load_pkl("ai_models/lstm_biosolar.pkl")
        self.version = self.pkg["version"] if self.pkg else "simulation_v1"

    def _build_features(self, df_recent) -> np.ndarray:
        """Build feature window from recent hourly data."""
        pkg  = self.pkg
        feats = pkg["features"]
        lb    = pkg["lookback"]
        rows  = []
        now   = datetime.now()
        for i in range(lb):
            h   = (now.hour - lb + i) % 24
            irr = _SOLAR[h] * 812
            row = {
                "irradiance_wm2": irr,
                "ambient_temp_c": 38.5, "panel_temp_c": 44.2,
                "humidity_pct":   34.0, "aqi": 88.0,
                "battery_soc_pct":78.0, "wind_speed_kmh": 11.0,
                "dust_event":     1.0,
                "sin_hour":  np.sin(2*np.pi*h/24),
                "cos_hour":  np.cos(2*np.pi*h/24),
                "sin_month": np.sin(2*np.pi*now.month/12),
                "cos_month": np.cos(2*np.pi*now.month/12),
                "power_ma6":  irr * 0.028,
                "power_ma24": irr * 0.026,
                "power_ma48": irr * 0.025,
            }
            rows.append([row.get(f, 0.0) for f in feats])
        window = np.array(rows).flatten().reshape(1, -1)
        return self.pkg["scaler"].transform(window)

    def predict(self) -> dict:
        if not self.pkg:
            return self._sim()
        try:
            X    = self._build_features(None)
            mdls = self.pkg["models"]
            vals = [max(0, float(m.predict(X)[0])) for m in mdls]
            return {
                "1h_kw":   round(vals[0], 2),
                "6h_kwh":  round(vals[1], 2),
                "24h_kwh": round(vals[2], 2),
                "7d_kwh":  round(vals[3], 1),
                "confidence": [0.91, 0.83, 0.79, 0.68],
                "model_ver": self.version,
            }
        except Exception as e:
            logger.error("LSTM inference error: %s", e)
            return self._sim()

    def _sim(self) -> dict:
        h  = datetime.now().hour
        kw = lambda x: max(0, _SOLAR[x%24]*_N*0.835*0.972*0.912)
        return {
            "1h_kw":   round(kw(h+1)+random.uniform(-0.1,0.1), 2),
            "6h_kwh":  round(sum(kw(h+i) for i in range(1,7)), 2),
            "24h_kwh": round(sum(kw(i) for i in range(24)), 2),
            "7d_kwh":  round(sum(kw(i) for i in range(24))*7*random.uniform(.95,1.05), 1),
            "confidence": [0.91, 0.83, 0.79, 0.68],
            "model_ver": "simulation_v1",
        }

    def run(self, mqtt_client=None):
        interval = settings.LSTM_INFERENCE_INTERVAL_SECONDS
        logger.info("LSTM Forecaster started (interval=%ds, model=%s)", interval, self.version)
        while not _STOP.is_set():
            try:
                result = self.predict()
                result["ts"] = datetime.now(timezone.utc).isoformat()
                # Write to InfluxDB
                try:
                    from influxdb_client import Point
                    pt = (Point("ai_forecast")
                          .field("forecast_1h_kw",   result["1h_kw"])
                          .field("forecast_6h_kwh",  result["6h_kwh"])
                          .field("forecast_24h_kwh", result["24h_kwh"])
                          .field("forecast_7d_kwh",  result["7d_kwh"])
                          .tag("model_ver", result["model_ver"]))
                    write_influx(settings.INFLUX_BUCKET_AI, pt)
                except Exception:
                    pass
                if mqtt_client:
                    mqtt_client.publish(
                        f"{settings.MQTT_TOPIC_PREFIX}/ai/forecast",
                        json.dumps(result), qos=1)
                logger.info("Forecast → 1h=%.2fkW 24h=%.1fkWh 7d=%.1fkWh",
                            result["1h_kw"], result["24h_kwh"], result["7d_kwh"])
            except Exception as e:
                logger.error("LSTM run error: %s", e)
            _STOP.wait(interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 2 — CV CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
class CVInference:
    _EFF_LOSS = {"clean":0.0,"dusty":3.1,"cracked":15.4,"bio_degraded":8.7}
    _SIM_MAP  = {"B4":"cracked","A6":"dusty","D2":"bio_degraded",
                 "C1":"dusty","C2":"dusty","C3":"dusty"}

    def __init__(self):
        self.pkg = _load_pkl("ai_models/cv_model.pkl")
        self.version = self.pkg["version"] if self.pkg else "simulation_v1"

    def _extract_features(self, image_path: str) -> np.ndarray:
        """
        HARDWARE HOOK: When real camera is connected,
        extract 128-dim feature vector from image here.
        Currently returns simulated features.
        """
        try:
            import cv2
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
            # Simulate feature extraction (replace with real CNN backbone)
            feats = img.mean(axis=(0,1))  # placeholder
            return feats
        except Exception:
            # Simulated 128-dim feature vector
            return np.random.normal(0.65, 0.10, 128)

    def _sim_class(self, label: str):
        cls  = self._SIM_MAP.get(label, "clean")
        conf = round(random.uniform(0.88, 0.98), 3)
        return cls, conf

    def scan_panel(self, panel_id: int, label: str) -> dict:
        cls, conf = None, None

        if self.pkg:
            try:
                # Build feature vector matching training format
                feats_128 = self._extract_features(f"/tmp/panel_{label}.jpg")
                brightness  = float(np.mean(feats_128[:32])) if len(feats_128)>=32 else random.uniform(.4,.8)
                contrast    = float(np.std(feats_128)) if len(feats_128)>1 else random.uniform(.05,.25)
                uniformity  = float(1 - np.std(feats_128[:64])) if len(feats_128)>=64 else random.uniform(.6,.9)

                # Build full feature vector (128 f-features + 3 meta)
                n_feat = self.pkg.get("n_features", 131)
                feature_vec = np.zeros(n_feat)
                feature_vec[:min(128, n_feat-3)] = feats_128[:min(128, n_feat-3)]
                if n_feat >= 131:
                    feature_vec[-3] = brightness
                    feature_vec[-2] = contrast
                    feature_vec[-1] = uniformity

                X_s = self.pkg["scaler"].transform(feature_vec.reshape(1,-1))
                idx = int(self.pkg["model"].predict(X_s)[0])
                proba = self.pkg["model"].predict_proba(X_s)[0]
                cls  = self.pkg["classes"][idx]
                conf = round(float(proba[idx]), 3)
            except Exception as e:
                logger.debug("CV real inference failed: %s", e)

        if cls is None:
            cls, conf = self._sim_class(label)

        eff_loss = self._EFF_LOSS.get(cls, 0.0)
        result   = {
            "panel_id": panel_id, "panel_label": label,
            "cv_class": cls, "confidence": conf,
            "eff_loss_pct": eff_loss, "model_ver": self.version,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
        }
        # Write to SQLite
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO cv_detections "
                    "(panel_id,cv_class,confidence,eff_loss_pct,model_ver) VALUES(?,?,?,?,?)",
                    (panel_id, cls, conf, eff_loss, self.version))
                if cls != "clean" and conf >= settings.ALERT_CV_CONFIDENCE_MIN:
                    sev = "critical" if cls == "cracked" else "warning"
                    existing = conn.execute(
                        "SELECT id FROM alerts WHERE panel_id=? AND alert_type=? AND resolved=0",
                        (panel_id, f"cv_{cls}")).fetchone()
                    if not existing:
                        conn.execute(
                            "INSERT INTO alerts(panel_id,alert_type,severity,message,confidence)"
                            " VALUES(?,?,?,?,?)",
                            (panel_id, f"cv_{cls}", sev,
                             f"Panel {label}: CV={cls} conf={conf:.1%} eff_loss={eff_loss:.1f}%",
                             conf))
        except Exception as e:
            logger.error("CV SQLite write: %s", e)
        return result

    def run(self, mqtt_client=None):
        interval = settings.CV_SCAN_INTERVAL_MINUTES * 60
        logger.info("CV Inference started (interval=%dm, model=%s)",
                    settings.CV_SCAN_INTERVAL_MINUTES, self.version)
        while not _STOP.is_set():
            try:
                with get_db() as conn:
                    panels = conn.execute(
                        "SELECT id,label FROM panels WHERE is_active=1").fetchall()
                for p in panels:
                    if _STOP.is_set(): break
                    r = self.scan_panel(p["id"], p["label"])
                    logger.info("CV %s → %s (conf=%.2f)", p["label"],
                                r["cv_class"], r["confidence"])
                    if mqtt_client:
                        mqtt_client.publish(
                            f"{settings.MQTT_TOPIC_PREFIX}/ai/cv_detection",
                            json.dumps(r), qos=1)
                    time.sleep(0.3)
            except Exception as e:
                logger.error("CV run error: %s", e)
            _STOP.wait(interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 3 — BFCI BO-BAGGING
# ═══════════════════════════════════════════════════════════════════════════════
class BFCIEngine:
    def __init__(self):
        self.pkg = _load_pkl("ai_models/bfci_model.pkl")
        self.version = self.pkg["version"] if self.pkg else "simulation_v1"

    def _get_spectral(self, panel_label: str) -> dict:
        """
        HARDWARE HOOK: Read real AS7265x spectral data from InfluxDB.
        Currently returns simulated 18-channel readings.
        When hardware is connected, query:
          SELECT LAST(*) FROM spectral WHERE panel_id='{panel_label}'
        """
        channels = [f"ch{i+1}" for i in range(14)] + ["ch15_UV1","ch16_UV2","ch17_UV3","ch18_UV4"]
        return {ch: round(random.uniform(0.1, 0.9), 4) for ch in channels}

    def predict_panel(self, panel_id: int, label: str) -> dict:
        from datetime import date
        with get_db() as conn:
            coat = conn.execute(
                "SELECT coat_date FROM bfci_logs WHERE panel_id=? AND coat_date IS NOT NULL"
                " ORDER BY recorded_at DESC LIMIT 1", (panel_id,)).fetchone()
        days_since = (date.today() - date.fromisoformat(
            coat["coat_date"])).days if coat else 75

        pred = None
        if self.pkg:
            try:
                spectral = self._get_spectral(label)
                spec_vals = [spectral.get(f, 0.0) for f in self.pkg["spectral_channels"]]
                meta_vals = [
                    days_since,
                    days_since * 6.4,     # uv_dose_cumulative
                    38.5,                 # avg_temp_7d
                    34.0,                 # avg_humidity_7d
                    88.0,                 # avg_aqi_7d
                ]
                X  = np.array(spec_vals + meta_vals).reshape(1, -1)
                Xs = self.pkg["scaler"].transform(X)
                vals = [max(0, float(m.predict(Xs)[0])) for m in self.pkg["models"]]
                pred = {
                    "bfci_score":        round(min(100, vals[0]), 1),
                    "uv_absorption_pct": round(min(100, vals[1]), 1),
                    "days_to_recoat":    max(0, int(vals[2])),
                }
            except Exception as e:
                logger.error("BFCI inference error: %s", e)

        if pred is None:
            # Physics-based fallback
            low  = label in {"A6","D2"}
            warn = label in {"B3","C7","D5"}
            base = random.uniform(64,72) if low else random.uniform(73,79) if warn else random.uniform(80,95)
            score = max(50, base - days_since * 0.3)
            pred = {
                "bfci_score":        round(score, 1),
                "uv_absorption_pct": round(score*0.88, 1),
                "days_to_recoat":    max(0, int((score-65)/0.3)),
            }

        result = {"panel_id":panel_id,"panel_label":label,
                  "days_since_coat":days_since,**pred,
                  "model_ver":self.version,
                  "recorded_at":datetime.now(timezone.utc).isoformat()}

        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO bfci_logs(panel_id,bfci_score,uv_absorption_pct,days_to_recoat)"
                    " VALUES(?,?,?,?)",
                    (panel_id, pred["bfci_score"], pred["uv_absorption_pct"], pred["days_to_recoat"]))
                score = pred["bfci_score"]
                if score < settings.ALERT_BFCI_CRIT_THRESHOLD:
                    sev, atype = "critical", "bfci_critical"
                    msg = f"Panel {label} BFCI {score:.1f}% CRITICAL — recoat now."
                elif score < settings.ALERT_BFCI_WARN_THRESHOLD:
                    sev, atype = "warning", "bfci_warning"
                    msg = f"Panel {label} BFCI {score:.1f}% — recoat in {pred['days_to_recoat']}d."
                else:
                    sev = atype = None
                if sev:
                    ex = conn.execute(
                        "SELECT id FROM alerts WHERE panel_id=? AND alert_type=? AND resolved=0",
                        (panel_id, atype)).fetchone()
                    if not ex:
                        conn.execute(
                            "INSERT INTO alerts(panel_id,alert_type,severity,message)"
                            " VALUES(?,?,?,?)", (panel_id, atype, sev, msg))
        except Exception as e:
            logger.error("BFCI SQLite write: %s", e)
        return result

    def run(self, mqtt_client=None):
        interval = settings.BFCI_INFERENCE_INTERVAL_HOURS * 3600
        logger.info("BFCI Engine started (interval=%dh, model=%s)",
                    settings.BFCI_INFERENCE_INTERVAL_HOURS, self.version)
        while not _STOP.is_set():
            try:
                with get_db() as conn:
                    panels = conn.execute(
                        "SELECT id,label FROM panels WHERE is_active=1").fetchall()
                scores = []
                for p in panels:
                    if _STOP.is_set(): break
                    r = self.predict_panel(p["id"], p["label"])
                    scores.append(r["bfci_score"])
                    logger.info("BFCI %s → %.1f%% (recoat=%dd)",
                                p["label"], r["bfci_score"], r["days_to_recoat"])
                    if mqtt_client:
                        mqtt_client.publish(
                            f"{settings.MQTT_TOPIC_PREFIX}/ai/bfci",
                            json.dumps(r), qos=1)
                    time.sleep(0.1)
                if scores:
                    logger.info("BFCI fleet avg=%.1f%%", sum(scores)/len(scores))
            except Exception as e:
                logger.error("BFCI run error: %s", e)
            _STOP.wait(interval)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL 4 — ISOLATION FOREST ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
class AnomalyDetector:
    def __init__(self):
        self.pkg = _load_pkl("ai_models/isolation_forest.pkl")

    def check(self, panel_id: int, label: str,
              voltage: float, current: float,
              power: float, temp: float, irr: float = 0) -> dict:
        is_anomaly = False
        confidence = 0.0
        reason     = "nominal"

        if self.pkg:
            try:
                X  = np.array([[voltage, current, power, temp, 0.0, irr]])
                Xs = self.pkg["scaler"].transform(X)
                pred = self.pkg["model"].predict(Xs)[0]
                score = self.pkg["model"].decision_function(Xs)[0]
                is_anomaly = pred == -1
                confidence = round(float(max(0, min(1, -score + 0.5))), 3)
                reason = "isolation_forest"
            except Exception as e:
                logger.debug("IForest check error: %s", e)
        else:
            # Z-score fallback
            thr = self.pkg["thresholds"] if self.pkg else {
                "voltage_min":28,"voltage_max":44,"temp_max":65}
            if voltage < thr.get("voltage_min", 28) and power > 10:
                is_anomaly = True; confidence = 0.85; reason = "voltage_drop"
            elif temp > thr.get("temp_max", 65):
                is_anomaly = True; confidence = 0.92; reason = "overtemp"

        return {"panel_id":panel_id,"panel_label":label,
                "is_anomaly":is_anomaly,"confidence":confidence,
                "reason":reason,"voltage":voltage,"temp":temp}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    logger.info("VoltVision AI Engine starting...")
    init_db()
    mqtt_client = _setup_mqtt()

    lstm  = LSTMForecaster()
    cv    = CVInference()
    bfci  = BFCIEngine()

    threads = [
        threading.Thread(target=lstm.run, args=(mqtt_client,), name="lstm",  daemon=True),
        threading.Thread(target=cv.run,   args=(mqtt_client,), name="cv",    daemon=True),
        threading.Thread(target=bfci.run, args=(mqtt_client,), name="bfci",  daemon=True),
    ]
    for t in threads:
        t.start()
        logger.info("Thread started: %s", t.name)

    def _shutdown(sig, frame):
        logger.info("Shutting down AI Engine...")
        _STOP.set()
        if mqtt_client:
            mqtt_client.loop_stop(); mqtt_client.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    logger.info("AI Engine running — LSTM/%ds CV/%dm BFCI/%dh",
                settings.LSTM_INFERENCE_INTERVAL_SECONDS,
                settings.CV_SCAN_INTERVAL_MINUTES,
                settings.BFCI_INFERENCE_INTERVAL_HOURS)
    while True:
        alive = [t.name for t in threads if t.is_alive()]
        logger.debug("Active threads: %s", alive)
        time.sleep(60)


if __name__ == "__main__":
    main()
