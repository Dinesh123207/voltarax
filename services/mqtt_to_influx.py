"""
VoltVision AI — MQTT to InfluxDB Bridge
Subscribes to all ESP32 MQTT topics, writes to InfluxDB.
In dev mode (no INFLUX_TOKEN), publishes simulated data every 30s.

Run: python3 services/mqtt_to_influx.py
"""
import json, logging, signal, sys, os, time, random
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paho.mqtt.client as mqtt
from config.settings import settings
from db.database import init_db, get_db, write_influx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mqtt_bridge")

_SOLAR = [0,0,0,0,0,0,.1,.3,.8,1.8,3.2,4.4,5.1,5.6,5.7,5.3,4.6,3.8,2.7,1.5,.6,.2,0,0]
_PANELS = [f"{chr(65+r)}{c}" for r in range(4) for c in range(1, 9)]


# ── Topic handlers ────────────────────────────────────────────────────────────

def _handle_power(panel_id: str, payload: dict):
    try:
        from influxdb_client import Point
        pt = (Point("panel_power")
              .tag("panel_id", panel_id)
              .field("voltage",   float(payload.get("V", 0)))
              .field("current",   float(payload.get("I", 0)))
              .field("power",     float(payload.get("W", 0)))
              .field("energy_wh", float(payload.get("Wh", 0))))
        write_influx(settings.INFLUX_BUCKET_POWER, pt)
        logger.debug("Power → InfluxDB panel=%s W=%.1f", panel_id, payload.get("W", 0))
    except Exception as e:
        logger.error("Power write error: %s", e)


def _handle_temp(panel_id: str, payload: dict):
    try:
        from influxdb_client import Point
        pt = (Point("panel_temp")
              .tag("panel_id", panel_id)
              .field("surface",   float(payload.get("surface", 0)))
              .field("ambient",   float(payload.get("ambient", 0)))
              .field("humidity",  float(payload.get("humidity", 0)))
              .field("enclosure", float(payload.get("enclosure", 0))))
        write_influx(settings.INFLUX_BUCKET_ENV, pt)
    except Exception as e:
        logger.error("Temp write error: %s", e)


def _handle_spectral(panel_id: str, payload: dict):
    try:
        from influxdb_client import Point
        pt = Point("spectral").tag("panel_id", panel_id)
        for k, v in payload.items():
            if k.startswith("ch") and isinstance(v, (int, float)):
                pt = pt.field(k, float(v))
        write_influx(settings.INFLUX_BUCKET_SPECTRAL, pt)
    except Exception as e:
        logger.error("Spectral write error: %s", e)


def _handle_env(payload: dict):
    try:
        from influxdb_client import Point
        pt = (Point("environment")
              .field("lux",         float(payload.get("lux", 0)))
              .field("irradiance",  float(payload.get("irradiance", 0)))
              .field("aqi",         float(payload.get("aqi", 0)))
              .field("wind_speed",  float(payload.get("wind", 0))))
        write_influx(settings.INFLUX_BUCKET_ENV, pt)
    except Exception as e:
        logger.error("Env write error: %s", e)


def _handle_anomaly(payload: dict):
    try:
        panel_label = payload.get("panel_id", "")
        with get_db() as conn:
            panel_row = conn.execute(
                "SELECT id FROM panels WHERE label=?", (panel_label,)
            ).fetchone()
            panel_db_id = panel_row["id"] if panel_row else None
            conn.execute(
                "INSERT INTO alerts (panel_id,alert_type,severity,message,confidence) "
                "VALUES (?,?,?,?,?)",
                (panel_db_id,
                 payload.get("type", "anomaly"),
                 payload.get("severity", "warning"),
                 payload.get("message", f"Anomaly on panel {panel_label}"),
                 payload.get("conf", None))
            )
        logger.warning("Anomaly alert: panel=%s type=%s", panel_label, payload.get("type"))
    except Exception as e:
        logger.error("Anomaly write error: %s", e)


def _handle_ai_out(suffix: str, payload: dict):
    try:
        from influxdb_client import Point
        pt = Point(suffix)
        for k, v in payload.items():
            if isinstance(v, (int, float)):
                pt = pt.field(k, float(v))
            elif isinstance(v, str) and k not in ("ts", "timestamp"):
                pt = pt.tag(k, v)
        write_influx(settings.INFLUX_BUCKET_AI, pt)
    except Exception as e:
        logger.error("AI out write error %s: %s", suffix, e)


# ── MQTT callbacks ────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        pfx = settings.MQTT_TOPIC_PREFIX
        topics = [
            (f"{pfx}/panel/+/power",    1),
            (f"{pfx}/panel/+/temp",     1),
            (f"{pfx}/panel/+/spectral", 1),
            (f"{pfx}/env/all",          0),
            (f"{pfx}/battery/soc",      1),
            (f"{pfx}/ai/#",             1),
            (f"{pfx}/system/heartbeat", 0),
        ]
        client.subscribe(topics)
        logger.info("MQTT connected to %s:%d — subscribed %d topics",
                    settings.MQTT_HOST, settings.MQTT_PORT, len(topics))
    else:
        logger.error("MQTT connect failed rc=%d", rc)


def on_disconnect(client, userdata, rc):
    if rc != 0:
        logger.warning("MQTT unexpected disconnect rc=%d", rc)


def on_message(client, userdata, msg):
    topic = msg.topic
    pfx   = settings.MQTT_TOPIC_PREFIX
    try:
        payload = json.loads(msg.payload.decode())
    except Exception:
        return  # ignore non-JSON

    try:
        parts = topic.split("/")
        if f"{pfx}/panel/" in topic and topic.endswith("/power"):
            _handle_power(parts[2], payload)
        elif f"{pfx}/panel/" in topic and topic.endswith("/temp"):
            _handle_temp(parts[2], payload)
        elif f"{pfx}/panel/" in topic and topic.endswith("/spectral"):
            _handle_spectral(parts[2], payload)
        elif topic == f"{pfx}/env/all":
            _handle_env(payload)
        elif topic == f"{pfx}/ai/anomaly":
            _handle_anomaly(payload)
        elif topic.startswith(f"{pfx}/ai/"):
            _handle_ai_out(parts[-1], payload)
        elif topic == f"{pfx}/system/heartbeat":
            logger.debug("Heartbeat from %s", payload.get("device", "?"))
    except Exception as e:
        logger.error("Message handler error on %s: %s", topic, e)


# ── Dev simulator ─────────────────────────────────────────────────────────────

def _publish_simulated(client):
    pfx  = settings.MQTT_TOPIC_PREFIX
    hour = datetime.now().hour
    base = _SOLAR[hour]

    for panel in _PANELS[:8]:  # publish 8 panels per cycle
        v = round(36.8 * max(0.01, base) * random.uniform(0.98, 1.02), 2) if base > 0 else 0
        i = round(13.2 * max(0.01, base) * random.uniform(0.98, 1.02), 2) if base > 0 else 0
        w = round(v * i, 1)
        payload = json.dumps({"V": v, "I": i, "W": w, "Wh": round(w * 0.00833, 4)})
        client.publish(f"{pfx}/panel/{panel}/power", payload, qos=1)

        temp_payload = json.dumps({
            "surface":   round(44.2 + random.uniform(-2, 2), 1),
            "ambient":   round(38.5 + random.uniform(-2, 2), 1),
            "humidity":  round(34.0 + random.uniform(-5, 5), 1),
            "enclosure": round(42.0 + random.uniform(-2, 2), 1),
        })
        client.publish(f"{pfx}/panel/{panel}/temp", temp_payload, qos=1)

    env = json.dumps({
        "lux":        round(random.uniform(60000, 80000) * max(0, base), 0),
        "irradiance": round(random.uniform(780, 850) * max(0, base), 1),
        "aqi":        round(random.uniform(65, 95), 0),
        "wind":       round(random.uniform(8, 15), 1),
    })
    client.publish(f"{pfx}/env/all", env, qos=0)
    logger.debug("Simulated data published")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    sim_mode = not settings.INFLUX_TOKEN or settings.APP_ENV == "development"
    logger.info("MQTT Bridge starting (sim_mode=%s)", sim_mode)

    init_db()

    client = mqtt.Client(client_id=settings.MQTT_CLIENT_ID + "-bridge")
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message

    if settings.MQTT_USERNAME:
        client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)

    try:
        client.connect(settings.MQTT_HOST, settings.MQTT_PORT, keepalive=60)
    except Exception as e:
        logger.error("Cannot connect to MQTT broker: %s", e)
        logger.error("Start Mosquitto first: mosquitto -d")
        sys.exit(1)

    client.loop_start()

    def _shutdown(sig, frame):
        logger.info("MQTT Bridge shutting down...")
        client.loop_stop()
        client.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    logger.info("MQTT Bridge running. Press Ctrl+C to stop.")
    while True:
        if sim_mode:
            _publish_simulated(client)
        time.sleep(30)


if __name__ == "__main__":
    main()
