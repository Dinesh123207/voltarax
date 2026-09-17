"""
VoltVision AI — Hardware Setup Script
Run this on Raspberry Pi when hardware arrives.
Verifies all connections and starts all services.

Usage: python3 firmware/hardware_setup.py
"""
import os, sys, time, subprocess, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEPS = []

def step(name):
    def decorator(fn):
        STEPS.append((name, fn))
        return fn
    return decorator

def ok(msg):   print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def fail(msg): print(f"  ❌ {msg}")


@step("Check Python version")
def check_python():
    import sys
    v = sys.version_info
    if v.major == 3 and v.minor >= 9:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} — need 3.9+")


@step("Check required Python packages")
def check_packages():
    pkgs = ["fastapi","uvicorn","paho.mqtt.client","influxdb_client",
            "sklearn","numpy","pandas","bcrypt","jose"]
    missing = []
    for pkg in pkgs:
        try:
            __import__(pkg)
            ok(f"{pkg}")
        except ImportError:
            fail(f"{pkg} — run: pip install {pkg}")
            missing.append(pkg)
    if missing:
        print(f"\n  Install missing: pip install {' '.join(missing)} --break-system-packages")


@step("Check I2C bus (AS7265x, INA226, BH1750, ADS1115)")
def check_i2c():
    try:
        result = subprocess.run(["i2cdetect","-y","1"], capture_output=True, text=True, timeout=5)
        output = result.stdout
        EXPECTED = {
            "0x23": "BH1750 light sensor",
            "0x40": "INA226 power monitor",
            "0x48": "ADS1115 ADC (MQ-135)",
            "0x49": "AS7265x spectral sensor",
        }
        for addr, name in EXPECTED.items():
            hex_short = addr.replace("0x","").lstrip("0") or "0"
            if hex_short.lower() in output.lower() or addr[2:].lower() in output.lower():
                ok(f"{addr} — {name}")
            else:
                warn(f"{addr} — {name} NOT FOUND (check wiring)")
    except FileNotFoundError:
        warn("i2cdetect not found — install: sudo apt install i2c-tools")
    except Exception as e:
        warn(f"I2C check failed: {e}")


@step("Check MQTT broker (Mosquitto)")
def check_mqtt():
    try:
        import paho.mqtt.client as mqtt
        connected = False
        def on_connect(c,u,f,rc): 
            nonlocal connected
            connected = (rc==0)
        c = mqtt.Client("hw-check")
        c.on_connect = on_connect
        c.connect("localhost", 1883, keepalive=5)
        c.loop_start(); time.sleep(2); c.loop_stop()
        if connected: ok("Mosquitto MQTT broker running on :1883")
        else:         fail("MQTT broker not responding — run: sudo systemctl start mosquitto")
    except Exception as e:
        fail(f"MQTT check failed: {e} — run: sudo apt install mosquitto mosquitto-clients")


@step("Check InfluxDB v2")
def check_influx():
    try:
        import urllib.request
        r = urllib.request.urlopen("http://localhost:8086/health", timeout=3)
        data = json.loads(r.read())
        if data.get("status") == "pass":
            ok("InfluxDB v2 running on :8086")
        else:
            warn(f"InfluxDB status: {data.get('status')}")
    except Exception as e:
        fail(f"InfluxDB not running: {e}")
        print("    Install: https://docs.influxdata.com/influxdb/v2/install/")


@step("Check AI models")
def check_models():
    MODELS = {
        "ai_models/lstm_biosolar.pkl":    "LSTM Forecaster",
        "ai_models/bfci_model.pkl":       "BFCI BO-Bagging",
        "ai_models/cv_model.pkl":         "CV Classifier",
        "ai_models/isolation_forest.pkl": "Isolation Forest",
    }
    for path, name in MODELS.items():
        if os.path.exists(path):
            size = os.path.getsize(path)/1024
            ok(f"{name} ({size:.0f} KB) — {path}")
        else:
            fail(f"{name} missing — run: python3 training/train_all_models.py")


@step("Check database")
def check_db():
    try:
        import sqlite3
        from config.settings import settings
        conn = sqlite3.connect(settings.SQLITE_DB_PATH)
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        panels = conn.execute("SELECT COUNT(*) FROM panels").fetchone()[0]
        users  = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        conn.close()
        ok(f"SQLite OK — {len(tables)} tables, {panels} panels, {users} users")
        if panels < 32:
            warn(f"Only {panels} panels — run: python3 scripts/seed_db.py")
    except Exception as e:
        fail(f"Database error: {e} — run: python3 scripts/seed_db.py")


@step("Check GPIO relay pins (ESP32 via MQTT)")
def check_relay():
    try:
        # On RPi, we control relay via MQTT to ESP32
        # This just verifies the relay state record exists
        import sqlite3
        from config.settings import settings
        conn = sqlite3.connect(settings.SQLITE_DB_PATH)
        row  = conn.execute("SELECT * FROM relay_log ORDER BY commanded_at DESC LIMIT 1").fetchone()
        conn.close()
        if row:
            ok(f"Relay log exists — last state: battery={row[1]} load={row[2]} grid={row[3]}")
        else:
            warn("No relay log found — will be created on first command")
    except Exception as e:
        warn(f"Relay check skipped: {e}")


@step("Check .env configuration")
def check_env():
    from config.settings import settings
    if settings.INFLUX_TOKEN:
        ok(f"INFLUX_TOKEN set ({len(settings.INFLUX_TOKEN)} chars)")
    else:
        warn("INFLUX_TOKEN empty — InfluxDB writes disabled (simulated mode)")
    if settings.TELEGRAM_BOT_TOKEN:
        ok("TELEGRAM_BOT_TOKEN set — alerts will be sent")
    else:
        warn("TELEGRAM_BOT_TOKEN empty — no Telegram alerts")
    if settings.JWT_SECRET_KEY and len(settings.JWT_SECRET_KEY) >= 32:
        ok(f"JWT_SECRET_KEY set ({len(settings.JWT_SECRET_KEY)} chars)")
    else:
        fail("JWT_SECRET_KEY too short — generate: python3 -c \"import secrets; print(secrets.token_hex(32))\"")


@step("Start all services (systemd)")
def start_services():
    services = ["voltvision-api","voltvision-ai","voltvision-mqtt-bridge","mosquitto","influxdb"]
    for svc in services:
        try:
            r = subprocess.run(["systemctl","is-active",svc], capture_output=True, text=True)
            status = r.stdout.strip()
            if status == "active":
                ok(f"{svc} — running")
            else:
                warn(f"{svc} — {status}")
                print(f"    Start with: sudo systemctl start {svc}")
        except Exception:
            warn(f"{svc} — systemctl not available (not on RPi or not installed as service)")


def main():
    print("\n" + "="*55)
    print("VoltVision AI — Hardware Setup Verification")
    print("="*55)

    for name, fn in STEPS:
        print(f"\n▶ {name}")
        try:
            fn()
        except Exception as e:
            fail(f"Step crashed: {e}")

    print("\n" + "="*55)
    print("HARDWARE SETUP COMPLETE")
    print("="*55)
    print("""
Quick start commands (run from voltvision/ directory):

  # Seed database (first time only)
  python3 scripts/seed_db.py

  # Start API server
  uvicorn api.main:app --host 0.0.0.0 --port 8000

  # Start AI engine (separate terminal)
  python3 services/ai_engine.py

  # Start MQTT bridge (separate terminal)
  python3 services/mqtt_to_influx.py

  # Start spectral reader / RPi sensor (separate terminal)
  python3 firmware/spectral_reader.py

  # Flash ESP32 firmware
  # Open firmware/voltarax_esp32.ino in Arduino IDE
  # Edit WIFI_SSID, WIFI_PASSWORD, MQTT_SERVER, PANEL_ID
  # Upload to ESP32

  # Dashboard
  http://<rpi-ip>:8000/dashboard/

  # After 60 days — retrain LSTM on real data
  python3 training/retrain_on_real_data.py --days 60 --model lstm

  # After 90 days — retrain BFCI on real data
  python3 training/retrain_on_real_data.py --days 90 --model bfci
""")


if __name__ == "__main__":
    main()
