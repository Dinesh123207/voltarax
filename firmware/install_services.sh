#!/bin/bash
# VoltVision AI — Raspberry Pi Installation Script
# Run: sudo bash firmware/install_services.sh
# Tested on: Raspberry Pi OS (Bookworm) 64-bit

set -e
INSTALL_DIR=/home/pi/voltvision
VENV_DIR=$INSTALL_DIR/venv

echo "========================================"
echo " VoltVision AI — RPi Setup"
echo "========================================"

# ── 1. System packages ─────────────────────────────────────────────────────────
echo "[1/8] Installing system packages..."
apt-get update -q
apt-get install -y -q \
    python3 python3-pip python3-venv \
    mosquitto mosquitto-clients \
    i2c-tools python3-smbus \
    git curl wget \
    libopenblas-dev    # for numpy on RPi

# Enable I2C
raspi-config nonint do_i2c 0
echo "  ✅ I2C enabled"

# ── 2. InfluxDB v2 ─────────────────────────────────────────────────────────────
echo "[2/8] Installing InfluxDB v2..."
if ! command -v influx &>/dev/null; then
    curl -s https://repos.influxdata.com/influxdata-archive_compat.key | gpg --dearmor > /etc/apt/trusted.gpg.d/influxdb.gpg
    echo "deb [signed-by=/etc/apt/trusted.gpg.d/influxdb.gpg] https://repos.influxdata.com/debian stable main" > /etc/apt/sources.list.d/influxdb.list
    apt-get update -q && apt-get install -y -q influxdb2
    systemctl enable influxdb && systemctl start influxdb
    sleep 3
    # Setup InfluxDB (edit token after setup)
    influx setup --username voltarax --password Voltarax@2026 \
                 --org voltarax --bucket biosolar_power \
                 --retention 90d --force || true
    echo "  ✅ InfluxDB installed"
else
    echo "  ✅ InfluxDB already installed"
fi

# ── 3. Python virtualenv ───────────────────────────────────────────────────────
echo "[3/8] Setting up Python venv..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi
source $VENV_DIR/bin/activate

pip install --upgrade pip -q
pip install -q \
    fastapi "uvicorn[standard]" \
    "python-jose[cryptography]" bcrypt \
    "pydantic[email]" pydantic-settings \
    python-multipart aiofiles \
    influxdb-client paho-mqtt \
    scikit-learn scikit-optimize \
    numpy pandas joblib \
    email-validator httpx \
    smbus2

# TFLite runtime for RPi
pip install -q tflite-runtime 2>/dev/null || \
    pip install -q tensorflow-cpu 2>/dev/null || \
    echo "  ⚠️  TFLite not installed — AI models will use sklearn fallback"

echo "  ✅ Python packages installed"

# ── 4. Mosquitto config ─────────────────────────────────────────────────────────
echo "[4/8] Configuring Mosquitto..."
cat > /etc/mosquitto/conf.d/voltvision.conf << 'MQTTCONF'
listener 1883
allow_anonymous true
persistence true
persistence_location /var/lib/mosquitto/
log_dest file /var/log/mosquitto/mosquitto.log
MQTTCONF
systemctl enable mosquitto && systemctl restart mosquitto
echo "  ✅ Mosquitto configured"

# ── 5. Seed database ───────────────────────────────────────────────────────────
echo "[5/8] Seeding database..."
cd $INSTALL_DIR
source $VENV_DIR/bin/activate
if [ ! -f "db/voltvision.db" ]; then
    python3 scripts/seed_db.py
fi
echo "  ✅ Database ready"

# ── 6. Train models (if not already trained) ───────────────────────────────────
echo "[6/8] Checking AI models..."
if [ ! -f "ai_models/lstm_biosolar.pkl" ]; then
    echo "  Training models on synthetic data..."
    python3 training/generate_data.py
    python3 training/train_all_models.py
else
    echo "  ✅ Models already trained"
fi

# ── 7. Install systemd services ────────────────────────────────────────────────
echo "[7/8] Installing systemd services..."

SERVICES="voltvision-api voltvision-ai voltvision-mqtt-bridge voltvision-spectral"

for SVC in $SERVICES; do
    cat > /etc/systemd/system/$SVC.service << EOF
[Unit]
Description=VoltVision AI — $SVC
After=network.target mosquitto.service
Wants=mosquitto.service

[Service]
Type=simple
User=pi
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$INSTALL_DIR/.env
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
EOF

    case $SVC in
        voltvision-api)
            echo "ExecStart=$VENV_DIR/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000" >> /etc/systemd/system/$SVC.service ;;
        voltvision-ai)
            echo "ExecStart=$VENV_DIR/bin/python3 services/ai_engine.py" >> /etc/systemd/system/$SVC.service ;;
        voltvision-mqtt-bridge)
            echo "ExecStart=$VENV_DIR/bin/python3 services/mqtt_to_influx.py" >> /etc/systemd/system/$SVC.service ;;
        voltvision-spectral)
            echo "ExecStart=$VENV_DIR/bin/python3 firmware/spectral_reader.py" >> /etc/systemd/system/$SVC.service ;;
    esac

    echo "" >> /etc/systemd/system/$SVC.service
    echo "[Install]" >> /etc/systemd/system/$SVC.service
    echo "WantedBy=multi-user.target" >> /etc/systemd/system/$SVC.service
done

systemctl daemon-reload
for SVC in $SERVICES; do
    systemctl enable $SVC
    systemctl start $SVC
    sleep 1
    STATUS=$(systemctl is-active $SVC)
    echo "  $SVC — $STATUS"
done

# ── 8. Final check ─────────────────────────────────────────────────────────────
echo "[8/8] Running hardware setup check..."
source $VENV_DIR/bin/activate
cd $INSTALL_DIR
python3 firmware/hardware_setup.py

echo ""
echo "========================================"
echo " VoltVision AI Installation Complete!"
echo "========================================"
echo ""
echo " Dashboard: http://$(hostname -I | awk '{print $1}'):8000/dashboard/"
echo " API docs:  http://$(hostname -I | awk '{print $1}'):8000/docs"
echo " Login:     admin@voltarax.in / Admin@123"
echo ""
echo " Flash ESP32 firmware:"
echo "   - Open firmware/voltarax_esp32.ino in Arduino IDE"
echo "   - Set MQTT_SERVER = \"$(hostname -I | awk '{print $1}')\""
echo "   - Set PANEL_ID per unit (A1, A2, ..., D8)"
echo "   - Upload to each ESP32"
echo ""
echo " Check service logs:"
echo "   journalctl -u voltvision-api -f"
echo "   journalctl -u voltvision-ai -f"
