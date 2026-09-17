"""
VoltVision AI — Raspberry Pi AS7265x Spectral Reader
Reads 18-channel spectral data from AS7265x sensor (I2C 0x49)
and publishes to MQTT for BFCI model inference.

Hardware: AS7265x connected to RPi I2C bus
Install:  pip install smbus2 paho-mqtt

Run: python3 firmware/spectral_reader.py
"""
import time, json, logging, signal, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paho.mqtt.client as mqtt
from config.settings import settings

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("spectral_reader")

# AS7265x Register map
AS7265x_ADDR        = 0x49
AS7265x_STATUS_REG  = 0x00
AS7265x_WRITE_REG   = 0x01
AS7265x_READ_REG    = 0x02
AS7265X_TX_VALID    = 0x02
AS7265X_RX_VALID    = 0x01

# Channel wavelength mapping (18 channels across 3 sensors)
CHANNELS = {
    "ch1_410nm":  "R",   # AS72651 (visible)
    "ch2_445nm":  "S",
    "ch3_480nm":  "T",
    "ch4_515nm":  "U",
    "ch5_555nm":  "V",
    "ch6_590nm":  "W",
    "ch7_630nm":  "G",   # AS72652 (visible ext)
    "ch8_680nm":  "H",
    "ch9_720nm":  "I",
    "ch10_760nm": "J",
    "ch11_810nm": "K",
    "ch12_860nm": "L",
    "ch13_900nm": "P",   # AS72653 (NIR)
    "ch14_940nm": "Q",
    "ch15_UV1":   "A",
    "ch16_UV2":   "B",
    "ch17_UV3":   "C",
    "ch18_UV4":   "D",
}

PANEL_ID = "A1"   # Change per physical sensor location


class AS7265xReader:
    """
    Driver for AS7265x 18-channel spectral sensor.
    Communicates over I2C via virtual register interface.
    """

    def __init__(self, i2c_address=AS7265x_ADDR):
        self.addr  = i2c_address
        self.bus   = None
        self._init_i2c()

    def _init_i2c(self):
        try:
            import smbus2
            self.bus = smbus2.SMBus(1)   # RPi I2C bus 1
            # Soft reset
            self._write_virtual(0x04, 0x01)
            time.sleep(1.0)
            # Set measurement mode: 3 (one-shot all channels)
            self._write_virtual(0x04, 0b00111010)
            # Set gain: 16x
            self._write_virtual(0x05, 0b00110000)
            logger.info("AS7265x initialised at I2C 0x%02x", self.addr)
        except ImportError:
            logger.warning("smbus2 not installed — using simulated readings")
            self.bus = None
        except Exception as e:
            logger.warning("AS7265x init failed: %s — using simulation", e)
            self.bus = None

    def _wait_ready(self, timeout=5.0):
        """Poll status register until data ready."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            status = self.bus.read_byte_data(self.addr, AS7265x_STATUS_REG)
            if status & AS7265X_RX_VALID:
                return True
            time.sleep(0.01)
        return False

    def _write_virtual(self, reg: int, value: int):
        """Write to a virtual register via I2C."""
        # Wait for TX ready
        for _ in range(100):
            s = self.bus.read_byte_data(self.addr, AS7265x_STATUS_REG)
            if not (s & AS7265X_TX_VALID):
                break
            time.sleep(0.005)
        self.bus.write_byte_data(self.addr, AS7265x_WRITE_REG, reg | 0x80)
        for _ in range(100):
            s = self.bus.read_byte_data(self.addr, AS7265x_STATUS_REG)
            if not (s & AS7265X_TX_VALID):
                break
            time.sleep(0.005)
        self.bus.write_byte_data(self.addr, AS7265x_WRITE_REG, value)

    def _read_virtual(self, reg: int) -> int:
        """Read from a virtual register."""
        for _ in range(100):
            s = self.bus.read_byte_data(self.addr, AS7265x_STATUS_REG)
            if not (s & AS7265X_TX_VALID):
                break
            time.sleep(0.005)
        self.bus.write_byte_data(self.addr, AS7265x_WRITE_REG, reg)
        for _ in range(100):
            s = self.bus.read_byte_data(self.addr, AS7265x_STATUS_REG)
            if s & AS7265X_RX_VALID:
                break
            time.sleep(0.005)
        return self.bus.read_byte_data(self.addr, AS7265x_READ_REG)

    def read_calibrated(self) -> dict:
        """
        Read all 18 calibrated channel values.
        Returns dict with normalised float values (0.0–1.0).
        """
        if not self.bus:
            return self._simulate()

        try:
            # Trigger one-shot measurement
            self._write_virtual(0x04, 0b00111010)
            time.sleep(0.8)   # integration time

            # Read 18 channels — each is 4 bytes (IEEE 754 float)
            channel_regs = {
                "R": 0x08, "S": 0x0A, "T": 0x0C, "U": 0x0E, "V": 0x10, "W": 0x12,
                "G": 0x14, "H": 0x16, "I": 0x18, "J": 0x1A, "K": 0x1C, "L": 0x1E,
                "P": 0x20, "Q": 0x22, "A": 0x24, "B": 0x26, "C": 0x28, "D": 0x2A,
            }
            raw = {}
            for ch_name, reg in channel_regs.items():
                b0 = self._read_virtual(reg)
                b1 = self._read_virtual(reg + 1)
                raw_16 = (b0 << 8) | b1
                # Convert to calibrated float (sensor provides 16-bit fixed-point)
                raw[ch_name] = raw_16 / 65535.0

            # Map to channel names and normalise 0–1
            result = {}
            for ch_key, sensor_ch in CHANNELS.items():
                result[ch_key] = round(raw.get(sensor_ch, 0.0), 4)
            return result

        except Exception as e:
            logger.error("AS7265x read error: %s — using simulation", e)
            return self._simulate()

    def _simulate(self) -> dict:
        """Realistic simulation when sensor not connected."""
        import random
        from datetime import datetime
        hour = datetime.now().hour
        solar_factor = max(0, 1 - abs(hour - 13) / 7) if 6 <= hour <= 20 else 0
        result = {}
        for ch_key, _ in CHANNELS.items():
            base = solar_factor * random.uniform(0.3, 0.9)
            if "UV" in ch_key:
                base *= random.uniform(0.6, 1.0)   # UV varies more
            result[ch_key] = round(max(0, min(1, base + random.gauss(0, 0.02))), 4)
        return result


def main():
    reader = AS7265xReader()

    # MQTT setup
    client = mqtt.Client(client_id="voltarax-spectral-rpi")
    if settings.MQTT_USERNAME:
        client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)
    try:
        client.connect(settings.MQTT_HOST, settings.MQTT_PORT, keepalive=60)
        client.loop_start()
        logger.info("MQTT connected to %s:%d", settings.MQTT_HOST, settings.MQTT_PORT)
    except Exception as e:
        logger.error("MQTT connection failed: %s", e)
        client = None

    def _shutdown(sig, frame):
        logger.info("Spectral reader shutting down...")
        if client:
            client.loop_stop(); client.disconnect()
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    PANELS = [f"{chr(65+r)}{c}" for r in range(4) for c in range(1, 9)]
    logger.info("Spectral reader running — reading every 6 hours per panel")

    while True:
        for panel_label in PANELS:
            readings = reader.read_calibrated()
            payload = {
                "panel_id": panel_label,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **readings,
            }
            topic = f"{settings.MQTT_TOPIC_PREFIX}/panel/{panel_label}/spectral"
            if client:
                client.publish(topic, json.dumps(payload), qos=1)
            logger.info("Spectral published for panel %s: UV1=%.3f ch8=%.3f",
                        panel_label, readings.get("ch15_UV1", 0),
                        readings.get("ch8_680nm", 0))
            time.sleep(2)   # Brief pause between panels

        # Wait 6 hours before next full scan
        logger.info("Full spectral scan complete. Next scan in 6 hours.")
        time.sleep(6 * 3600)


if __name__ == "__main__":
    main()
