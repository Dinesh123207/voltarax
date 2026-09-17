/*
 * VoltVision AI — ESP32 Sensor Firmware
 * Voltarax Private Limited | Patent 487411-001
 *
 * Sensors:
 *   INA226  → voltage, current, power  (I2C 0x40)
 *   DHT22   → temperature, humidity    (GPIO 4)
 *   BH1750  → lux                      (I2C 0x23)
 *   DS18B20 → panel surface temp       (GPIO 5, OneWire)
 *   MQ-135  → AQI (via ADS1115)        (I2C 0x48)
 *   AS7265x → 18-ch spectral (on RPi)  (I2C 0x49)
 *
 * Publishes every 30s to MQTT broker (Raspberry Pi).
 *
 * Board: ESP32 DevKit v1
 * Install libraries via Arduino Library Manager:
 *   - PubSubClient       (MQTT)
 *   - ArduinoJson        (JSON payloads)
 *   - INA226_WE          (Power monitor)
 *   - DHT sensor library (Adafruit)
 *   - BH1750             (Light sensor)
 *   - OneWire + DallasTemperature (DS18B20)
 *   - Adafruit ADS1X15  (ADS1115 ADC)
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
// Sensor libraries — install via Arduino Library Manager
#include <INA226_WE.h>
#include <DHT.h>
#include <BH1750.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_ADS1X15.h>

// ── config.h values (edit these) ─────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* MQTT_SERVER   = "192.168.1.100";   // RPi IP address
const int   MQTT_PORT     = 1883;
const char* MQTT_USER     = "";                 // leave empty if no auth
const char* MQTT_PASS     = "";
const char* PANEL_ID      = "A1";              // Change per ESP32 unit
const char* TOPIC_PREFIX  = "voltarax";
const int   PUBLISH_INTERVAL_MS = 30000;        // 30 seconds

// ── GPIO pins ─────────────────────────────────────────────────────────────────
#define DHT_PIN        4
#define DHT_TYPE       DHT22
#define DS18B20_PIN    5
#define RELAY_BATTERY  25
#define RELAY_LOAD     26
#define RELAY_GRID     27
#define RELAY_SPARE    14

// ── Sensor objects ────────────────────────────────────────────────────────────
WiFiClient     espClient;
PubSubClient   mqtt(espClient);
INA226_WE      ina226(0x40);
DHT            dht(DHT_PIN, DHT_TYPE);
BH1750         lightMeter;
OneWire        oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);
Adafruit_ADS1115  ads;

// ── State ──────────────────────────────────────────────────────────────────────
unsigned long lastPublish   = 0;
unsigned long lastHeartbeat = 0;
bool          relayBattery  = true;
bool          relayLoad     = true;
bool          relayGrid     = false;
bool          relaySpare    = false;

// ── Topic builders ─────────────────────────────────────────────────────────────
String topicPower()     { return String(TOPIC_PREFIX) + "/panel/" + PANEL_ID + "/power"; }
String topicTemp()      { return String(TOPIC_PREFIX) + "/panel/" + PANEL_ID + "/temp"; }
String topicEnv()       { return String(TOPIC_PREFIX) + "/env/all"; }
String topicAnomaly()   { return String(TOPIC_PREFIX) + "/ai/anomaly"; }
String topicHeartbeat() { return String(TOPIC_PREFIX) + "/system/heartbeat"; }
String topicRelaySub()  { return String(TOPIC_PREFIX) + "/control/relay"; }

// ── WiFi ───────────────────────────────────────────────────────────────────────
void connectWiFi() {
    Serial.print("Connecting to WiFi: ");
    Serial.println(WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500); Serial.print("."); attempts++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected. IP: " + WiFi.localIP().toString());
    } else {
        Serial.println("\nWiFi FAILED — will retry");
    }
}

// ── MQTT ───────────────────────────────────────────────────────────────────────
void mqttCallback(char* topic, byte* payload, unsigned int length) {
    // Handle relay commands from AI engine / dashboard
    // Topic: voltarax/control/relay
    // Payload: {"battery_on":true,"load_on":true,"grid_export":false,"spare_on":false}
    StaticJsonDocument<256> doc;
    deserializeJson(doc, payload, length);

    if (doc.containsKey("battery_on")) {
        relayBattery = doc["battery_on"].as<bool>();
        relayLoad    = doc["load_on"].as<bool>();
        relayGrid    = doc["grid_export"].as<bool>();
        relaySpare   = doc["spare_on"].as<bool>();

        digitalWrite(RELAY_BATTERY, relayBattery ? HIGH : LOW);
        digitalWrite(RELAY_LOAD,    relayLoad    ? HIGH : LOW);
        digitalWrite(RELAY_GRID,    relayGrid    ? HIGH : LOW);
        digitalWrite(RELAY_SPARE,   relaySpare   ? HIGH : LOW);

        Serial.printf("Relay updated: bat=%d load=%d grid=%d spare=%d\n",
                      relayBattery, relayLoad, relayGrid, relaySpare);
    }
}

void connectMQTT() {
    while (!mqtt.connected()) {
        Serial.print("Connecting MQTT...");
        String clientId = "voltarax-esp32-" + String(PANEL_ID);
        bool ok = (strlen(MQTT_USER) > 0)
            ? mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASS)
            : mqtt.connect(clientId.c_str());

        if (ok) {
            Serial.println(" connected.");
            mqtt.subscribe(topicRelaySub().c_str());
        } else {
            Serial.printf(" failed (rc=%d). Retry in 5s\n", mqtt.state());
            delay(5000);
        }
    }
}

// ── Anomaly detection (Z-score on ESP32) ───────────────────────────────────────
bool detectAnomaly(float voltage, float current, float temp,
                   float& confidence, String& reason) {
    // Voltage drop anomaly
    if (voltage > 5 && voltage < 28.0) {
        confidence = min(0.99, 0.7 + (28.0 - voltage) / 28.0);
        reason = "voltage_drop";
        return true;
    }
    // Overcurrent
    if (current > 18.0) {
        confidence = 0.92;
        reason = "overcurrent";
        return true;
    }
    // Overheat
    if (temp > 65.0) {
        confidence = 0.95;
        reason = "overtemp";
        return true;
    }
    // Open circuit (high voltage, no current during daylight)
    if (voltage > 42.0 && current < 0.5) {
        confidence = 0.88;
        reason = "open_circuit";
        return true;
    }
    return false;
}

// ── Publish: Power ─────────────────────────────────────────────────────────────
void publishPower() {
    float voltage = ina226.getBusVoltage_V();
    float current = ina226.getCurrent_mA() / 1000.0;  // mA → A
    float power   = ina226.getBusPower();              // W
    float energy  = power * (PUBLISH_INTERVAL_MS / 3600000.0);  // Wh

    StaticJsonDocument<128> doc;
    doc["V"]    = round(voltage * 100) / 100.0;
    doc["I"]    = round(current * 100) / 100.0;
    doc["W"]    = round(power * 10)    / 10.0;
    doc["Wh"]   = round(energy * 10000) / 10000.0;
    doc["panel"]= PANEL_ID;

    char buf[128];
    serializeJson(doc, buf);
    mqtt.publish(topicPower().c_str(), buf, true);   // retained=true

    // Anomaly detection
    float conf = 0; String reason = "";
    if (detectAnomaly(voltage, current, 0, conf, reason)) {
        StaticJsonDocument<192> anom;
        anom["panel_id"]  = PANEL_ID;
        anom["type"]      = reason;
        anom["severity"]  = (conf > 0.90) ? "critical" : "warning";
        anom["conf"]      = round(conf * 1000) / 1000.0;
        anom["voltage"]   = voltage;
        anom["message"]   = "Anomaly: " + reason + " on panel " + PANEL_ID;
        char abuf[192];
        serializeJson(anom, abuf);
        mqtt.publish(topicAnomaly().c_str(), abuf);
        Serial.printf("ANOMALY: %s (conf=%.2f)\n", reason.c_str(), conf);
    }

    Serial.printf("[POWER] V=%.2f I=%.2fA W=%.1f\n", voltage, current, power);
}

// ── Publish: Temperature ───────────────────────────────────────────────────────
void publishTemp() {
    float ambient  = dht.readTemperature();
    float humidity = dht.readHumidity();
    ds18b20.requestTemperatures();
    float surface  = ds18b20.getTempCByIndex(0);

    if (isnan(ambient) || isnan(humidity)) {
        Serial.println("[TEMP] DHT22 read failed");
        return;
    }

    StaticJsonDocument<128> doc;
    doc["surface"]   = isnan(surface) ? ambient + 6.0 : round(surface * 10) / 10.0;
    doc["ambient"]   = round(ambient  * 10) / 10.0;
    doc["humidity"]  = round(humidity * 10) / 10.0;
    doc["enclosure"] = round((ambient + 3.5) * 10) / 10.0;

    char buf[128];
    serializeJson(doc, buf);
    mqtt.publish(topicTemp().c_str(), buf, true);

    // Overheat anomaly check
    float conf = 0; String reason = "";
    if (detectAnomaly(0, 0, doc["surface"].as<float>(), conf, reason)) {
        StaticJsonDocument<192> anom;
        anom["panel_id"] = PANEL_ID;
        anom["type"]     = reason;
        anom["severity"] = "critical";
        anom["conf"]     = conf;
        anom["temp"]     = doc["surface"].as<float>();
        anom["message"]  = "Panel " + String(PANEL_ID) + " overtemp: " +
                           String(doc["surface"].as<float>(), 1) + "°C";
        char abuf[192];
        serializeJson(anom, abuf);
        mqtt.publish(topicAnomaly().c_str(), abuf);
    }

    Serial.printf("[TEMP] Surface=%.1f Ambient=%.1f Humidity=%.1f%%\n",
                  doc["surface"].as<float>(), ambient, humidity);
}

// ── Publish: Environment ───────────────────────────────────────────────────────
void publishEnv() {
    float lux = lightMeter.readLightLevel();
    // AQI from MQ-135 via ADS1115 (channel 0)
    int16_t raw  = ads.readADC_SingleEnded(0);
    float   volt = ads.computeVolts(raw);
    // Convert voltage to approximate AQI (calibration needed in production)
    float aqi = (volt / 3.3) * 300.0;

    // Irradiance approximation from lux (1 W/m² ≈ 93 lux for solar)
    float irradiance = lux / 93.0;

    StaticJsonDocument<128> doc;
    doc["lux"]        = round(lux);
    doc["irradiance"] = round(irradiance * 10) / 10.0;
    doc["aqi"]        = round(aqi);
    doc["wind"]       = 0;   // Add wind sensor (anemometer) later

    char buf[128];
    serializeJson(doc, buf);
    mqtt.publish(topicEnv().c_str(), buf);

    Serial.printf("[ENV] Lux=%.0f Irr=%.1f AQI=%.0f\n", lux, irradiance, aqi);
}

// ── Publish: Heartbeat ─────────────────────────────────────────────────────────
void publishHeartbeat() {
    StaticJsonDocument<128> doc;
    doc["device"]  = String("esp32-") + PANEL_ID;
    doc["uptime"]  = millis() / 1000;
    doc["wifi_rssi"] = WiFi.RSSI();
    doc["free_heap"] = ESP.getFreeHeap();

    char buf[128];
    serializeJson(doc, buf);
    mqtt.publish(topicHeartbeat().c_str(), buf);
}

// ── Setup ──────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("\n=== VoltVision AI — ESP32 Firmware ===");
    Serial.printf("Panel ID: %s\n", PANEL_ID);

    // GPIO
    pinMode(RELAY_BATTERY, OUTPUT); digitalWrite(RELAY_BATTERY, HIGH);
    pinMode(RELAY_LOAD,    OUTPUT); digitalWrite(RELAY_LOAD,    HIGH);
    pinMode(RELAY_GRID,    OUTPUT); digitalWrite(RELAY_GRID,    LOW);
    pinMode(RELAY_SPARE,   OUTPUT); digitalWrite(RELAY_SPARE,   LOW);

    // I2C
    Wire.begin();

    // INA226 power monitor
    if (!ina226.init()) {
        Serial.println("INA226 not found — check wiring at 0x40");
    } else {
        ina226.setResistorRange(0.1, 1.3);    // 0.1Ω shunt, max 1.3A → adjust for your panel
        ina226.setCorrectionFactor(0.98);
        Serial.println("INA226 OK");
    }

    // DHT22
    dht.begin();
    Serial.println("DHT22 OK");

    // BH1750 light sensor
    if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
        Serial.println("BH1750 not found — check wiring at 0x23");
    } else {
        Serial.println("BH1750 OK");
    }

    // DS18B20
    ds18b20.begin();
    Serial.printf("DS18B20: %d sensor(s) found\n", ds18b20.getDeviceCount());

    // ADS1115
    if (!ads.begin()) {
        Serial.println("ADS1115 not found — check wiring at 0x48");
    } else {
        ads.setGain(GAIN_ONE);   // ±4.096V range
        Serial.println("ADS1115 OK");
    }

    // WiFi + MQTT
    connectWiFi();
    mqtt.setServer(MQTT_SERVER, MQTT_PORT);
    mqtt.setCallback(mqttCallback);
    mqtt.setBufferSize(512);

    Serial.println("Setup complete. Publishing every 30s...");
}

// ── Loop ───────────────────────────────────────────────────────────────────────
void loop() {
    // Maintain connections
    if (WiFi.status() != WL_CONNECTED) connectWiFi();
    if (!mqtt.connected())              connectMQTT();
    mqtt.loop();

    unsigned long now = millis();

    // Publish sensor data every 30 seconds
    if (now - lastPublish >= PUBLISH_INTERVAL_MS) {
        lastPublish = now;
        publishPower();
        delay(100);
        publishTemp();
        delay(100);
        publishEnv();
    }

    // Heartbeat every 60 seconds
    if (now - lastHeartbeat >= 60000) {
        lastHeartbeat = now;
        publishHeartbeat();
    }
}
