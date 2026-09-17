from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "VoltVision AI"
    APP_VERSION: str = "2.1.0"
    APP_ENV: str = "development"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    DEBUG: bool = True

    JWT_SECRET_KEY: str = "changeme_32chars_minimum_xxxxxxxx"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    CORS_ORIGINS: str = "http://localhost:8000"

    SQLITE_DB_PATH: str = "./db/voltvision.db"

    INFLUX_URL: str = "http://localhost:8086"
    INFLUX_TOKEN: str = ""
    INFLUX_ORG: str = "voltarax"
    INFLUX_BUCKET_POWER: str = "biosolar_power"
    INFLUX_BUCKET_ENV: str = "biosolar_env"
    INFLUX_BUCKET_SPECTRAL: str = "biosolar_spectral"
    INFLUX_BUCKET_AI: str = "biosolar_ai_out"
    INFLUX_RETENTION_DAYS: int = 90

    MQTT_HOST: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_CLIENT_ID: str = "voltvision-backend"
    MQTT_USERNAME: str = ""
    MQTT_PASSWORD: str = ""
    MQTT_TOPIC_PREFIX: str = "voltarax"

    AI_MODELS_DIR: str = "./ai_models"
    LSTM_MODEL_PATH: str = "./ai_models/lstm_biosolar.tflite"
    CV_MODEL_PATH: str = "./ai_models/mobilenet_cv.tflite"
    BFCI_MODEL_PATH: str = "./ai_models/bfci_model.pkl"
    IFOREST_MODEL_PATH: str = "./ai_models/isolation_forest.pkl"

    LSTM_INFERENCE_INTERVAL_SECONDS: int = 60
    CV_SCAN_INTERVAL_MINUTES: int = 30
    BFCI_INFERENCE_INTERVAL_HOURS: int = 6

    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    ALERT_VOLTAGE_DROP_PCT: float = 0.15
    ALERT_TEMP_CRITICAL_C: float = 60.0
    ALERT_BFCI_WARN_THRESHOLD: float = 75.0
    ALERT_BFCI_CRIT_THRESHOLD: float = 65.0
    ALERT_CV_CONFIDENCE_MIN: float = 0.85

    SITE_NAME: str = "Jaipur BioSolar Site"
    SITE_LOCATION: str = "Jaipur, Rajasthan, India"
    SITE_PANEL_COUNT: int = 32
    GRID_EMISSION_FACTOR_KG_KWH: float = 0.82

    LLM_PROVIDER: str = "anthropic"
    LLM_API_KEY: str = ""
    LLM_MODEL: str = "claude-sonnet-4-6"

    ADMIN_SEED_EMAIL: str = "admin@voltarax.in"
    ADMIN_SEED_PASSWORD: str = "Admin@123"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
