import sqlite3, os, logging
from contextlib import contextmanager
from config.settings import settings

logger = logging.getLogger(__name__)


def get_db_path() -> str:
    path = settings.SQLITE_DB_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return path


@contextmanager
def get_db():
    conn = sqlite3.connect(get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    full_name     TEXT    NOT NULL DEFAULT '',
    role          TEXT    NOT NULL DEFAULT 'viewer'
                          CHECK(role IN ('admin','operator','viewer')),
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    last_login    TEXT
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

CREATE TABLE IF NOT EXISTS refresh_tokens (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT    NOT NULL UNIQUE,
    expires_at TEXT    NOT NULL,
    revoked    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_rt_hash ON refresh_tokens(token_hash);

CREATE TABLE IF NOT EXISTS panels (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    label        TEXT    NOT NULL UNIQUE,
    row_label    TEXT    NOT NULL,
    col_number   INTEGER NOT NULL,
    installed_at TEXT    NOT NULL DEFAULT (date('now')),
    is_active    INTEGER NOT NULL DEFAULT 1,
    notes        TEXT    NOT NULL DEFAULT '',
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_panels_label ON panels(label);

CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id     INTEGER REFERENCES panels(id) ON DELETE SET NULL,
    alert_type   TEXT    NOT NULL,
    severity     TEXT    NOT NULL CHECK(severity IN ('info','warning','critical')),
    message      TEXT    NOT NULL,
    confidence   REAL,
    resolved     INTEGER NOT NULL DEFAULT 0,
    resolved_at  TEXT,
    resolved_by  INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_alerts_panel    ON alerts(panel_id);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved);

CREATE TABLE IF NOT EXISTS work_orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id     INTEGER REFERENCES panels(id) ON DELETE SET NULL,
    alert_id     INTEGER REFERENCES alerts(id) ON DELETE SET NULL,
    title        TEXT    NOT NULL,
    description  TEXT    NOT NULL DEFAULT '',
    status       TEXT    NOT NULL DEFAULT 'open'
                         CHECK(status IN ('open','in_progress','closed')),
    priority     TEXT    NOT NULL DEFAULT 'medium'
                         CHECK(priority IN ('low','medium','high','critical')),
    created_by   INTEGER REFERENCES users(id) ON DELETE SET NULL,
    assigned_to  INTEGER REFERENCES users(id) ON DELETE SET NULL,
    closed_at    TEXT,
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bfci_logs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id          INTEGER NOT NULL REFERENCES panels(id) ON DELETE CASCADE,
    bfci_score        REAL    NOT NULL,
    uv_absorption_pct REAL,
    days_to_recoat    INTEGER,
    coat_date         TEXT,
    batch_id          TEXT,
    notes             TEXT    NOT NULL DEFAULT '',
    recorded_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_bfci_panel ON bfci_logs(panel_id);

CREATE TABLE IF NOT EXISTS cv_detections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    panel_id      INTEGER REFERENCES panels(id) ON DELETE SET NULL,
    cv_class      TEXT    NOT NULL CHECK(cv_class IN ('clean','dusty','cracked','bio_degraded','pending')),
    confidence    REAL    NOT NULL,
    eff_loss_pct  REAL    DEFAULT 0,
    image_path    TEXT,
    model_ver     TEXT    NOT NULL DEFAULT 'v1',
    scanned_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cv_panel ON cv_detections(panel_id);

CREATE TABLE IF NOT EXISTS relay_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    battery_on    INTEGER NOT NULL DEFAULT 0,
    load_on       INTEGER NOT NULL DEFAULT 1,
    grid_export   INTEGER NOT NULL DEFAULT 0,
    spare_on      INTEGER NOT NULL DEFAULT 0,
    reason        TEXT    NOT NULL DEFAULT 'auto',
    commanded_by  TEXT    NOT NULL DEFAULT 'ai_ems',
    soc_pct       REAL,
    commanded_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS system_config (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action     TEXT    NOT NULL,
    resource   TEXT    NOT NULL,
    detail     TEXT    NOT NULL DEFAULT '',
    ip_address TEXT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
"""


def init_db():
    logger.info("Initialising SQLite at %s", get_db_path())
    with get_db() as conn:
        conn.executescript(SCHEMA)
    logger.info("Schema ready.")


# ── InfluxDB (optional) ───────────────────────────────────────────────────────
_influx_client = None


def get_influx_client():
    global _influx_client
    if _influx_client is not None:
        return _influx_client
    if not settings.INFLUX_TOKEN:
        return None
    try:
        from influxdb_client import InfluxDBClient
        _influx_client = InfluxDBClient(
            url=settings.INFLUX_URL, token=settings.INFLUX_TOKEN, org=settings.INFLUX_ORG
        )
        logger.info("InfluxDB connected to %s", settings.INFLUX_URL)
    except Exception as e:
        logger.error("InfluxDB error: %s", e)
    return _influx_client


def write_influx(bucket: str, record) -> bool:
    client = get_influx_client()
    if not client:
        return False
    try:
        client.write_api().write(bucket=bucket, org=settings.INFLUX_ORG, record=record)
        return True
    except Exception as e:
        logger.error("InfluxDB write: %s", e)
        return False


def query_influx(flux: str) -> list:
    client = get_influx_client()
    if not client:
        return []
    try:
        tables = client.query_api().query(flux, org=settings.INFLUX_ORG)
        return [r.values for t in tables for r in t.records]
    except Exception as e:
        logger.error("InfluxDB query: %s", e)
        return []
