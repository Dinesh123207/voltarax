import os, sys, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.update({
    "SQLITE_DB_PATH":        "/tmp/vv_test.db",
    "JWT_SECRET_KEY":        "test_secret_key_32chars_minimum_xxxx",
    "INFLUX_TOKEN":          "",
    "TELEGRAM_BOT_TOKEN":    "",
    "ADMIN_SEED_EMAIL":      "admin@test.in",
    "ADMIN_SEED_PASSWORD":   "Admin@123",
})

from fastapi.testclient import TestClient
from db.database import init_db, get_db
from scripts.seed_db import (seed_panels, seed_admin, seed_demo_users,
                               seed_system_config, seed_bfci_data,
                               seed_cv_data, seed_alerts, seed_relay_log)
from api.main import app

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup():
    if os.path.exists("/tmp/vv_test.db"): os.remove("/tmp/vv_test.db")
    init_db()
    with get_db() as conn:
        for fn in [seed_panels, seed_admin, seed_demo_users, seed_system_config,
                   seed_bfci_data, seed_cv_data, seed_alerts, seed_relay_log]:
            fn(conn)
    yield
    if os.path.exists("/tmp/vv_test.db"): os.remove("/tmp/vv_test.db")


def _login(email, pwd):
    r = client.post("/api/auth/login", json={"email": email, "password": pwd})
    assert r.status_code == 200, f"Login failed: {r.text}"
    return r.json()["access_token"]

def _h(tok): return {"Authorization": f"Bearer {tok}"}


@pytest.fixture(scope="session")
def admin(setup):    return _login("admin@test.in",          "Admin@123")
@pytest.fixture(scope="session")
def operator(setup): return _login("operator@voltarax.in",   "Operator@123")
@pytest.fixture(scope="session")
def viewer(setup):   return _login("viewer@voltarax.in",     "Viewer@123")


# ── Health ────────────────────────────────────────────────────────────────────
def test_root():        assert client.get("/").status_code == 200
def test_health():      assert client.get("/health").json()["status"] == "ok"

# ── Auth ──────────────────────────────────────────────────────────────────────
def test_login_admin():
    r = client.post("/api/auth/login", json={"email":"admin@test.in","password":"Admin@123"})
    assert r.status_code == 200
    assert r.json()["user"]["role"] == "admin"

def test_login_bad_password():
    assert client.post("/api/auth/login", json={"email":"admin@test.in","password":"Wrong1"}).status_code == 401

def test_login_bad_email():
    assert client.post("/api/auth/login", json={"email":"nope@x.com","password":"Admin@123"}).status_code == 401

def test_me(admin):
    r = client.get("/api/auth/me", headers=_h(admin))
    assert r.status_code == 200
    assert r.json()["email"] == "admin@test.in"

def test_me_no_token():   assert client.get("/api/auth/me").status_code in (401, 403)
def test_me_bad_token():  assert client.get("/api/auth/me", headers=_h("bad.tok.here")).status_code == 401

def test_refresh():
    d  = client.post("/api/auth/login", json={"email":"operator@voltarax.in","password":"Operator@123"}).json()
    r2 = client.post("/api/auth/refresh", json={"refresh_token": d["refresh_token"]})
    assert r2.status_code == 200 and "access_token" in r2.json()

def test_refresh_bad():
    assert client.post("/api/auth/refresh", json={"refresh_token":"bad"}).status_code == 401

def test_logout():
    d  = client.post("/api/auth/login", json={"email":"viewer@voltarax.in","password":"Viewer@123"}).json()
    r  = client.post("/api/auth/logout", json={"refresh_token":d["refresh_token"]},
                     headers=_h(d["access_token"]))
    assert r.status_code == 200
    r2 = client.post("/api/auth/refresh", json={"refresh_token":d["refresh_token"]})
    assert r2.status_code == 401

# ── RBAC ──────────────────────────────────────────────────────────────────────
def test_admin_can_list_users(admin):    assert client.get("/api/users", headers=_h(admin)).status_code == 200
def test_operator_no_users(operator):   assert client.get("/api/users", headers=_h(operator)).status_code == 403
def test_viewer_no_users(viewer):       assert client.get("/api/users", headers=_h(viewer)).status_code == 403
def test_viewer_can_panels(viewer):     assert client.get("/api/panels", headers=_h(viewer)).status_code == 200
def test_viewer_no_relay_post(viewer):
    assert client.post("/api/relay", headers=_h(viewer),
                       json={"battery_on":True,"load_on":True,"grid_export":False,"spare_on":False}).status_code == 403
def test_operator_relay_post(operator):
    assert client.post("/api/relay", headers=_h(operator),
                       json={"battery_on":True,"load_on":True,"grid_export":False,"spare_on":False,"reason":"test"}).status_code == 200

# ── Users ─────────────────────────────────────────────────────────────────────
def test_list_users(admin):
    r = client.get("/api/users", headers=_h(admin))
    assert r.status_code == 200 and r.json()["total"] >= 3

def test_create_user(admin):
    r = client.post("/api/users", headers=_h(admin),
                    json={"email":"new@test.in","password":"NewUser@1","full_name":"New User","role":"viewer"})
    assert r.status_code == 201 and r.json()["role"] == "viewer"

def test_create_user_dup(admin):
    r = client.post("/api/users", headers=_h(admin),
                    json={"email":"admin@test.in","password":"Admin@123","full_name":"D","role":"viewer"})
    assert r.status_code in (409, 422)  # 409=conflict if passes validation, 422=validation error first

def test_create_user_weak_pass(admin):
    assert client.post("/api/users", headers=_h(admin),
                       json={"email":"weak@test.in","password":"weak","full_name":"Weak","role":"viewer"}).status_code == 422

def test_search_users(admin):
    assert client.get("/api/users?search=admin", headers=_h(admin)).json()["total"] >= 1

def test_cannot_delete_self(admin):
    me = client.get("/api/auth/me", headers=_h(admin)).json()
    assert client.delete(f"/api/users/{me['id']}", headers=_h(admin)).status_code == 400

# ── Panels ────────────────────────────────────────────────────────────────────
def test_list_panels(viewer):
    r = client.get("/api/panels", headers=_h(viewer))
    assert r.status_code == 200 and len(r.json()) == 32

def test_panel_fields(viewer):
    panels = client.get("/api/panels", headers=_h(viewer)).json()
    for p in panels:
        assert "label" in p and "status" in p and "efficiency_pct" in p

def test_panel_by_row(viewer):
    r = client.get("/api/panels?row=A", headers=_h(viewer))
    assert r.status_code == 200 and len(r.json()) == 8
    for p in r.json(): assert p["row_label"] == "A"

def test_panel_detail(viewer):
    r = client.get("/api/panels/1", headers=_h(viewer))
    assert r.status_code == 200 and r.json()["id"] == 1

def test_panel_404(viewer):
    assert client.get("/api/panels/99999", headers=_h(viewer)).status_code == 404

def test_panels_no_auth():
    assert client.get("/api/panels").status_code in (401, 403)

# ── Alerts ────────────────────────────────────────────────────────────────────
def test_list_alerts(viewer):
    assert len(client.get("/api/alerts", headers=_h(viewer)).json()) >= 4

def test_unresolved_alerts(viewer):
    for a in client.get("/api/alerts?resolved=false", headers=_h(viewer)).json():
        assert a["resolved"] == False

def test_critical_alerts(viewer):
    for a in client.get("/api/alerts?severity=critical", headers=_h(viewer)).json():
        assert a["severity"] == "critical"

def test_resolve_alert(operator):
    r = client.put("/api/alerts/1/resolve", headers=_h(operator), json={"resolved":True})
    assert r.status_code == 200 and r.json()["resolved"] == True

def test_viewer_no_resolve(viewer):
    assert client.put("/api/alerts/2/resolve", headers=_h(viewer), json={"resolved":True}).status_code == 403

def test_alert_404(viewer):
    assert client.get("/api/alerts/99999", headers=_h(viewer)).status_code == 404

# ── Work Orders ───────────────────────────────────────────────────────────────
def test_create_wo(operator):
    r = client.post("/api/work-orders", headers=_h(operator),
                    json={"panel_id":4,"title":"Inspect B4 crack","priority":"critical"})
    assert r.status_code == 201 and r.json()["priority"] == "critical"

def test_list_wo(viewer):
    assert isinstance(client.get("/api/work-orders", headers=_h(viewer)).json(), list)

def test_update_wo(operator):
    r = client.post("/api/work-orders", headers=_h(operator),
                    json={"title":"WO for update test","priority":"medium"})
    wid = r.json()["id"]
    r2  = client.put(f"/api/work-orders/{wid}", headers=_h(operator), json={"status":"in_progress"})
    assert r2.status_code == 200 and r2.json()["status"] == "in_progress"

def test_viewer_no_wo(viewer):
    assert client.post("/api/work-orders", headers=_h(viewer),
                       json={"title":"V WO","priority":"low"}).status_code == 403

def test_wo_bad_panel(operator):
    assert client.post("/api/work-orders", headers=_h(operator),
                       json={"panel_id":9999,"title":"Bad panel WO","priority":"low"}).status_code == 404

# ── BFCI ─────────────────────────────────────────────────────────────────────
def test_bfci_fleet(viewer):
    r = client.get("/api/bfci", headers=_h(viewer))
    assert r.status_code == 200
    fleet = r.json()
    assert len(fleet) == 32
    for e in fleet:
        assert 0 <= e["bfci_score"] <= 100
        assert e["status"] in ["good","warning","critical"]

def test_bfci_panel(viewer):
    r = client.get("/api/bfci/1", headers=_h(viewer))
    assert r.status_code == 200
    d = r.json()
    assert "latest" in d and "history" in d and "shap_values" in d

def test_bfci_record(operator):
    r = client.post("/api/bfci/1", headers=_h(operator), params={"bfci_score":72.5,"batch_id":"MANGO-#7"})
    assert r.status_code == 201 and r.json()["bfci_score"] == 72.5

def test_bfci_bad_score(operator):
    assert client.post("/api/bfci/1", headers=_h(operator), params={"bfci_score":150}).status_code == 422

def test_shap(viewer):
    r = client.get("/api/bfci/shap/1", headers=_h(viewer))
    assert r.status_code == 200 and "shap" in r.json()

# ── Forecast ──────────────────────────────────────────────────────────────────
def test_forecast(viewer):
    r = client.get("/api/forecast", headers=_h(viewer))
    d = r.json()
    assert r.status_code == 200
    assert d["forecast_1h_kw"] >= 0
    assert len(d["confidence"]) == 4
    assert all(0 <= c <= 1 for c in d["confidence"])

def test_hourly(viewer):
    r = client.get("/api/forecast/hourly", headers=_h(viewer))
    assert r.status_code == 200 and len(r.json()["profile"]) == 24

def test_forecast_history(viewer):
    r = client.get("/api/forecast/history?days=7", headers=_h(viewer))
    assert r.status_code == 200 and len(r.json()["history"]) == 7

def test_forecast_no_auth():
    assert client.get("/api/forecast").status_code in (401, 403)

# ── CV ────────────────────────────────────────────────────────────────────────
def test_cv_list(viewer):
    r = client.get("/api/cv", headers=_h(viewer))
    assert r.status_code == 200
    for d in r.json(): assert d["cv_class"] in ["clean","dusty","cracked","bio_degraded","pending"]

def test_cv_summary(viewer):
    r = client.get("/api/cv/summary", headers=_h(viewer))
    assert "total_panels" in r.json() and "counts" in r.json()

def test_cv_scan_operator(operator):
    assert client.post("/api/cv/scan", headers=_h(operator), json={"panel_ids":[1,2]}).status_code == 200

def test_cv_scan_viewer_denied(viewer):
    assert client.post("/api/cv/scan", headers=_h(viewer), json={}).status_code == 403

# ── Relay ─────────────────────────────────────────────────────────────────────
def test_get_relay(viewer):
    r = client.get("/api/relay", headers=_h(viewer))
    assert r.status_code == 200
    assert "battery_on" in r.json() and "commanded_at" in r.json()

def test_set_relay(operator):
    r = client.post("/api/relay", headers=_h(operator),
                    json={"battery_on":True,"load_on":True,"grid_export":True,"spare_on":False,
                          "reason":"manual_test","soc_pct":85.0})
    assert r.status_code == 200 and r.json()["grid_export"] == True

def test_relay_history(viewer):
    r = client.get("/api/relay/history?limit=5", headers=_h(viewer))
    assert r.status_code == 200 and len(r.json()) <= 5

# ── Carbon ────────────────────────────────────────────────────────────────────
def test_carbon(viewer):
    r = client.get("/api/carbon", headers=_h(viewer))
    d = r.json()
    assert d["emission_factor"] == pytest.approx(0.82, rel=0.01)
    assert d["year_co2_kg"] >= d["month_co2_kg"] >= d["today_co2_kg"]

def test_carbon_monthly(viewer):
    r = client.get("/api/carbon/monthly?months=6", headers=_h(viewer))
    assert len(r.json()["months"]) == 6

# ── System ────────────────────────────────────────────────────────────────────
def test_system_health(viewer):
    r = client.get("/api/system", headers=_h(viewer))
    assert r.status_code == 200
    d = r.json()
    assert d["sqlite_ok"] == True and d["panel_count"] >= 32

def test_system_config_admin_only(viewer):
    assert client.get("/api/system/config", headers=_h(viewer)).status_code == 403

def test_system_config_admin(admin):
    assert client.get("/api/system/config", headers=_h(admin)).status_code == 200

# ── Dashboard ─────────────────────────────────────────────────────────────────
def test_dashboard(viewer):
    r = client.get("/api/dashboard", headers=_h(viewer))
    d = r.json()
    assert r.status_code == 200
    assert d["panels_optimal"] + d["panels_warning"] + d["panels_critical"] + d["panels_offline"] == 32
    assert "relay_state" in d and "forecast_1h_kw" in d

def test_dashboard_no_auth():
    assert client.get("/api/dashboard").status_code in (401, 403)

# ── Security ──────────────────────────────────────────────────────────────────
def test_sql_injection(admin):
    r = client.get("/api/users?search='; DROP TABLE users; --", headers=_h(admin))
    assert r.status_code == 200   # parameterised — safe

def test_idor(viewer):
    me = client.get("/api/auth/me", headers=_h(viewer)).json()
    if me["id"] != 1:
        assert client.get("/api/users/1", headers=_h(viewer)).status_code == 403

def test_expired_token():
    from jose import jwt
    from datetime import datetime, timedelta, timezone
    tok = jwt.encode({"sub":"1","email":"admin@test.in","role":"admin","type":"access",
                      "exp": datetime.now(timezone.utc)-timedelta(minutes=5),
                      "iat": datetime.now(timezone.utc)-timedelta(minutes=35)},
                     "test_secret_key_32chars_minimum_xxxx","HS256")
    assert client.get("/api/auth/me", headers=_h(tok)).status_code == 401

def test_no_stack_trace_404(viewer):
    r = client.get("/api/panels/999999", headers=_h(viewer))
    assert r.status_code == 404 and "Traceback" not in r.text

def test_bfci_boundary(operator):
    for s in [0.0, 100.0]:
        assert client.post("/api/bfci/1", headers=_h(operator), params={"bfci_score":s}).status_code == 201
