import sys, os, random
from datetime import datetime, date, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import init_db, get_db
from config.security import hash_password
from config.settings import settings


def seed_panels(conn):
    if conn.execute("SELECT COUNT(*) FROM panels").fetchone()[0] >= 32:
        print("  ✅ Panels already seeded"); return
    panels = [(f"{chr(65+r)}{c}", chr(65+r), c) for r in range(4) for c in range(1,9)]
    conn.executemany(
        "INSERT OR IGNORE INTO panels (label,row_label,col_number,installed_at) VALUES (?,?,?,?)",
        [(l,r,c,"2025-01-01") for l,r,c in panels]
    )
    print(f"  ✅ {len(panels)} panels seeded (A1–D8)")


def seed_admin(conn):
    if conn.execute("SELECT id FROM users WHERE email=?", (settings.ADMIN_SEED_EMAIL,)).fetchone():
        print(f"  ✅ Admin exists: {settings.ADMIN_SEED_EMAIL}"); return
    conn.execute("INSERT INTO users (email,password_hash,full_name,role) VALUES (?,?,?,?)",
                 (settings.ADMIN_SEED_EMAIL, hash_password(settings.ADMIN_SEED_PASSWORD),
                  "Voltarax Admin","admin"))
    print(f"  ✅ Admin created: {settings.ADMIN_SEED_EMAIL}")


def seed_demo_users(conn):
    for email,pwd,name,role in [
        ("operator@voltarax.in","Operator@123","Site Operator","operator"),
        ("viewer@voltarax.in",  "Viewer@123",  "Dashboard Viewer","viewer"),
    ]:
        if not conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone():
            conn.execute("INSERT INTO users (email,password_hash,full_name,role) VALUES (?,?,?,?)",
                         (email, hash_password(pwd), name, role))
            print(f"  ✅ User created: {email}")
        else:
            print(f"  ✅ User exists: {email}")


def seed_system_config(conn):
    for k,v in [("site.name",settings.SITE_NAME),("site.location",settings.SITE_LOCATION),
                ("site.panel_count",str(settings.SITE_PANEL_COUNT)),
                ("grid.emission_factor",str(settings.GRID_EMISSION_FACTOR_KG_KWH)),
                ("bfci.warn_threshold",str(settings.ALERT_BFCI_WARN_THRESHOLD)),
                ("bfci.crit_threshold",str(settings.ALERT_BFCI_CRIT_THRESHOLD))]:
        conn.execute("INSERT OR REPLACE INTO system_config (key,value) VALUES (?,?)",(k,v))
    print("  ✅ System config seeded")


def seed_bfci_data(conn):
    if conn.execute("SELECT COUNT(*) FROM bfci_logs").fetchone()[0] > 0:
        print("  ✅ BFCI data exists"); return
    panels = conn.execute("SELECT id,label FROM panels").fetchall()
    today  = date.today()
    recs   = []
    for p in panels:
        score = 100.0
        low   = p["label"] in {"A6","D2"}
        warn  = p["label"] in {"B3","C7","D5"}
        for i in range(18):
            score -= random.uniform(1.5,2.5) if low else random.uniform(1,1.8) if warn else random.uniform(0.6,1.4)
            score  = max(50, score)
            rec    = str(today - timedelta(days=(17-i)*5))
            recs.append((p["id"],round(score,1),round(score*.88,1),max(0,int((score-70)*2)),"2025-06-01","MANGO-#6",rec))
    conn.executemany(
        "INSERT INTO bfci_logs (panel_id,bfci_score,uv_absorption_pct,days_to_recoat,coat_date,batch_id,recorded_at) "
        "VALUES (?,?,?,?,?,?,?)", recs
    )
    print(f"  ✅ BFCI seeded ({len(recs)} records)")


def seed_cv_data(conn):
    if conn.execute("SELECT COUNT(*) FROM cv_detections").fetchone()[0] > 0:
        print("  ✅ CV data exists"); return
    panels = conn.execute("SELECT id,label FROM panels").fetchall()
    preset = {"B4":("cracked",0.963),"A6":("dusty",0.941),"D2":("bio_degraded",0.917),
              "C1":("dusty",0.882),"C2":("dusty",0.895),"C3":("dusty",0.872),"C4":("dusty",0.859)}
    eff    = {"clean":0,"dusty":3.1,"cracked":15.4,"bio_degraded":8.7}
    recs   = []
    now    = datetime.now(timezone.utc)
    for p in panels:
        cls,conf = preset.get(p["label"],("clean",round(random.uniform(0.91,0.99),3)))
        recs.append((p["id"],cls,conf,eff.get(cls,0),"mobilenet_v3_small_v1",str(now)))
    conn.executemany(
        "INSERT INTO cv_detections (panel_id,cv_class,confidence,eff_loss_pct,model_ver,scanned_at) VALUES (?,?,?,?,?,?)",
        recs
    )
    print(f"  ✅ CV seeded ({len(recs)} panels)")


def seed_alerts(conn):
    if conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0] > 0:
        print("  ✅ Alerts exist"); return
    def pid(label):
        r = conn.execute("SELECT id FROM panels WHERE label=?", (label,)).fetchone()
        return r["id"] if r else None
    alerts = [
        (pid("B4"),"voltage_drop","critical","Panel B4: crack detected — voltage drop 18%.",0.964,0),
        (pid("C1"),"cv_dusty","warning","Zone C (C1-C4): dust across 4 panels. ~3.1% eff loss.",0.921,0),
        (pid("A6"),"bfci_warning","warning","Panel A6 BFCI 71.3% — recoating in 12 days.",0.900,0),
        (None,"weather","info","IMD: Dust storm advisory Jaipur. LSTM forecast adjusted -38%.",None,0),
        (pid("C3"),"cv_dusty","warning","Panel C3 dust. 2.8% eff loss.",0.882,1),
    ]
    conn.executemany(
        "INSERT INTO alerts (panel_id,alert_type,severity,message,confidence,resolved) VALUES (?,?,?,?,?,?)",
        alerts
    )
    print(f"  ✅ {len(alerts)} alerts seeded")


def seed_relay_log(conn):
    if conn.execute("SELECT COUNT(*) FROM relay_log").fetchone()[0] > 0:
        print("  ✅ Relay log exists"); return
    conn.execute("INSERT INTO relay_log (battery_on,load_on,grid_export,spare_on,reason,commanded_by,soc_pct) "
                 "VALUES (?,?,?,?,?,?,?)", (1,1,0,0,"default_safe_state","system",78.0))
    print("  ✅ Relay log seeded")


def main():
    print("\n🌱 VoltVision AI — Seeder")
    print("=" * 40)
    init_db()
    print("  ✅ Schema ready")
    with get_db() as conn:
        seed_panels(conn)
        seed_admin(conn)
        seed_demo_users(conn)
        seed_system_config(conn)
        seed_bfci_data(conn)
        seed_cv_data(conn)
        seed_alerts(conn)
        seed_relay_log(conn)
    print("\n✅ Done!")
    print(f"   Admin:    {settings.ADMIN_SEED_EMAIL} / {settings.ADMIN_SEED_PASSWORD}")
    print("   Operator: operator@voltarax.in / Operator@123")
    print("   Viewer:   viewer@voltarax.in / Viewer@123")
    print("\n🚀  uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
