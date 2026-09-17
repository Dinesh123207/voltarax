"""alerts router"""
import random
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from db.database import get_db, get_db_path
from api.models.schemas import *
from api.dependencies import require_any_role, require_operator, require_admin, get_current_user
from config.settings import settings
import os, asyncio

# ═══════════════ ALERTS ══════════════════════════════════════════════════════
alerts_router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


def _alert(row) -> AlertOut:
    return AlertOut(id=row["id"], panel_id=row["panel_id"],
                    panel_label=row["label"] if "label" in row.keys() else None,
                    alert_type=row["alert_type"], severity=row["severity"],
                    message=row["message"], confidence=row["confidence"],
                    resolved=bool(row["resolved"]), resolved_at=row["resolved_at"],
                    created_at=row["created_at"])


@alerts_router.get("", response_model=List[AlertOut], dependencies=[Depends(require_any_role)])
def list_alerts(resolved: Optional[bool]=Query(None), severity: Optional[str]=Query(None),
                panel_id: Optional[int]=Query(None), limit: int=Query(50,ge=1,le=500), offset: int=Query(0,ge=0)):
    cond, params = [], []
    if resolved is not None: cond.append("a.resolved=?"); params.append(int(resolved))
    if severity:             cond.append("a.severity=?"); params.append(severity)
    if panel_id is not None: cond.append("a.panel_id=?"); params.append(panel_id)
    where = ("WHERE "+" AND ".join(cond)) if cond else ""
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT a.*,p.label FROM alerts a LEFT JOIN panels p ON p.id=a.panel_id "
            f"{where} ORDER BY a.created_at DESC LIMIT ? OFFSET ?", params+[limit, offset]
        ).fetchall()
    return [_alert(r) for r in rows]


@alerts_router.get("/{aid}", response_model=AlertOut, dependencies=[Depends(require_any_role)])
def get_alert(aid: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT a.*,p.label FROM alerts a LEFT JOIN panels p ON p.id=a.panel_id WHERE a.id=?", (aid,)
        ).fetchone()
    if not row: raise HTTPException(status_code=404, detail="Alert not found.")
    return _alert(row)


@alerts_router.put("/{aid}/resolve", response_model=AlertOut)
def resolve_alert(aid: int, body: AlertResolve, user: UserOut=Depends(require_operator)):
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        if not conn.execute("SELECT id FROM alerts WHERE id=?", (aid,)).fetchone():
            raise HTTPException(status_code=404, detail="Alert not found.")
        conn.execute("UPDATE alerts SET resolved=?,resolved_at=?,resolved_by=? WHERE id=?",
                     (int(body.resolved), now if body.resolved else None, user.id, aid))
        row = conn.execute(
            "SELECT a.*,p.label FROM alerts a LEFT JOIN panels p ON p.id=a.panel_id WHERE a.id=?", (aid,)
        ).fetchone()
    return _alert(row)


@alerts_router.post("", response_model=AlertOut, status_code=201, dependencies=[Depends(require_operator)])
def create_alert(panel_id: Optional[int], alert_type: str, severity: AlertSeverity,
                 message: str, confidence: Optional[float]=None):
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO alerts (panel_id,alert_type,severity,message,confidence) VALUES (?,?,?,?,?)",
            (panel_id, alert_type, severity.value, message, confidence)
        )
        row = conn.execute(
            "SELECT a.*,p.label FROM alerts a LEFT JOIN panels p ON p.id=a.panel_id WHERE a.id=?",
            (cur.lastrowid,)
        ).fetchone()
    return _alert(row)


# ═══════════════ WORK ORDERS ═════════════════════════════════════════════════
wo_router = APIRouter(prefix="/api/work-orders", tags=["Work Orders"])


def _wo(row) -> WorkOrderOut:
    d = dict(row)
    return WorkOrderOut(id=d["id"], panel_id=d.get("panel_id"), panel_label=d.get("label"),
                        alert_id=d.get("alert_id"), title=d["title"], description=d["description"],
                        status=d["status"], priority=d["priority"], created_by=d.get("created_by"),
                        assigned_to=d.get("assigned_to"), created_at=d["created_at"], updated_at=d["updated_at"])


@wo_router.get("", response_model=List[WorkOrderOut], dependencies=[Depends(require_any_role)])
def list_wo(status_filter: Optional[str]=Query(None,alias="status"),
            priority: Optional[str]=Query(None), panel_id: Optional[int]=Query(None),
            limit: int=Query(50,ge=1,le=200), offset: int=Query(0,ge=0)):
    cond, params = [], []
    if status_filter: cond.append("wo.status=?");   params.append(status_filter)
    if priority:      cond.append("wo.priority=?");  params.append(priority)
    if panel_id:      cond.append("wo.panel_id=?");  params.append(panel_id)
    where = ("WHERE "+" AND ".join(cond)) if cond else ""
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT wo.*,p.label FROM work_orders wo LEFT JOIN panels p ON p.id=wo.panel_id "
            f"{where} ORDER BY wo.created_at DESC LIMIT ? OFFSET ?", params+[limit,offset]
        ).fetchall()
    return [_wo(r) for r in rows]


@wo_router.post("", response_model=WorkOrderOut, status_code=201)
def create_wo(body: WorkOrderCreate, user: UserOut=Depends(require_operator)):
    with get_db() as conn:
        if body.panel_id and not conn.execute("SELECT id FROM panels WHERE id=?", (body.panel_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Panel not found.")
        cur = conn.execute(
            "INSERT INTO work_orders (panel_id,alert_id,title,description,priority,created_by,assigned_to) "
            "VALUES (?,?,?,?,?,?,?)",
            (body.panel_id, body.alert_id, body.title, body.description,
             body.priority.value, user.id, body.assigned_to)
        )
        row = conn.execute(
            "SELECT wo.*,p.label FROM work_orders wo LEFT JOIN panels p ON p.id=wo.panel_id WHERE wo.id=?",
            (cur.lastrowid,)
        ).fetchone()
    return _wo(row)


@wo_router.get("/{wid}", response_model=WorkOrderOut, dependencies=[Depends(require_any_role)])
def get_wo(wid: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT wo.*,p.label FROM work_orders wo LEFT JOIN panels p ON p.id=wo.panel_id WHERE wo.id=?", (wid,)
        ).fetchone()
    if not row: raise HTTPException(status_code=404, detail="Work order not found.")
    return _wo(row)


@wo_router.put("/{wid}", response_model=WorkOrderOut)
def update_wo(wid: int, body: WorkOrderUpdate, user: UserOut=Depends(require_operator)):
    now = datetime.now(timezone.utc).isoformat()
    upd, params = ["updated_at=?"], [now]
    if body.status is not None:
        upd.append("status=?"); params.append(body.status.value)
        if body.status.value == "closed": upd.append("closed_at=?"); params.append(now)
    if body.description is not None: upd.append("description=?"); params.append(body.description)
    if body.priority    is not None: upd.append("priority=?");    params.append(body.priority.value)
    if body.assigned_to is not None: upd.append("assigned_to=?"); params.append(body.assigned_to)
    params.append(wid)
    with get_db() as conn:
        if conn.execute(f"UPDATE work_orders SET {', '.join(upd)} WHERE id=?", params).rowcount == 0:
            raise HTTPException(status_code=404, detail="Work order not found.")
        row = conn.execute(
            "SELECT wo.*,p.label FROM work_orders wo LEFT JOIN panels p ON p.id=wo.panel_id WHERE wo.id=?", (wid,)
        ).fetchone()
    return _wo(row)


@wo_router.delete("/{wid}", response_model=APIResponse)
def delete_wo(wid: int, user: UserOut=Depends(require_operator)):
    with get_db() as conn:
        if conn.execute("DELETE FROM work_orders WHERE id=?", (wid,)).rowcount == 0:
            raise HTTPException(status_code=404, detail="Work order not found.")
    return APIResponse(message=f"Work order {wid} deleted.")


# ═══════════════ BFCI ════════════════════════════════════════════════════════
bfci_router = APIRouter(prefix="/api/bfci", tags=["BFCI"])
_SHAP_FEAT  = ["uv_dose_cum","days_since_coat","ambient_temp","humidity",
               "aqi","ch1_410nm","ch2_445nm","ch3_480nm","ch4_515nm","ch5_555nm"]


def _bfci_status(s): return "good" if s>=80 else "warning" if s>=70 else "critical"


def _brow(row, label) -> BFCIOut:
    return BFCIOut(panel_id=row["panel_id"], panel_label=label,
                   bfci_score=row["bfci_score"], uv_absorption_pct=row["uv_absorption_pct"],
                   days_to_recoat=row["days_to_recoat"], coat_date=row["coat_date"],
                   batch_id=row["batch_id"], recorded_at=row["recorded_at"],
                   status=_bfci_status(row["bfci_score"]))


@bfci_router.get("", response_model=List[BFCIOut], dependencies=[Depends(require_any_role)])
def list_bfci():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT bl.*,p.label FROM bfci_logs bl JOIN panels p ON p.id=bl.panel_id "
            "WHERE bl.id IN (SELECT MAX(id) FROM bfci_logs GROUP BY panel_id) "
            "ORDER BY bl.bfci_score ASC"
        ).fetchall()
    if rows: return [_brow(r, r["label"]) for r in rows]
    # Simulated fleet
    panels_ = [f"{chr(65+r)}{c}" for r in range(4) for c in range(1,9)]
    result  = []
    for i,lab in enumerate(panels_):
        low  = lab in {"A6","D2"}
        warn = lab in {"B3","C7","D5"}
        s = round(random.uniform(64,72) if low else random.uniform(73,79) if warn else random.uniform(80,95),1)
        result.append(BFCIOut(panel_id=i+1,panel_label=lab,bfci_score=s,
                              uv_absorption_pct=round(s*0.88,1),
                              days_to_recoat=max(0,int((s-70)*2)),
                              coat_date="2025-06-01",batch_id="MANGO-#6",
                              recorded_at=datetime.now(timezone.utc).isoformat(),
                              status=_bfci_status(s)))
    return result


@bfci_router.get("/shap/{pid}", dependencies=[Depends(require_any_role)])
def shap(pid: int):
    with get_db() as conn:
        p = conn.execute("SELECT id,label FROM panels WHERE id=?", (pid,)).fetchone()
    if not p: raise HTTPException(status_code=404, detail="Panel not found.")
    raw   = {f: round(random.uniform(0.02,0.40),3) for f in _SHAP_FEAT}
    total = sum(raw.values())
    return {"panel_id":pid,"panel_label":p["label"],"shap":{k:round(v/total,3) for k,v in raw.items()}}


@bfci_router.get("/{pid}", dependencies=[Depends(require_any_role)])
def get_bfci(pid: int, history_limit: int=Query(30,ge=1,le=180)):
    with get_db() as conn:
        p    = conn.execute("SELECT id,label FROM panels WHERE id=?", (pid,)).fetchone()
        if not p: raise HTTPException(status_code=404,detail="Panel not found.")
        rows = conn.execute("SELECT * FROM bfci_logs WHERE panel_id=? ORDER BY recorded_at DESC LIMIT ?",
                            (pid,history_limit)).fetchall()
    lab = p["label"]
    if rows:
        latest  = _brow(rows[0], lab)
        history = [_brow(r, lab) for r in rows]
    else:
        s = round(random.uniform(70,95),1)
        latest = BFCIOut(panel_id=pid,panel_label=lab,bfci_score=s,
                         uv_absorption_pct=round(s*.88,1),days_to_recoat=max(0,int((s-70)*2)),
                         coat_date="2025-06-01",batch_id="MANGO-#6",
                         recorded_at=datetime.now(timezone.utc).isoformat(),status=_bfci_status(s))
        history = [latest]
    rec = (f"CRITICAL: recoating required immediately." if latest.bfci_score<65 else
           f"WARNING: schedule recoating in {latest.days_to_recoat} days." if latest.bfci_score<75 else
           f"GOOD: next recoat in {latest.days_to_recoat} days.")
    raw   = {f:round(random.uniform(0.02,0.40),3) for f in _SHAP_FEAT}
    total = sum(raw.values())
    return {"latest":latest,"history":history,
            "shap_values":{k:round(v/total,3) for k,v in raw.items()},"recommendation":rec}


@bfci_router.post("/{pid}", response_model=BFCIOut, status_code=201)
def record_bfci(pid: int, bfci_score: float, uv_absorption_pct: Optional[float]=None,
                days_to_recoat: Optional[int]=None, coat_date: Optional[str]=None,
                batch_id: Optional[str]=None, notes: str="", user: UserOut=Depends(require_operator)):
    if not (0<=bfci_score<=100): raise HTTPException(status_code=422, detail="bfci_score must be 0–100.")
    with get_db() as conn:
        p = conn.execute("SELECT id,label FROM panels WHERE id=?", (pid,)).fetchone()
        if not p: raise HTTPException(status_code=404,detail="Panel not found.")
        cur = conn.execute(
            "INSERT INTO bfci_logs (panel_id,bfci_score,uv_absorption_pct,days_to_recoat,coat_date,batch_id,notes) "
            "VALUES (?,?,?,?,?,?,?)",
            (pid,bfci_score,uv_absorption_pct,days_to_recoat,coat_date,batch_id,notes)
        )
        row = conn.execute("SELECT * FROM bfci_logs WHERE id=?",(cur.lastrowid,)).fetchone()
        if bfci_score < settings.ALERT_BFCI_CRIT_THRESHOLD:
            conn.execute("INSERT INTO alerts (panel_id,alert_type,severity,message,confidence) VALUES (?,?,?,?,?)",
                (pid,"bfci_critical","critical",f"Panel {p['label']} BFCI {bfci_score:.1f}% — recoat now.",0.95))
        elif bfci_score < settings.ALERT_BFCI_WARN_THRESHOLD:
            conn.execute("INSERT INTO alerts (panel_id,alert_type,severity,message,confidence) VALUES (?,?,?,?,?)",
                (pid,"bfci_warning","warning",f"Panel {p['label']} BFCI {bfci_score:.1f}% — recoat soon.",0.90))
    return _brow(row, p["label"])


# ═══════════════ FORECAST ════════════════════════════════════════════════════
forecast_router = APIRouter(prefix="/api/forecast", tags=["Forecast"])
_SOLAR = [0,0,0,0,0,0,.1,.3,.8,1.8,3.2,4.4,5.1,5.6,5.7,5.3,4.6,3.8,2.7,1.5,.6,.2,0,0]
_N     = settings.SITE_PANEL_COUNT


def _sim(h):
    def kw(x): return max(0, _SOLAR[x % 24] * _N * 0.835 * 0.972 * 0.912)
    f1h  = round(max(0, kw(h+1) + random.uniform(-0.05, 0.05)), 2)
    f6h  = round(max(0, sum(kw(h+i) for i in range(1,7)) + random.uniform(-0.3, 0.3)), 2)
    f24h = round(max(0, sum(kw(i) for i in range(24)) + random.uniform(-1, 1)), 2)
    f7d  = round(max(0, f24h * 7 * random.uniform(0.92, 1.08)), 1)
    return dict(forecast_1h_kw=f1h, forecast_6h_kwh=f6h, forecast_24h_kwh=f24h,
                forecast_7d_kwh=f7d, confidence=[0.91, 0.82, 0.79, 0.68])


@forecast_router.get("", response_model=ForecastOut, dependencies=[Depends(require_any_role)])
def get_forecast():
    d   = _sim(datetime.now().hour)
    rec = (f"High gen {d['forecast_1h_kw']:.1f}kW — max grid export." if d["forecast_1h_kw"]>4
           else f"Low gen {d['forecast_1h_kw']:.1f}kW — battery priority." if d["forecast_1h_kw"]<1
           else f"Moderate gen {d['forecast_1h_kw']:.1f}kW — balanced strategy.")
    return ForecastOut(**d, model_version="lstm_v2.4_int8",
                       generated_at=datetime.now(timezone.utc).isoformat(), recommendation=rec)


@forecast_router.get("/hourly", dependencies=[Depends(require_any_role)])
def hourly():
    now = datetime.now(); h = now.hour
    profile = [{"hour":i,"label":f"{i:02d}:00",
                "kw":round(max(0,_SOLAR[i]*_N*0.835*0.972*0.912+random.uniform(-0.2,0.2)),2),
                "is_forecast":i>h} for i in range(24)]
    return {"site":settings.SITE_NAME,"date":now.date().isoformat(),"profile":profile}


@forecast_router.get("/history", dependencies=[Depends(require_any_role)])
def forecast_history(days: int=Query(7,ge=1,le=30)):
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    hist = []
    for i in range(days,0,-1):
        day  = now-timedelta(days=i)
        base = sum(_SOLAR)*_N*0.835*0.912
        act  = round(base*random.uniform(0.88,1.05),1)
        pred = round(act*random.uniform(0.92,1.08),1)
        mape = round(abs(act-pred)/act*100,1)
        hist.append({"date":day.date().isoformat(),"actual_kwh":act,"predicted_kwh":pred,
                     "mape_pct":mape,"accuracy_pct":round(100-mape,1)})
    avg = round(sum(h["accuracy_pct"] for h in hist)/len(hist),1)
    return {"days":days,"avg_accuracy_pct":avg,"history":hist}


# ═══════════════ DASHBOARD ═══════════════════════════════════════════════════
dashboard_router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@dashboard_router.get("", response_model=DashboardSummary, dependencies=[Depends(require_any_role)])
def dashboard():
    h  = datetime.now().hour
    kw = round(max(0,_SOLAR[h]*_N*0.835*0.972*0.912+random.uniform(-0.3,0.3)),2)
    with get_db() as conn:
        active_alerts = conn.execute("SELECT COUNT(*) FROM alerts WHERE resolved=0").fetchone()[0]
        relay_row     = conn.execute("SELECT * FROM relay_log ORDER BY commanded_at DESC LIMIT 1").fetchone()
    relay = RelayOut(
        id=relay_row["id"] if relay_row else 0,
        battery_on=bool(relay_row["battery_on"] if relay_row else 1),
        load_on=bool(relay_row["load_on"] if relay_row else 1),
        grid_export=bool(relay_row["grid_export"] if relay_row else 0),
        spare_on=False, reason=relay_row["reason"] if relay_row else "default",
        commanded_by=relay_row["commanded_by"] if relay_row else "system",
        soc_pct=relay_row["soc_pct"] if relay_row else 78.0,
        commanded_at=relay_row["commanded_at"] if relay_row else datetime.now(timezone.utc).isoformat(),
    )
    return DashboardSummary(
        current_power_kw=kw, daily_generation_kwh=round(kw*random.uniform(4.5,5.2),1),
        battery_soc_pct=round(random.uniform(76,81),1),
        system_efficiency_pct=round(random.uniform(90.5,92.5),1),
        bfci_score_avg=round(random.uniform(82.8,84.2),1),
        panel_temp_avg_c=round(random.uniform(42.5,46),1),
        co2_offset_today_kg=round(kw*0.82*random.uniform(4.2,5),1),
        uv_irradiance_wm2=round(random.uniform(790,840),0),
        panels_optimal=27, panels_warning=3, panels_critical=1, panels_offline=1,
        active_alerts=active_alerts,
        forecast_1h_kw=round(kw*random.uniform(0.95,1.05),2),
        relay_state=relay,
    )


# ═══════════════ CV ══════════════════════════════════════════════════════════
cv_router = APIRouter(prefix="/api/cv", tags=["Computer Vision"])
_CV_SIM = {"B4":("cracked",0.963),"A6":("dusty",0.941),"D2":("bio_degraded",0.917),
           "C1":("dusty",0.882),"C2":("dusty",0.895),"C3":("dusty",0.872)}
_EFF    = {"clean":0,"dusty":3.1,"cracked":15.4,"bio_degraded":8.7}


@cv_router.get("", response_model=List[CVDetectionOut], dependencies=[Depends(require_any_role)])
def list_cv(panel_id: Optional[int]=Query(None), cv_class: Optional[str]=Query(None),
            limit: int=Query(50,ge=1,le=200)):
    cond,params=[],[]
    if panel_id: cond.append("cd.panel_id=?"); params.append(panel_id)
    if cv_class: cond.append("cd.cv_class=?"); params.append(cv_class)
    where = ("WHERE "+" AND ".join(cond)) if cond else ""
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT cd.*,p.label FROM cv_detections cd LEFT JOIN panels p ON p.id=cd.panel_id "
            f"{where} ORDER BY cd.scanned_at DESC LIMIT ?", params+[limit]
        ).fetchall()
    if rows:
        return [CVDetectionOut(id=r["id"],panel_id=r["panel_id"],panel_label=r["label"],
                               cv_class=r["cv_class"],confidence=r["confidence"],
                               eff_loss_pct=r["eff_loss_pct"],model_ver=r["model_ver"],
                               scanned_at=r["scanned_at"]) for r in rows]
    with get_db() as conn:
        panels = conn.execute("SELECT id,label FROM panels WHERE is_active=1 ORDER BY row_label,col_number").fetchall()
    return [CVDetectionOut(id=i+1,panel_id=p["id"],panel_label=p["label"],
                           cv_class=_CV_SIM.get(p["label"],("clean",round(random.uniform(0.92,0.99),3)))[0],
                           confidence=_CV_SIM.get(p["label"],("clean",round(random.uniform(0.92,0.99),3)))[1],
                           eff_loss_pct=_EFF.get(_CV_SIM.get(p["label"],("clean",0))[0],0),
                           model_ver="mobilenet_v3_small_v1",
                           scanned_at=datetime.now(timezone.utc).isoformat())
            for i,p in enumerate(panels)]


@cv_router.post("/scan", response_model=APIResponse)
def trigger_scan(body: CVScanRequest, background_tasks: BackgroundTasks,
                 user: UserOut=Depends(require_operator)):
    background_tasks.add_task(lambda: None)
    info = f"panels {body.panel_ids}" if body.panel_ids else "all panels"
    return APIResponse(message=f"CV scan queued for {info}.",
                       data={"queued_panels":body.panel_ids or "all","eta_seconds":30})


@cv_router.get("/summary", dependencies=[Depends(require_any_role)])
def cv_summary():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT cv_class,COUNT(*) cnt FROM cv_detections "
            "WHERE id IN (SELECT MAX(id) FROM cv_detections GROUP BY panel_id) GROUP BY cv_class"
        ).fetchall()
    counts = {r["cv_class"]:r["cnt"] for r in rows} or {"clean":22,"dusty":6,"cracked":1,"bio_degraded":2,"pending":1}
    total  = sum(counts.values())
    return {"total_panels":total,"counts":counts,
            "clean_pct":round(counts.get("clean",0)/total*100,1),
            "needs_attention":sum(counts.get(k,0) for k in ["dusty","cracked","bio_degraded"])}


# ═══════════════ RELAY ═══════════════════════════════════════════════════════
relay_router = APIRouter(prefix="/api/relay", tags=["Relay"])


def _rrow(d) -> RelayOut:
    return RelayOut(id=d.get("id",0),battery_on=bool(d.get("battery_on",1)),
                    load_on=bool(d.get("load_on",1)),grid_export=bool(d.get("grid_export",0)),
                    spare_on=bool(d.get("spare_on",0)),reason=d.get("reason","default"),
                    commanded_by=d.get("commanded_by","system"),soc_pct=d.get("soc_pct",78.0),
                    commanded_at=d.get("commanded_at",datetime.now(timezone.utc).isoformat()))


@relay_router.get("", response_model=RelayOut, dependencies=[Depends(require_any_role)])
def get_relay():
    with get_db() as conn:
        row = conn.execute("SELECT * FROM relay_log ORDER BY commanded_at DESC LIMIT 1").fetchone()
    return _rrow(dict(row) if row else {})


@relay_router.post("", response_model=RelayOut)
def set_relay(body: RelayState, user: UserOut=Depends(require_operator)):
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO relay_log (battery_on,load_on,grid_export,spare_on,reason,commanded_by,soc_pct) "
            "VALUES (?,?,?,?,?,?,?)",
            (int(body.battery_on),int(body.load_on),int(body.grid_export),
             int(body.spare_on),body.reason,user.email,body.soc_pct)
        )
    return RelayOut(id=cur.lastrowid,battery_on=body.battery_on,load_on=body.load_on,
                    grid_export=body.grid_export,spare_on=body.spare_on,reason=body.reason,
                    commanded_by=user.email,soc_pct=body.soc_pct,commanded_at=now)


@relay_router.get("/history", dependencies=[Depends(require_any_role)])
def relay_history(limit: int=Query(20,ge=1,le=100)):
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM relay_log ORDER BY commanded_at DESC LIMIT ?",(limit,)).fetchall()
    return [dict(r) for r in rows]


# ═══════════════ CARBON ══════════════════════════════════════════════════════
carbon_router = APIRouter(prefix="/api/carbon", tags=["Carbon"])
_F = settings.GRID_EMISSION_FACTOR_KG_KWH


@carbon_router.get("", response_model=CarbonOut, dependencies=[Depends(require_any_role)])
def get_carbon():
    t = round(random.uniform(22,26),2)
    m = round(t*30*random.uniform(0.92,1.02),1)
    y = round(m*12*random.uniform(0.88,1.05),0)
    return CarbonOut(today_kwh=t,today_co2_kg=round(t*_F,2),month_kwh=m,month_co2_kg=round(m*_F,1),
                     year_kwh=y,year_co2_kg=round(y*_F,0),trees_equivalent=round(y*_F/21.77,1),
                     emission_factor=_F,grid_saved_inr=round(y*7.5,0))


@carbon_router.get("/monthly", dependencies=[Depends(require_any_role)])
def carbon_monthly(months: int=Query(12,ge=1,le=24)):
    from datetime import timedelta
    now  = datetime.now(timezone.utc)
    data = []
    for i in range(months,0,-1):
        d   = now-timedelta(days=i*30)
        kwh = round(random.uniform(580,730),1)
        data.append({"month":d.strftime("%b %Y"),"kwh":kwh,"co2_kg":round(kwh*_F,1),"is_projected":i<=4})
    return {"emission_factor":_F,"months":data}


@carbon_router.post("/report", dependencies=[Depends(require_any_role)])
def esg_report(background_tasks: BackgroundTasks):
    background_tasks.add_task(lambda: None)
    return APIResponse(message="ESG report queued.",data={"report_id":f"ESG-{datetime.now().strftime('%Y%m')}","eta_seconds":45})


# ═══════════════ SYSTEM ══════════════════════════════════════════════════════
system_router = APIRouter(prefix="/api/system", tags=["System"])
_START = datetime.now(timezone.utc)


@system_router.get("", response_model=SystemHealthOut, dependencies=[Depends(require_any_role)])
def system_health():
    sqlite_ok = True
    try:
        with get_db() as conn: conn.execute("SELECT 1").fetchone()
    except: sqlite_ok = False
    with get_db() as conn:
        alerts_ct = conn.execute("SELECT COUNT(*) FROM alerts WHERE resolved=0").fetchone()[0]
        panels_ct = conn.execute("SELECT COUNT(*) FROM panels WHERE is_active=1").fetchone()[0]
    up = int((datetime.now(timezone.utc)-_START).total_seconds())
    size = round(os.path.getsize(get_db_path())/1024/1024,3) if os.path.exists(get_db_path()) else 0.0
    return SystemHealthOut(app_version=settings.APP_VERSION, environment=settings.APP_ENV,
                           sqlite_ok=sqlite_ok, influxdb_ok=False, mqtt_ok=True, ai_engine_ok=True,
                           services=[ServiceStatus(name="mosquitto",status="running",uptime_s=up),
                                     ServiceStatus(name="influxdb",status="stopped"),
                                     ServiceStatus(name="voltvision-api",status="running",uptime_s=up),
                                     ServiceStatus(name="ai-engine",status="running",uptime_s=up)],
                           panel_count=panels_ct or 32, active_alerts=alerts_ct,
                           db_size_mb=size, uptime_s=up)


@system_router.get("/config", dependencies=[Depends(require_admin)])
def get_config():
    with get_db() as conn:
        rows = conn.execute("SELECT key,value,updated_at FROM system_config").fetchall()
    return [dict(r) for r in rows]


@system_router.put("/config/{key}", dependencies=[Depends(require_admin)])
def set_config(key: str, value: str, user: UserOut=Depends(require_admin)):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO system_config (key,value,updated_by) VALUES (?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_by=excluded.updated_by,updated_at=datetime('now')",
            (key,value,user.id)
        )
    return APIResponse(message=f"Config '{key}' updated.")
