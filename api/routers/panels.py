import random
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from db.database import get_db
from api.models.schemas import PanelOut, PanelDetail
from api.dependencies import require_any_role, require_operator
from api.models.schemas import UserOut

router = APIRouter(prefix="/api/panels", tags=["Panels"])

_SOLAR  = [0,0,0,0,0,0,.1,.3,.8,1.8,3.2,4.4,5.1,5.6,5.7,5.3,4.6,3.8,2.7,1.5,.6,.2,0,0]
_CRIT   = {"B4"}
_WARN   = {"A6","D2","C3"}
_OFFLN  = {"D8"}


def _metrics(label: str) -> dict:
    if label in _OFFLN:
        return dict(voltage_v=0,current_a=0,power_w=0,temp_c=22.0,efficiency_pct=0)
    h = datetime.now().hour
    base = _SOLAR[h]
    f = 0.55 if label in _CRIT else 0.78 if label in _WARN else 1.0
    v = round(36.8*f*(1+random.uniform(-0.02,0.02)), 2)
    i = round(13.2*f*(1+random.uniform(-0.02,0.02)), 2)
    return dict(voltage_v=v, current_a=i, power_w=round(v*i,1),
                temp_c=round(44.2+(1-f)*12+random.uniform(-1,1),1),
                efficiency_pct=round(92.4*f+random.uniform(-1,1),1))


def _status(eff: float) -> str:
    if eff >= 88: return "optimal"
    if eff >= 70: return "warning"
    if eff > 0:   return "critical"
    return "offline"


def _bfci(panel_id: int) -> Optional[float]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT bfci_score FROM bfci_logs WHERE panel_id=? ORDER BY recorded_at DESC LIMIT 1",
            (panel_id,)
        ).fetchone()
    return row["bfci_score"] if row else None


def _cv(panel_id: int) -> dict:
    with get_db() as conn:
        row = conn.execute(
            "SELECT cv_class,confidence FROM cv_detections WHERE panel_id=? ORDER BY scanned_at DESC LIMIT 1",
            (panel_id,)
        ).fetchone()
    return {"cv_class": row["cv_class"], "cv_confidence": row["confidence"]} if row else {"cv_class": "pending", "cv_confidence": None}


@router.get("", response_model=List[PanelDetail], dependencies=[Depends(require_any_role)])
def list_panels(status_filter: Optional[str] = Query(None, alias="status"),
                row: Optional[str] = Query(None)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id,label,row_label,col_number,is_active,installed_at,notes FROM panels "
            "WHERE is_active=1 ORDER BY row_label,col_number"
        ).fetchall()
    result = []
    for r in rows:
        m  = _metrics(r["label"])
        cv = _cv(r["id"])
        eff = m["efficiency_pct"]
        st  = _status(eff)
        if status_filter and st != status_filter: continue
        if row and r["row_label"] != row.upper(): continue
        result.append(PanelDetail(
            id=r["id"], label=r["label"], row_label=r["row_label"], col_number=r["col_number"],
            is_active=bool(r["is_active"]), installed_at=r["installed_at"], notes=r["notes"],
            efficiency_pct=eff, bfci_score=_bfci(r["id"]),
            voltage_v=m["voltage_v"], current_a=m["current_a"], power_w=m["power_w"],
            temp_c=m["temp_c"], cv_class=cv["cv_class"], cv_confidence=cv["cv_confidence"], status=st,
        ))
    return result


@router.get("/{pid}", response_model=PanelDetail, dependencies=[Depends(require_any_role)])
def get_panel(pid: int):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id,label,row_label,col_number,is_active,installed_at,notes FROM panels WHERE id=?", (pid,)
        ).fetchone()
    if not row: raise HTTPException(status_code=404, detail="Panel not found.")
    m  = _metrics(row["label"])
    cv = _cv(row["id"])
    eff = m["efficiency_pct"]
    return PanelDetail(
        id=row["id"], label=row["label"], row_label=row["row_label"], col_number=row["col_number"],
        is_active=bool(row["is_active"]), installed_at=row["installed_at"], notes=row["notes"],
        efficiency_pct=eff, bfci_score=_bfci(row["id"]),
        voltage_v=m["voltage_v"], current_a=m["current_a"], power_w=m["power_w"],
        temp_c=m["temp_c"], cv_class=cv["cv_class"], cv_confidence=cv["cv_confidence"],
        status=_status(eff),
    )


@router.post("", response_model=PanelOut, status_code=201)
def create_panel(label: str, row_label: str, col_number: int, notes: str = "",
                 user: UserOut = Depends(require_operator)):
    label = label.upper()
    with get_db() as conn:
        if conn.execute("SELECT id FROM panels WHERE label=?", (label,)).fetchone():
            raise HTTPException(status_code=409, detail=f"Panel {label} exists.")
        cur = conn.execute(
            "INSERT INTO panels (label,row_label,col_number,notes) VALUES (?,?,?,?)",
            (label, row_label.upper(), col_number, notes)
        )
        row = conn.execute(
            "SELECT id,label,row_label,col_number,is_active,installed_at,notes FROM panels WHERE id=?",
            (cur.lastrowid,)
        ).fetchone()
    return PanelOut(**dict(row))
