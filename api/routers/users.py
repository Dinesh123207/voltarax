from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import Optional
from config.security import hash_password
from db.database import get_db
from api.models.schemas import UserCreate, UserUpdate, UserOut, APIResponse, PaginatedResponse
from api.dependencies import require_admin, get_current_user

router = APIRouter(prefix="/api/users", tags=["Users"])


@router.get("", response_model=PaginatedResponse, dependencies=[Depends(require_admin)])
def list_users(page: int = Query(1, ge=1), per_page: int = Query(20, ge=1, le=100),
               role: Optional[str] = None, search: Optional[str] = None):
    offset = (page - 1) * per_page
    cond, params = [], []
    if role:   cond.append("role=?");                    params.append(role)
    if search: cond.append("(email LIKE ? OR full_name LIKE ?)"); params += [f"%{search}%"]*2
    where = ("WHERE " + " AND ".join(cond)) if cond else ""
    with get_db() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM users {where}", params).fetchone()[0]
        rows  = conn.execute(
            f"SELECT id,email,full_name,role,is_active,created_at,last_login FROM users "
            f"{where} ORDER BY created_at DESC LIMIT ? OFFSET ?", params+[per_page, offset]
        ).fetchall()
    return PaginatedResponse(total=total, page=page, per_page=per_page, data=[dict(r) for r in rows])


@router.post("", response_model=UserOut, status_code=201, dependencies=[Depends(require_admin)])
def create_user(body: UserCreate):
    with get_db() as conn:
        if conn.execute("SELECT id FROM users WHERE email=?", (body.email.lower(),)).fetchone():
            raise HTTPException(status_code=409, detail="Email already exists.")
        cur = conn.execute(
            "INSERT INTO users (email,password_hash,full_name,role) VALUES (?,?,?,?)",
            (body.email.lower(), hash_password(body.password), body.full_name, body.role.value)
        )
        row = conn.execute("SELECT id,email,full_name,role,is_active,created_at,last_login FROM users WHERE id=?",
                           (cur.lastrowid,)).fetchone()
    return UserOut(**dict(row))


@router.get("/{uid}", response_model=UserOut)
def get_user(uid: int, caller: UserOut = Depends(get_current_user)):
    if caller.role != "admin" and caller.id != uid:
        raise HTTPException(status_code=403, detail="Access denied.")
    with get_db() as conn:
        row = conn.execute("SELECT id,email,full_name,role,is_active,created_at,last_login FROM users WHERE id=?",
                           (uid,)).fetchone()
    if not row: raise HTTPException(status_code=404, detail="User not found.")
    return UserOut(**dict(row))


@router.put("/{uid}", response_model=UserOut)
def update_user(uid: int, body: UserUpdate, caller: UserOut = Depends(get_current_user)):
    if caller.role != "admin":
        if caller.id != uid: raise HTTPException(status_code=403, detail="Access denied.")
        if body.role or body.is_active is not None:
            raise HTTPException(status_code=403, detail="Only admins can change role/status.")
    updates, params = ["updated_at=datetime('now')"], []
    if body.full_name is not None: updates.append("full_name=?"); params.append(body.full_name)
    if body.role      is not None: updates.append("role=?");      params.append(body.role.value)
    if body.is_active is not None: updates.append("is_active=?"); params.append(int(body.is_active))
    params.append(uid)
    with get_db() as conn:
        conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=?", params)
        row = conn.execute("SELECT id,email,full_name,role,is_active,created_at,last_login FROM users WHERE id=?",
                           (uid,)).fetchone()
    if not row: raise HTTPException(status_code=404, detail="User not found.")
    return UserOut(**dict(row))


@router.delete("/{uid}", response_model=APIResponse, dependencies=[Depends(require_admin)])
def delete_user(uid: int, caller: UserOut = Depends(require_admin)):
    if caller.id == uid: raise HTTPException(status_code=400, detail="Cannot delete yourself.")
    with get_db() as conn:
        if conn.execute("DELETE FROM users WHERE id=?", (uid,)).rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found.")
    return APIResponse(message=f"User {uid} deleted.")
