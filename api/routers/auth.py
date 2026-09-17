from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Request, status, Depends
from config.security import verify_password, create_access_token, create_refresh_token, hash_refresh_token
from db.database import get_db
from api.models.schemas import LoginRequest, LoginResponse, RefreshRequest, TokenResponse, UserOut, APIResponse
from api.dependencies import get_current_user

router = APIRouter(prefix="/api/auth", tags=["Auth"])


def _now(): return datetime.now(timezone.utc).isoformat()


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, request: Request):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id,email,full_name,role,is_active,password_hash,created_at FROM users WHERE email=?",
            (body.email.lower(),)
        ).fetchone()

    _err = HTTPException(status_code=401, detail="Invalid email or password.")
    if not row: raise _err
    if not row["is_active"]: raise HTTPException(status_code=403, detail="Account deactivated.")
    if not verify_password(body.password, row["password_hash"]): raise _err

    uid = row["id"]
    access = create_access_token(uid, row["email"], row["role"])
    raw_rt, rt_hash, rt_exp = create_refresh_token()

    with get_db() as conn:
        conn.execute("INSERT INTO refresh_tokens (user_id,token_hash,expires_at) VALUES (?,?,?)",
                     (uid, rt_hash, rt_exp.isoformat()))
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (_now(), uid))
        conn.execute("INSERT INTO audit_log (user_id,action,resource,detail,ip_address) VALUES (?,?,?,?,?)",
                     (uid, "login", "auth", "Login", request.client.host if request.client else ""))

    return LoginResponse(
        access_token=access, refresh_token=raw_rt,
        user=UserOut(id=uid, email=row["email"], full_name=row["full_name"],
                     role=row["role"], is_active=bool(row["is_active"]),
                     created_at=row["created_at"], last_login=_now())
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh(body: RefreshRequest):
    h = hash_refresh_token(body.refresh_token)
    with get_db() as conn:
        row = conn.execute(
            "SELECT rt.*,u.email,u.role,u.is_active FROM refresh_tokens rt "
            "JOIN users u ON u.id=rt.user_id WHERE rt.token_hash=?", (h,)
        ).fetchone()
    if not row or row["revoked"]:
        raise HTTPException(status_code=401, detail="Invalid refresh token.")
    from datetime import datetime
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired.")
    if not row["is_active"]:
        raise HTTPException(status_code=403, detail="Account deactivated.")
    return TokenResponse(access_token=create_access_token(row["user_id"], row["email"], row["role"]))


@router.post("/logout", response_model=APIResponse)
def logout(body: RefreshRequest, user: UserOut = Depends(get_current_user)):
    h = hash_refresh_token(body.refresh_token)
    with get_db() as conn:
        conn.execute("UPDATE refresh_tokens SET revoked=1 WHERE token_hash=? AND user_id=?", (h, user.id))
        conn.execute("INSERT INTO audit_log (user_id,action,resource,detail) VALUES (?,?,?,?)",
                     (user.id, "logout", "auth", "Logout"))
    return APIResponse(message="Logged out.")


@router.get("/me", response_model=UserOut)
def me(user: UserOut = Depends(get_current_user)):
    return user
