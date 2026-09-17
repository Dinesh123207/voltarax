from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.security import decode_access_token
from db.database import get_db
from api.models.schemas import UserOut, UserRole

_bearer = HTTPBearer(auto_error=True)


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> UserOut:
    payload = decode_access_token(creds.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token.",
                            headers={"WWW-Authenticate": "Bearer"})
    with get_db() as conn:
        row = conn.execute(
            "SELECT id,email,full_name,role,is_active,created_at,last_login "
            "FROM users WHERE id=?", (int(payload["sub"]),)
        ).fetchone()
    if not row or not row["is_active"]:
        raise HTTPException(status_code=401, detail="User not found or deactivated.")
    return UserOut(**dict(row))


def require_roles(*roles: UserRole):
    def _check(user: UserOut = Depends(get_current_user)) -> UserOut:
        if user.role not in roles:
            raise HTTPException(status_code=403,
                detail=f"Required role(s): {[r.value for r in roles]}")
        return user
    return _check


def require_admin(user: UserOut = Depends(require_roles(UserRole.admin))) -> UserOut:
    return user


def require_operator(user: UserOut = Depends(
        require_roles(UserRole.admin, UserRole.operator))) -> UserOut:
    return user


def require_any_role(user: UserOut = Depends(get_current_user)) -> UserOut:
    return user
