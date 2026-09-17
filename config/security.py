"""
VoltVision AI — Security Utilities
Uses bcrypt directly — compatible with Python 3.13+
passlib is NOT used here because it depends on the deprecated
crypt module removed in Python 3.13.
"""
import hashlib
import secrets
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from config.settings import settings


# ── Password hashing (bcrypt directly, no passlib) ────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plaintext password using bcrypt. Returns utf-8 string."""
    pwd_bytes = plain.encode("utf-8")
    salt      = bcrypt.gensalt(rounds=12)
    hashed    = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── JWT ───────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_access_token(user_id: int, email: str, role: str) -> str:
    payload = {
        "sub":   str(user_id),
        "email": email,
        "role":  role,
        "type":  "access",
        "exp":   _now() + timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat":   _now(),
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token():
    """Returns (raw_token, token_hash, expires_at)."""
    raw     = secrets.token_urlsafe(64)
    h       = hashlib.sha256(raw.encode()).hexdigest()
    expires = _now() + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    return raw, h, expires


def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None
