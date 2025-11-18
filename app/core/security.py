import os
import time
import hashlib
import hmac
import base64
import json
from typing import Optional

SECRET_KEY = os.getenv("SECRET_KEY", "change-me")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    pad = '=' * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def create_access_token(sub: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {"sub": sub, "iat": now, "exp": now + ACCESS_TOKEN_EXPIRE_MINUTES * 60}
    h = _b64url(json.dumps(header).encode())
    p = _b64url(json.dumps(payload).encode())
    sig = hmac.new(SECRET_KEY.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url(sig)}"


def verify_token(token: str) -> Optional[dict]:
    try:
        h, p, s = token.split(".")
        sig = hmac.new(SECRET_KEY.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64url(sig), s):
            return None
        payload = json.loads(_b64url_decode(p))
        if int(time.time()) > payload.get("exp", 0):
            return None
        return payload
    except Exception:
        return None
