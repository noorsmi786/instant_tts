"""
Authentication for the Vapi TTS endpoint.

Vapi sends the secret via the header configured in the Custom Credentials
(or legacy 'secret' field) as: X-Vapi-Secret: <your-secret>

The secret is stored only in .env and read at startup.
It is never logged, never returned in responses.
"""

import os
import secrets
from fastapi import Header, HTTPException, status


def _load_secret() -> str:
    """Load the Vapi secret from environment at import time."""
    secret = os.environ.get("VAPI_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "VAPI_SECRET is not set in .env. "
            "Please copy .env.example to .env and set a strong secret."
        )
    return secret


# Loaded once at module import (after dotenv is loaded in main.py)
_VAPI_SECRET: str = ""


def init_secret() -> None:
    """Call this from main.py after loading dotenv."""
    global _VAPI_SECRET
    _VAPI_SECRET = _load_secret()


async def require_vapi_auth(
    x_vapi_secret: str = Header(
        default=None,
        alias="x-vapi-secret",
        description="Secret token configured in Vapi Custom Credentials",
    )
) -> None:
    """
    FastAPI dependency: validates the X-Vapi-Secret header using
    a constant-time comparison to prevent timing attacks.
    Raises HTTP 401 on missing/wrong secret.
    """
    if not _VAPI_SECRET:
        # Should never happen if init_secret() was called, but guard anyway
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not configured",
        )

    if x_vapi_secret is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Vapi-Secret header",
        )

    if not secrets.compare_digest(x_vapi_secret.encode(), _VAPI_SECRET.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-Vapi-Secret",
        )
