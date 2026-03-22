from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Key Header ─────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ── Valid API Keys ─────────────────────────────────────────────
# In production these would be stored in PostgreSQL
# For now we use a simple dict
VALID_API_KEYS = {
    "test-key-123": "test_client",
    "hospital-key-456": "hospital_client",
    "insurance-key-789": "insurance_client",
}

# ── Verify API Key ─────────────────────────────────────────────
async def verify_api_key(api_key: str = Security(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No API key provided. Add X-API-Key header."
        )

    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key."
        )

    return VALID_API_KEYS[api_key]

# ── Get Client Name from Key ───────────────────────────────────
def get_client_name(api_key: str) -> str:
    return VALID_API_KEYS.get(api_key, "unknown")