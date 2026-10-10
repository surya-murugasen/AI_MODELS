import os
import hmac
import hashlib
import time
from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel

# Initialize router
admin_router = APIRouter(prefix="/api/admin")

# Startup time for uptime calculation
START_TIME = time.time()

# Parse .env manually if python-dotenv isn't guaranteed
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

load_env()

MAX_DRIFT_MS = 5 * 60 * 1000  # 5 minutes

async def verify_hmac(request: Request):
    secret = os.environ.get("ADMIN_INTEGRATION_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="Server misconfiguration: missing secret")
    
    signature = request.headers.get("x-admin-signature")
    timestamp = request.headers.get("x-admin-timestamp")
    nonce = request.headers.get("x-admin-nonce")
    
    if not signature or not timestamp or not nonce:
        raise HTTPException(status_code=401, detail="Missing authentication headers")
    
    try:
        req_time = int(timestamp)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid timestamp format")
    
    now = int(time.time() * 1000)
    if abs(now - req_time) > MAX_DRIFT_MS:
        raise HTTPException(status_code=401, detail="Request expired (timestamp drift)")
    
    # Canonical string: METHOD\nPATH\nTIMESTAMP\nNONCE
    method = request.method.upper()
    path = request.url.path
    canonical_string = f"{method}\n{path}\n{timestamp}\n{nonce}"
    
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        canonical_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    # Constant-time comparison
    if not hmac.compare_digest(signature, expected_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    return True

@admin_router.get("/health", dependencies=[Depends(verify_hmac)])
def admin_health():
    """
    Secure health endpoint for Central Admin integration.
    """
    return {
        "status": "healthy",
        "service": "AI MODELS",
        "version": "1.0.0",
        "timestamp": int(time.time() * 1000),
        "uptimeSeconds": round(time.time() - START_TIME, 2)
    }

@admin_router.get("/stats", dependencies=[Depends(verify_hmac)])
def admin_stats():
    """
    Secure stats endpoint returning safe application aggregate metrics.
    """
    # Defer importing generator to avoid circular dependencies if any
    try:
        from .model import generator
        model_id = generator.current_model_id or "Not Loaded"
        device = getattr(generator, "device", "unknown")
    except ImportError:
        model_id = "unknown"
        device = "unknown"

    return {
        "service": "AI MODELS",
        "timestamp": int(time.time() * 1000),
        "metrics": {
            "current_model": model_id,
            "device": device,
            "uptimeSeconds": round(time.time() - START_TIME, 2)
        }
    }
