# AI MODELS Integration Changelog

## [2026-09-24] — [Phase 3B: AI MODELS Integration]

### Introduced
- Server-to-Server HMAC authentication middleware in Python (`backend/admin_integration.py`).
- `/api/admin/health` and `/api/admin/stats` endpoints specifically for the Central Admin integration.

### Modified
- `backend/main.py`: Included the new `admin_router`.

### Removed
- None

### Deprecated
- None

### Security Changes
- Environment variable `ADMIN_INTEGRATION_SECRET` is now evaluated by `verify_hmac` dependency.
- Used Python's standard `hmac`, `hashlib`, and `time` modules. No new dependencies were required.
- Implemented constant-time signature comparison (`hmac.compare_digest`) and a strict 5-minute request drift tolerance.

### Validation
- Existing frontend API endpoints (`/api/health`, `/api/generate`, `/api/models`) remain completely untouched to preserve AI inference features and UI layout functionality.
- Validated `GET /api/admin/health` with a correct HMAC signature from Central Admin.
- Validated `GET /api/admin/stats` aggregating lightweight model and device states without forcing heavy preloads.
- Invalid configurations securely return `401 Unauthorized`.
