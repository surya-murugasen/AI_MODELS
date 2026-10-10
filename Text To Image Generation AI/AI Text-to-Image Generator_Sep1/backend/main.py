"""
main.py
-------
FastAPI backend for the AI Text-to-Image Generator.

Architecture recap:

    User -> Web UI -> FastAPI -> Diffusion Model -> Generated Image -> UI

This file is the "FastAPI" box: it exposes one HTTP endpoint the frontend
calls, delegates the actual image generation to model.py, and serves the
static frontend files so the whole app runs from a single `uvicorn` command.
"""

import base64
import io
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .model import (
    AVAILABLE_MODELS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    MODEL_ID,
    generator,
)
from .admin_integration import admin_router

ALLOWED_MODEL_IDS = {m["id"] for m in AVAILABLE_MODELS}

# Diffusion U-Nets downsample by a factor of 8 internally, so width/height
# must be multiples of 8. We only allow a small, known-good set of sizes
# (matching the UI's Square/Landscape/Portrait picker) rather than any
# arbitrary value, to keep generation times predictable on a laptop CPU/GPU.
ALLOWED_DIMENSIONS = {512, 768}

app = FastAPI(
    title="AI Text-to-Image Generator",
    description="A minimal FastAPI + Hugging Face Diffusers demo.",
    version="1.0.0",
)

# Include Central Admin Integration Router
app.include_router(admin_router)

# Wide-open CORS is fine here: this is a local, single-user learning project
# with no auth and no sensitive data. Do NOT do this for anything real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt describing the image to generate")
    steps: int | None = Field(None, ge=1, le=50, description="Number of denoising steps (optional)")
    guidance_scale: float | None = Field(None, ge=0, le=20, description="How closely to follow the prompt (optional)")
    seed: int | None = Field(None, description="Random seed for reproducible results (optional)")
    width: int | None = Field(None, description="Image width in pixels (must be 512 or 768)")
    height: int | None = Field(None, description="Image height in pixels (must be 512 or 768)")
    model_id: str | None = Field(None, description="Hugging Face model id to generate with (must be one of GET /api/models)")


class GenerateResponse(BaseModel):
    image_base64: str  # PNG image, base64-encoded, ready for an <img src="data:image/png;base64,...">
    prompt: str
    seconds: float
    width: int
    height: int
    model_id: str


@app.post("/api/generate", response_model=GenerateResponse)
def generate_image(req: GenerateRequest):
    """Run the diffusion model on the given prompt and return the image."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    for dim_name, dim_value in (("width", req.width), ("height", req.height)):
        if dim_value is not None and dim_value not in ALLOWED_DIMENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"{dim_name} must be one of {sorted(ALLOWED_DIMENSIONS)}, got {dim_value}.",
            )

    if req.model_id is not None and req.model_id not in ALLOWED_MODEL_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"model_id must be one of {sorted(ALLOWED_MODEL_IDS)}, got '{req.model_id}'.",
        )

    start = time.time()
    try:
        image = generator.generate(
            prompt=prompt,
            steps=req.steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed,
            width=req.width,
            height=req.height,
            model_id=req.model_id,
        )
    except Exception as exc:  # noqa: BLE001 - surface model errors to the caller
        raise HTTPException(status_code=500, detail=f"Image generation failed: {exc}") from exc
    elapsed = time.time() - start

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return GenerateResponse(
        image_base64=encoded,
        prompt=prompt,
        seconds=round(elapsed, 2),
        width=image.width,
        height=image.height,
        model_id=generator.current_model_id or MODEL_ID,
    )


@app.get("/api/health")
def health():
    """Liveness check. Also reports the model/device/defaults the React UI
    displays in its stat tiles and Settings view, so the frontend never has
    to hardcode those values. `model_id` reflects whichever model is
    actually loaded right now (it can differ from the configured default
    after a request switches models via GET /api/models)."""
    return {
        "status": "ok",
        "device": generator.device,
        "model_id": generator.current_model_id or MODEL_ID,
        "default_steps": DEFAULT_STEPS,
        "default_guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "default_width": DEFAULT_WIDTH,
        "default_height": DEFAULT_HEIGHT,
    }


@app.get("/api/models")
def list_models():
    """The curated model list the UI's Model dropdown offers, plus which
    one is currently loaded. Only one model is held in memory at a time —
    picking a different one and generating unloads the current model and
    loads the new one, which can take a while (see model.py)."""
    return {
        "available": AVAILABLE_MODELS,
        "current": generator.current_model_id or MODEL_ID,
    }


# Serve the built React frontend (frontend/dist, produced by `npm run build`)
# as static files, mounted last so it doesn't shadow the /api routes above.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    print(
        f"[main] Warning: '{FRONTEND_DIST}' not found — the API will still work, "
        "but no UI will be served. Run 'npm install && npm run build' inside "
        "the frontend/ folder first (see README.md)."
    )
