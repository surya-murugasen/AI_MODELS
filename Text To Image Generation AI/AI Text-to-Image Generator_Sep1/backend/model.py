"""
model.py
--------
Thin wrapper around a Hugging Face Diffusers text-to-image pipeline.

This is the "AI" layer in the architecture:

    User -> Web UI -> FastAPI -> [ Diffusion Model ] -> Generated Image -> UI

Kept deliberately simple for learning purposes:
- One class, one job: turn a text prompt into a PIL Image.
- The model is loaded lazily (on first request) so the server starts instantly
  and only pays the (large) download/load cost when it's actually needed.
- Device selection (CUDA GPU / Apple Silicon MPS / CPU) is automatic.
"""

import os
import threading

import torch
from diffusers import AutoPipelineForText2Image

# The model to load from the Hugging Face Hub. Can be overridden with an
# environment variable so you can swap models without touching code, e.g.:
#   MODEL_ID="runwayml/stable-diffusion-v1-5" uvicorn backend.main:app
#
# Default is "stabilityai/sd-turbo": a small, distilled model that can
# generate a decent image in a single inference step, which makes it far
# more usable on a laptop CPU than full Stable Diffusion (which is slow
# without a GPU). Swap in a bigger model if you have a CUDA GPU.
DEFAULT_MODEL_ID = "stabilityai/sd-turbo"
MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)

# A small curated list the UI offers in its Model dropdown. Keeping this
# short is deliberate: each entry is a multi-gigabyte download the first
# time it's picked, and only one model is ever held in memory at once (see
# ImageGenerator.ensure_model below) to keep RAM/VRAM use bounded on a
# laptop — switching models means waiting for the new one to load, not
# instant. Add more entries here if you want more choices.
AVAILABLE_MODELS = [
    {"id": "stabilityai/sd-turbo", "label": "SD-Turbo (fast, 1 step)"},
    {"id": "custom-scratch-model", "label": "Custom Model (Trained from Scratch)"},
    {"id": "runwayml/stable-diffusion-v1-5", "label": "Stable Diffusion 1.5 (classic, ~25 steps)"},
    {"id": "stabilityai/stable-diffusion-2-1-base", "label": "Stable Diffusion 2.1 base (~25 steps)"},
]
# Whatever MODEL_ID resolves to (default or env override) should always be
# selectable too, even if it's not one of the curated entries above.
if not any(m["id"] == MODEL_ID for m in AVAILABLE_MODELS):
    AVAILABLE_MODELS.insert(0, {"id": MODEL_ID, "label": f"{MODEL_ID} (configured default)"})

# sd-turbo is designed for very few steps and guidance_scale=0.
# If you switch to a "normal" Stable Diffusion model, bump these up
# (e.g. steps=25, guidance_scale=7.5) for better quality.
DEFAULT_STEPS = int(os.environ.get("NUM_INFERENCE_STEPS", "1"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "0.0"))

# Width/height are validated in main.py and passed straight through to the
# pipeline; these are just what the UI pre-selects on first load.
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512


def _pick_device_and_dtype():
    """Pick the best available compute device and a matching tensor dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():  # Apple Silicon (M1/M2/M3...) GPU
        return "mps", torch.float32
    return "cpu", torch.float32


class ImageGenerator:
    """Loads a Diffusers pipeline or custom scratch model and reuses it for every request — until
    asked to switch models, at which point it unloads the current one and
    loads the new one. Only ever one model in memory at a time."""

    def __init__(self):
        self._pipe = None
        self._custom_model = None
        self._lock = threading.Lock()  # guards loading/switching
        self.device, self.dtype = _pick_device_and_dtype()
        self.current_model_id: str | None = None  # None until first load

    def _unload_current(self):
        self._pipe = None
        self._custom_model = None
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def ensure_model(self, model_id: str | None = None):
        """Make sure the requested model (or MODEL_ID if none given) is the
        one currently loaded, loading/switching if necessary. Safe to call
        on every request — it's a no-op once the right model is already
        loaded."""
        target = model_id or MODEL_ID
        with self._lock:
            if (self._pipe is not None or self._custom_model is not None) and self.current_model_id == target:
                return

            if self._pipe is not None or self._custom_model is not None:
                print(f"[model] Switching from '{self.current_model_id}' to '{target}' "
                      "— unloading the current model first to keep memory use bounded...")
                self._unload_current()

            if target == "custom-scratch-model":
                print("[model] Selected 'custom-scratch-model' (Custom AI Model built from scratch).")
                # Custom model pipeline hook: at Step 1 (Problem Definition), checkpoints are planned.
                self.current_model_id = target
                return

            print(f"[model] Loading '{target}' onto device='{self.device}' "
                  f"dtype={self.dtype} (this can take a while the first time, "
                  "as the model is downloaded)...")
            pipe = AutoPipelineForText2Image.from_pretrained(
                target,
                torch_dtype=self.dtype,
            )
            pipe = pipe.to(self.device)
            self._pipe = pipe
            self.current_model_id = target
            print("[model] Model loaded and ready.")

    def generate(self, prompt: str, steps: int | None = None,
                 guidance_scale: float | None = None, seed: int | None = None,
                 width: int | None = None, height: int | None = None,
                 model_id: str | None = None):
        """Run inference and return a single PIL.Image."""
        target = model_id or MODEL_ID
        self.ensure_model(target)

        if target == "custom-scratch-model":
            raise RuntimeError(
                "Custom Scratch Model is currently at 'Step 1: Problem Definition'. "
                "Training pipeline (Step 6) and model checkpoints will be generated in upcoming steps."
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            num_inference_steps=steps if steps is not None else DEFAULT_STEPS,
            guidance_scale=guidance_scale if guidance_scale is not None else DEFAULT_GUIDANCE_SCALE,
            generator=generator,
        )
        # width/height are optional — omitted, the pipeline uses its own
        # trained default resolution (512x512 for sd-turbo).
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height

        result = self._pipe(**kwargs)
        return result.images[0]


# A single shared instance used by the FastAPI app (backend/main.py).
generator = ImageGenerator()
