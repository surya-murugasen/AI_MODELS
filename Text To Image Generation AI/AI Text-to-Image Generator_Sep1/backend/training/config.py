"""
config.py
---------
Configuration for the dataset-preparation pipeline.

Everything here is overridable via environment variables, following the
same pattern as `backend/model.py` — but this file is intentionally NOT
imported by, and does not import, `backend/model.py`. The two are kept
completely separate on purpose.
"""

import os
from pathlib import Path

# --- Source dataset ---------------------------------------------------------
# Any Hugging Face Hub dataset with an image column and a caption/text
# column works here. The default, "sayakpaul/naruto-blip-captions", is a
# small (~1200 image), publicly-listed captioned dataset commonly used in
# Hugging Face's own diffusion fine-tuning examples.
#
# NOTE: dataset access on the Hub can change (a dataset can become gated,
# renamed, or removed at any time) — if you hit a "gated dataset" or "not
# found" error, see backend/training/README.md's "If a dataset stops
# working" section for alternatives and how to authenticate.
DATASET_ID = os.environ.get("TRAINING_DATASET_ID", "sayakpaul/naruto-blip-captions")
DATASET_CONFIG = os.environ.get("TRAINING_DATASET_CONFIG") or None
DATASET_SPLIT = os.environ.get("TRAINING_DATASET_SPLIT", "train")

# Datasets on the Hub name their columns differently. We auto-detect by
# checking for these candidates, in order, among the dataset's columns.
IMAGE_COLUMN_CANDIDATES = ["image", "img", "photo"]
CAPTION_COLUMN_CANDIDATES = ["text", "caption", "captions", "prompt", "alt_text"]

# Cap how many raw examples to pull — useful for a fast first run on a
# laptop. None (the default) means "use the whole split."
MAX_SAMPLES = int(os.environ["TRAINING_MAX_SAMPLES"]) if os.environ.get("TRAINING_MAX_SAMPLES") else None

# --- Validation thresholds ---------------------------------------------------
MIN_IMAGE_SIDE = 64        # px — smaller is almost always a broken/placeholder image
MIN_CAPTION_CHARS = 3
MAX_CAPTION_CHARS = 500

# --- Preprocessing ------------------------------------------------------------
# 512 matches Stable Diffusion / SD-Turbo's native training resolution, so a
# model fine-tuned on this data would stay size-compatible with the live
# app. This is just a shared convention with model.py's default, not a
# code dependency on it.
IMAGE_SIZE = int(os.environ.get("TRAINING_IMAGE_SIZE", "512"))
IMAGE_FORMAT = "JPEG"     # smaller on disk than PNG across thousands of images
JPEG_QUALITY = 95

# --- Output locations ---------------------------------------------------------
TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_PATH = DATA_DIR / "metadata.jsonl"
STATS_PATH = DATA_DIR / "dataset_info.json"

# --- Future fine-tuning (not implemented yet — see finetune.py) --------------
CHECKPOINTS_DIR = TRAINING_DIR / "checkpoints"
# Same default model id as the live app (backend/model.py), so a future
# fine-tune starts from the same base model this app already generates
# with. Duplicated as a plain string on purpose — see module docstring.
BASE_MODEL_ID = os.environ.get("MODEL_ID", "stabilityai/sd-turbo")
