"""
preprocess.py
-------------
Pure image/caption preprocessing helpers used by dataset_pipeline.py.
No I/O here — just transforms of in-memory data, which keeps this easy to
unit test.
"""

import re

from PIL import Image


def normalize_caption(text: str) -> str:
    """Collapse repeated whitespace/newlines into single spaces and trim."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_image(image: Image.Image, size: int) -> Image.Image:
    """Convert to RGB and resize + center-crop to a square `size x size`.

    This is the standard preprocessing step for training/fine-tuning
    diffusion models, which expect fixed-size square inputs: resize the
    shorter side to `size` (preserving aspect ratio), then crop the excess
    off the longer side from the center.
    """
    image = image.convert("RGB")
    width, height = image.size

    scale = size / min(width, height)
    new_width, new_height = round(width * scale), round(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    left = (new_width - size) // 2
    top = (new_height - size) // 2
    return image.crop((left, top, left + size, top + size))
