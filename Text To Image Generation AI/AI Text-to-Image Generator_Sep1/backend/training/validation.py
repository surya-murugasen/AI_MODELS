"""
validation.py
--------------
Column auto-detection and per-example validation for the dataset pipeline.
"""

from PIL import Image, UnidentifiedImageError

from . import config


def find_column(column_names, candidates, kind):
    """Return the first candidate name present in `column_names`, or raise
    a clear error listing what was actually available."""
    for name in candidates:
        if name in column_names:
            return name
    raise ValueError(
        f"Could not find a {kind} column in this dataset. Looked for "
        f"{candidates} among the dataset's columns: {list(column_names)}. "
        "Add the right column name to the *_COLUMN_CANDIDATES list in "
        "backend/training/config.py."
    )


def validate_example(raw_example: dict, image_col: str, caption_col: str):
    """Validate one raw example from the source dataset.

    Returns (is_valid, reject_reason, image, caption):
      - On success: reject_reason is None, image is a loaded PIL.Image,
        caption is a non-empty str.
      - On failure: reject_reason is a short machine-readable string
        (see the README's "What gets rejected" table), image/caption None.
    """
    image = raw_example.get(image_col)
    if image is None:
        return False, "missing_image", None, None

    try:
        # Most Hub image datasets hand back a decoded PIL.Image already;
        # a few hand back a path or raw bytes, so we open it ourselves too.
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        return False, "unreadable_image", None, None

    width, height = image.size
    if min(width, height) < config.MIN_IMAGE_SIDE:
        return False, "image_too_small", None, None

    caption = raw_example.get(caption_col)
    if caption is None:
        return False, "missing_caption", None, None

    caption = str(caption).strip()
    if len(caption) < config.MIN_CAPTION_CHARS:
        return False, "caption_too_short", None, None
    if len(caption) > config.MAX_CAPTION_CHARS:
        return False, "caption_too_long", None, None

    return True, None, image, caption
