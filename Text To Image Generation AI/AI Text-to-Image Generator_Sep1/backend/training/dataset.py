"""
dataset.py
----------
Reads the prepared dataset back as a Hugging Face `datasets.Dataset` — the
form a training loop would actually consume. This is groundwork for
fine-tuning (see finetune.py); it is not used by the live generation app.
"""

from . import config


def load_prepared_dataset():
    """Load backend/training/data/ (images/ + metadata.jsonl) back as a
    Hugging Face `Dataset` with 'image' and 'text' columns, using the
    `datasets` library's built-in "imagefolder" loader — the same format
    prepare_dataset.py wrote."""
    if not config.METADATA_PATH.exists():
        raise FileNotFoundError(
            f"No prepared dataset found at {config.METADATA_PATH}. "
            "Run `python -m backend.training.prepare_dataset` first."
        )

    from datasets import load_dataset

    return load_dataset("imagefolder", data_dir=str(config.DATA_DIR), split="train")
