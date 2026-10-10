"""
dataset_pipeline.py
--------------------
Loads a Hugging Face image+caption dataset, validates each example,
preprocesses images/captions, and writes the result to
backend/training/data/ in a layout the Hugging Face `datasets` library can
read straight back (an "imagefolder" dataset with a metadata.jsonl
manifest) — ready to feed into a future diffusion fine-tuning script (see
finetune.py).

This module is intentionally standalone: it does not import from, and is
not imported by, backend/model.py or backend/main.py. Running it (or not)
has zero effect on the live SD-Turbo generation app.

Usage (from the project root, with the venv active):

    pip install -r requirements-training.txt   # one-time, adds `datasets`
    python -m backend.training.prepare_dataset
"""

import json
import time
from collections import Counter

from tqdm import tqdm

from . import config
from .preprocess import normalize_caption, preprocess_image
from .validation import find_column, validate_example


def run(dataset_id=None, dataset_config=None, split=None, max_samples=None, image_size=None):
    """Run the full pipeline once and return a stats dict. Also writes:
      - backend/training/data/images/*.jpg      (preprocessed images)
      - backend/training/data/metadata.jsonl    (file_name + caption per image)
      - backend/training/data/dataset_info.json (summary stats, for reference)
    """
    dataset_id = dataset_id or config.DATASET_ID
    dataset_config = dataset_config if dataset_config is not None else config.DATASET_CONFIG
    split = split or config.DATASET_SPLIT
    max_samples = max_samples if max_samples is not None else config.MAX_SAMPLES
    image_size = image_size or config.IMAGE_SIZE

    # Imported lazily so the main app's `requirements.txt` never needs to
    # depend on the `datasets` library — only this optional pipeline does
    # (see requirements-training.txt).
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    print(f"[dataset_pipeline] Loading '{dataset_id}' (split={split}) from the Hugging Face Hub...")
    try:
        raw = load_dataset(dataset_id, name=dataset_config, split=split)
    except DatasetNotFoundError as exc:
        # Dataset access on the Hub can change at any time (a dataset can
        # become gated, renamed, or removed) — fail with something
        # actionable instead of a bare traceback.
        raise RuntimeError(
            f"Could not load dataset '{dataset_id}' from the Hugging Face Hub: {exc}\n\n"
            "This usually means the dataset is gated (requires accepting terms and "
            "authenticating) or no longer exists under that name. Try:\n"
            "  1. A different dataset:  python -m backend.training.prepare_dataset --dataset <owner>/<name>\n"
            "     (see backend/training/README.md's \"If a dataset stops working\" section for suggestions)\n"
            "  2. If you specifically want this gated dataset and have accepted its terms on\n"
            "     huggingface.co, authenticate first: `huggingface-cli login`, or set the\n"
            "     HF_TOKEN environment variable, then re-run this command.\n"
        ) from exc

    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    image_col = find_column(raw.column_names, config.IMAGE_COLUMN_CANDIDATES, "image")
    caption_col = find_column(raw.column_names, config.CAPTION_COLUMN_CANDIDATES, "caption")
    print(f"[dataset_pipeline] Using image column '{image_col}' and caption column '{caption_col}'.")

    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    total_seen = 0
    kept = 0
    rejected = Counter()
    records = []

    for i, raw_example in enumerate(tqdm(raw, desc="Validating + preprocessing")):
        total_seen += 1
        ok, reason, image, caption = validate_example(raw_example, image_col, caption_col)
        if not ok:
            rejected[reason] += 1
            continue

        original_width, original_height = image.size
        processed = preprocess_image(image, image_size)
        caption = normalize_caption(caption)

        example_id = f"{i:06d}"
        file_name = f"images/{example_id}.jpg"
        processed.save(config.DATA_DIR / file_name, format=config.IMAGE_FORMAT, quality=config.JPEG_QUALITY)

        records.append(
            {
                "file_name": file_name,
                "text": caption,
                "example_id": example_id,
                "source_index": i,
                "original_width": original_width,
                "original_height": original_height,
                "width": processed.width,
                "height": processed.height,
            }
        )
        kept += 1

    with open(config.METADATA_PATH, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    stats = {
        "source_dataset": dataset_id,
        "source_config": dataset_config,
        "source_split": split,
        "image_column": image_col,
        "caption_column": caption_col,
        "image_size": image_size,
        "total_seen": total_seen,
        "kept": kept,
        "rejected_total": sum(rejected.values()),
        "rejected_by_reason": dict(rejected),
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(config.STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[dataset_pipeline] Done. Kept {kept}/{total_seen}, rejected {sum(rejected.values())}.")
    if rejected:
        print(f"[dataset_pipeline] Rejection breakdown: {dict(rejected)}")
    print(f"[dataset_pipeline] Images:   {config.IMAGES_DIR}")
    print(f"[dataset_pipeline] Metadata: {config.METADATA_PATH}")
    print(f"[dataset_pipeline] Stats:    {config.STATS_PATH}")
    return stats
