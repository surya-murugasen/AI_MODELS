"""
prepare_dataset.py
-------------------
CLI entry point for the dataset-preparation pipeline. Run from the project
root, with the venv active:

    pip install -r requirements-training.txt   # one-time
    python -m backend.training.prepare_dataset

See backend/training/README.md for the full explanation and options.
"""

import argparse

from . import config
from .dataset_pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="Download, validate, and preprocess a Hugging Face image+caption "
        "dataset for future diffusion fine-tuning. Does not train anything."
    )
    parser.add_argument("--dataset", default=config.DATASET_ID, help="Hugging Face Hub dataset id")
    parser.add_argument("--config-name", default=config.DATASET_CONFIG, help="Dataset config/subset name, if the dataset has one")
    parser.add_argument("--split", default=config.DATASET_SPLIT, help="Dataset split to use")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=config.MAX_SAMPLES,
        help="Cap the number of raw examples pulled, for a quick first run (omit for the full split)",
    )
    parser.add_argument("--image-size", type=int, default=config.IMAGE_SIZE, help="Output image size in pixels (square)")
    args = parser.parse_args()

    run(
        dataset_id=args.dataset,
        dataset_config=args.config_name,
        split=args.split,
        max_samples=args.max_samples,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
