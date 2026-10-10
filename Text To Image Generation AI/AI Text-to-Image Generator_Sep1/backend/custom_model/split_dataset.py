"""
split_dataset.py
-----------------
STEP 4: DATASET SPLITTING

Partitions the valid (image, caption) pairs produced by Step 3
(data/processed/processed_metadata.jsonl) into Train / Validation / Test
subsets, and writes each subset out as its own metadata file. This step
only SPLITS existing processed data -- it does not resize, re-normalize,
re-tokenize, augment, build a model, train, or evaluate anything. Those
come later (Steps 5-8).

WHY THREE SPLITS
    Train (80%)      -- the only rows the model's weights are ever fit on.
    Validation (10%) -- held out from training; used later to monitor
                         overfitting and choose hyperparameters/checkpoints
                         (Steps 6-7). Never used to update weights directly.
    Test (10%)       -- touched exactly once, at the very end (Step 8), to
                         report a final, unbiased performance estimate on
                         data that influenced the model in no way at all --
                         not during training, not during tuning.
    Keeping these three separate is what makes "the model works" a
    trustworthy claim instead of the model having just memorized what it
    was graded on.

HOW LEAKAGE IS PREVENTED
    Step 2/3 verification found that a handful of procedurally generated
    samples render to pixel-identical images despite different ids/
    captions (same shape+color+background combo, random jitter rounding
    to the same pixels at 64x64). If two samples with identical image
    content ended up in different splits, the model would effectively be
    "graded" on an image it already trained on -- silently inflating
    validation/test performance without the model having generalized at
    all. To prevent this, every sample is first grouped with any other
    sample whose *processed* image is byte-for-byte identical; splitting
    then assigns whole groups (never a subset of a group) to exactly one
    split. Combined with a fixed random seed for the group shuffle, this
    makes the split both leak-free and reproducible.

FOLDER STRUCTURE (see config.py path constants)
    backend/custom_model/data/splits/
      train_metadata.jsonl   # ~80% of samples
      val_metadata.jsonl     # ~10% of samples
      test_metadata.jsonl    # ~10% of samples
      split_summary.json     # counts, percentages, seed, leakage-group info

    Each split file uses the exact same row schema as Step 3's
    processed_metadata.jsonl (id, image_file, tensor_file, prompt,
    token_ids, token_count, shape, color, background, width, height,
    channels, pixel_range) -- rows are a straight subset, not a
    transformation, so an (image, caption) pair is never split apart or
    modified by this step. Only the metadata is split; the underlying
    processed_00XXX.png / tensor_00XXX.pt files stay in their single
    shared location (data/processed/) and are referenced by filename from
    whichever split's metadata lists them, exactly like Step 3 referenced
    Step 2's raw images without duplicating them.

REPRODUCIBILITY
    Re-running this script with the same config.py (same random_seed and
    ratios) against the same Step 3 output always produces byte-identical
    split files -- the only randomness (which group goes to which split)
    is seeded.
"""

import hashlib
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.custom_model.config import (
    DEFAULT_SPLITTING_CONFIG,
    PROCESSED_IMAGES_DIR,
    PROCESSED_METADATA_PATH,
    SPLITS_DIR,
    SPLIT_SUMMARY_PATH,
    TEST_METADATA_PATH,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    DataSplittingConfig,
)

SPLIT_NAMES = ["train", "val", "test"]


class DatasetSplitter:
    """Executes Step 4: groups Step 3's processed samples by exact image
    content (to prevent leakage), then deterministically splits those
    groups into train/val/test according to the configured ratios."""

    def __init__(self, config: Optional[DataSplittingConfig] = None):
        self.config = config or DEFAULT_SPLITTING_CONFIG

    @staticmethod
    def _load_processed_metadata() -> List[Dict[str, Any]]:
        if not PROCESSED_METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Processed dataset metadata not found at '{PROCESSED_METADATA_PATH}'. "
                "Ensure Step 3: Data Cleaning & Preprocessing has been run first."
            )
        records = []
        with open(PROCESSED_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _content_hash(image_path: Path) -> str:
        """Exact hash of an image's decoded RGB pixel bytes -- two samples
        hash identically if and only if they render to the same picture."""
        with Image.open(image_path) as img:
            rgb = img.convert("RGB") if img.mode != "RGB" else img
            return hashlib.sha256(rgb.tobytes()).hexdigest()

    def _group_by_image_content(self, records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rec in records:
            if self.config.prevent_image_content_leakage:
                h = self._content_hash(PROCESSED_IMAGES_DIR / rec["image_file"])
            else:
                # Leakage prevention disabled: every sample is its own group.
                h = f"__no_grouping__{rec['id']}"
            groups[h].append(rec)
        return list(groups.values())

    def _assign_groups(self, groups: List[List[Dict[str, Any]]], total_n: int) -> Dict[str, List[Dict[str, Any]]]:
        rng = random.Random(self.config.random_seed)
        shuffled = groups[:]
        rng.shuffle(shuffled)

        targets = {
            "train": self.config.train_ratio * total_n,
            "val": self.config.val_ratio * total_n,
            "test": self.config.test_ratio * total_n,
        }
        counts = {name: 0 for name in SPLIT_NAMES}
        assigned: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}

        for group in shuffled:
            # Assign the whole group to whichever split is currently
            # furthest below its target share -- keeps the realized split
            # as close as possible to the configured ratios despite a few
            # multi-member (leakage) groups that can't be split up.
            deficits = [(targets[name] - counts[name], SPLIT_NAMES.index(name), name) for name in SPLIT_NAMES]
            best_split = max(deficits)[2]
            assigned[best_split].extend(group)
            counts[best_split] += len(group)

        for name in SPLIT_NAMES:
            assigned[name].sort(key=lambda r: r["id"])
        return assigned

    def run(self) -> Dict[str, Any]:
        print("=" * 70)
        print("  AI TEXT-TO-IMAGE GENERATOR: CUSTOM MODEL FROM SCRATCH")
        print("  STEP 4: DATASET SPLITTING REPORT")
        print("=" * 70)

        cfg = self.config
        print("\n1. SPLIT CONFIGURATION:")
        print(f"   - Target ratios: train={cfg.train_ratio:.0%} val={cfg.val_ratio:.0%} test={cfg.test_ratio:.0%}")
        print(f"   - Random seed (reproducibility): {cfg.random_seed}")
        print(f"   - Leakage prevention (group by identical image content): {cfg.prevent_image_content_leakage}")

        print(f"\n2. LOADING Step 3 output from: {PROCESSED_METADATA_PATH}")
        records = self._load_processed_metadata()
        total_n = len(records)
        print(f"   - Valid samples found: {total_n}")

        print("\n3. GROUPING samples by exact processed-image content...")
        groups = self._group_by_image_content(records)
        multi_groups = [g for g in groups if len(g) > 1]
        print(f"   - Unique image-content groups: {len(groups)}")
        print(f"   - Groups with more than one sample (kept together to avoid leakage): {len(multi_groups)}")

        print("\n4. ASSIGNING groups to splits...")
        assigned = self._assign_groups(groups, total_n)

        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        path_by_name = {"train": TRAIN_METADATA_PATH, "val": VAL_METADATA_PATH, "test": TEST_METADATA_PATH}

        print("\n5. WRITING split metadata files...")
        for name in SPLIT_NAMES:
            rows = assigned[name]
            with open(path_by_name[name], "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))
            print(f"   - {name:<5} {len(rows):>4} samples -> {path_by_name[name]}")

        # --- Self-check: no leakage, no loss, no duplication ---
        all_ids = [r["id"] for name in SPLIT_NAMES for r in assigned[name]]
        assert len(all_ids) == total_n, "Split sample count does not match Step 3 input -- samples were lost or duplicated."
        assert len(set(all_ids)) == len(all_ids), "The same sample id appears in more than one split."
        train_ids, val_ids, test_ids = (set(r["id"] for r in assigned[n]) for n in SPLIT_NAMES)
        assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids), \
            "Overlap detected between splits."

        summary = {
            "step": "Step 4: Dataset Splitting",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source": {
                "processed_metadata": str(PROCESSED_METADATA_PATH),
                "total_valid_samples": total_n,
            },
            "config": {
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
                "test_ratio": cfg.test_ratio,
                "random_seed": cfg.random_seed,
                "prevent_image_content_leakage": cfg.prevent_image_content_leakage,
            },
            "splits": {
                name: {
                    "count": len(assigned[name]),
                    "percent": round(100 * len(assigned[name]) / total_n, 2),
                    "target_percent": round(
                        {"train": cfg.train_ratio, "val": cfg.val_ratio, "test": cfg.test_ratio}[name] * 100, 2
                    ),
                    "metadata_file": str(path_by_name[name]),
                }
                for name in SPLIT_NAMES
            },
            "leakage_prevention": {
                "method": "group-by-exact-processed-image-pixel-hash",
                "total_groups": len(groups),
                "multi_sample_groups": len(multi_groups),
                "multi_sample_group_sizes": sorted((len(g) for g in multi_groups), reverse=True),
                "cross_split_overlap_ids": [],
            },
            "checks": {
                "sample_count_preserved": len(all_ids) == total_n,
                "no_duplicate_ids_across_splits": len(set(all_ids)) == len(all_ids),
                "no_split_overlap": not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids),
            },
            "notes": (
                "Step 4 split only -- rows are an unmodified subset of Step 3's "
                "processed_metadata.jsonl. No model has been built, trained, "
                "evaluated, or optimized on this data yet (Steps 5-8)."
            ),
        }
        with open(SPLIT_SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n6. SUMMARY REPORT saved to: {SPLIT_SUMMARY_PATH}")
        print("\n" + "=" * 70)
        print("  STEP 4 SUMMARY:")
        for name in SPLIT_NAMES:
            s = summary["splits"][name]
            print(f"  - {name.capitalize():<10} {s['count']:>4} samples  ({s['percent']}%, target {s['target_percent']}%)")
        print(f"  - Leakage-prevention groups > 1 sample: {len(multi_groups)}")
        print(f"  - No sample id appears in more than one split: {summary['checks']['no_split_overlap']}")
        print("=" * 70)
        print("Step 4 (Dataset Splitting) complete. No model built, trained,")
        print("evaluated, or optimized yet (Steps 5-8).")
        print("=" * 70)

        return summary


def run_step4_dataset_splitting() -> None:
    """CLI entry point, matching the style of Steps 1-3."""
    splitter = DatasetSplitter()
    splitter.run()


if __name__ == "__main__":
    run_step4_dataset_splitting()
