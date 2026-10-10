"""
data_collection.py
-------------------
STEP 2: DATA COLLECTION

Produces the raw, paired (image, text prompt) dataset that later steps
(3: Cleaning & Preprocessing, 4: Splitting, ...) will consume. This step
only COLLECTS and STORES data in its rawest usable form -- no resizing
beyond generation-time, no filtering, no normalization, no train/val/test
split. That comes later.

WHAT DATA IS NEEDED
    Per Step 1 (backend/custom_model/config.py: ProblemSpecification), the
    model consumes 64x64 RGB images (range [-1, 1] after Step 3
    normalization) conditioned on a text prompt tokenized to at most
    `max_text_length` tokens against a `vocab_size`-sized vocabulary. So
    every collected sample must be:
      - a real, decodable RGB image at (or above) the target resolution
      - paired 1:1 with a short natural-language caption describing it
      - captions must exercise words the tiny scratch vocabulary can
        actually learn (recurring color/shape/background terms), rather
        than the long-tail free-text captions found in photo corpora.

WHERE IT COMES FROM
    Source: procedurally generated shapes (see config.DataCollectionConfig
    for the full rationale). In short -- real captioned photo datasets
    (COCO Captions, Flickr8k/30k, Conceptual Captions, LAION subsets) are
    the eventual production-scale source, but need multi-GB downloads,
    license review, and far more model capacity than a ~15-25M parameter,
    randomly-initialized, <=3h-laptop-training model can learn from. A
    template-composed synthetic dataset gives an exact, verifiable
    word<->pixel mapping, which is what a tiny from-scratch model needs to
    demonstrably learn text conditioning at all. See the module docstring
    of `config.DataCollectionConfig` for the full write-up.

IMAGE + CAPTION FORMAT
    Image : PNG, RGB, exactly `image_width` x `image_height` px (64x64 by
            default), one flat-colored shape rendered on a solid or
            near-solid background.
    Caption : a single lowercase English sentence composed from a template,
            e.g. "a glowing purple star on a dark background". Assembled
            from `{adjective} {color} {shape} {background}` vocabulary so
            every word recurs across many samples.

FOLDER STRUCTURE (see config.py path constants)
    backend/custom_model/data/raw/
      images/
        sample_00000.png
        sample_00001.png
        ...
      raw_metadata.jsonl      # one JSON object per line, one per image
      dataset_summary.json    # aggregate manifest for the whole collection run

METADATA (one JSON object per line in raw_metadata.jsonl)
    id            int    row index, matches the sample_XXXXX numbering
    image_file    str    filename relative to images/
    prompt        str    the paired caption
    shape/color/background  str  the generating attributes (kept for
                  traceability/debugging; Step 3 is free to drop them)
    width, height, channels  int  actual pixel dimensions of the PNG
    source        str    "synthetic_procedural"
    license       str    usage rights for this sample
    seed          int    per-sample RNG seed (exact reproducibility)
    created_at    str    ISO-8601 UTC collection timestamp

STORAGE
    Plain local filesystem under backend/custom_model/data/raw/. Images are
    individual PNG files (keeps the collector streaming-friendly and lets
    any single sample be inspected/deleted without touching the rest).
    Metadata is JSON Lines (append-friendly, greppable, no full-file parse
    needed to add more samples later) plus one summary JSON manifest for
    quick dataset-level stats. This directory is the read-only input to
    Step 3 -- nothing in this script cleans, filters, resizes-down,
    augments, or splits it.
"""

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.parent))

from backend.custom_model.config import (
    DataCollectionConfig,
    DEFAULT_DATA_COLLECTION_CONFIG,
    RAW_DATA_DIR,
    RAW_IMAGES_DIR,
    RAW_METADATA_PATH,
    DATASET_SUMMARY_PATH,
)

# Approximate RGB values for the color vocabulary. Kept separate from the
# config's `colors` word list: the words are what the model reads, these
# are only how the renderer paints them.
COLOR_RGB = {
    "red": (220, 40, 40), "blue": (50, 100, 220), "green": (40, 170, 90),
    "yellow": (230, 200, 40), "purple": (130, 60, 190), "orange": (230, 120, 30),
    "pink": (235, 110, 170), "cyan": (40, 190, 210), "magenta": (210, 40, 170),
    "turquoise": (40, 180, 170), "gold": (210, 170, 40), "silver": (190, 190, 195),
    "crimson": (180, 20, 50), "emerald": (30, 150, 90), "violet": (140, 80, 210),
    "indigo": (70, 50, 160),
}
BACKGROUND_RGB = {
    "a dark background": (18, 18, 20), "a deep sky": (15, 25, 55),
    "a white background": (245, 245, 245), "a soft gradient": (60, 55, 70),
    "a starry night": (10, 12, 30), "a pastel background": (235, 225, 235),
    "a black canvas": (8, 8, 8), "a foggy backdrop": (150, 150, 155),
}


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple, rgb: tuple) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = (x1 - x0) / 2

    if shape == "circle":
        draw.ellipse(box, fill=rgb)
    elif shape == "square":
        draw.rectangle(box, fill=rgb)
    elif shape == "triangle":
        draw.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=rgb)
    elif shape == "diamond":
        draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=rgb)
    elif shape == "cross":
        third = (x1 - x0) / 3
        draw.rectangle((x0 + third, y0, x1 - third, y1), fill=rgb)
        draw.rectangle((x0, y0 + third, x1, y1 - third), fill=rgb)
    elif shape == "hexagon":
        pts = []
        for i in range(6):
            import math
            angle = math.pi / 3 * i - math.pi / 2
            pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(pts, fill=rgb)
    elif shape == "star":
        import math
        pts = []
        for i in range(10):
            angle = math.pi / 5 * i - math.pi / 2
            radius = r if i % 2 == 0 else r * 0.45
            pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(pts, fill=rgb)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def _render_sample(cfg: DataCollectionConfig, shape: str, color: str, background: str, rng: random.Random) -> Image.Image:
    bg_rgb = BACKGROUND_RGB[background]
    fg_rgb = COLOR_RGB[color]

    img = Image.new("RGB", (cfg.image_width, cfg.image_height), color=bg_rgb)
    draw = ImageDraw.Draw(img)

    # Mild jitter so repeated (shape, color, background) combos don't
    # collapse into pixel-identical images.
    margin = int(min(cfg.image_width, cfg.image_height) * 0.18)
    scale = rng.uniform(0.85, 1.0)
    w, h = cfg.image_width, cfg.image_height
    box_size = (min(w, h) - 2 * margin) * scale
    max_offset = (min(w, h) - 2 * margin - box_size) / 2
    off_x = rng.uniform(-max_offset, max_offset) if max_offset > 0 else 0
    off_y = rng.uniform(-max_offset, max_offset) if max_offset > 0 else 0

    cx, cy = w / 2 + off_x, h / 2 + off_y
    x0, y0 = cx - box_size / 2, cy - box_size / 2
    x1, y1 = cx + box_size / 2, cy + box_size / 2

    _draw_shape(draw, shape, (x0, y0, x1, y1), fg_rgb)
    return img


def _make_prompt(cfg: DataCollectionConfig, shape: str, color: str, background: str, rng: random.Random) -> str:
    template = rng.choice(cfg.caption_templates)
    adjective = rng.choice(cfg.style_adjectives)
    return template.format(adjective=adjective, color=color, shape=shape, background=background)


def collect_dataset(cfg: DataCollectionConfig = DEFAULT_DATA_COLLECTION_CONFIG) -> dict:
    """Generates `cfg.num_samples` (image, prompt) pairs and writes them
    under RAW_DATA_DIR per the on-disk contract documented at the top of
    this module. Returns the summary dict that is also written to
    DATASET_SUMMARY_PATH. Re-running replaces any previous raw collection
    (this is the collector, not an incremental crawler)."""

    rng = random.Random(cfg.random_seed)

    # Fresh run: clear any prior raw collection so the directory always
    # reflects exactly one collection run / one config.
    if RAW_IMAGES_DIR.exists():
        for f in RAW_IMAGES_DIR.glob("*.png"):
            f.unlink()
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    shape_counts: dict = {s: 0 for s in cfg.shapes}
    color_counts: dict = {c: 0 for c in cfg.colors}

    with open(RAW_METADATA_PATH, "w", encoding="utf-8") as meta_f:
        for i in range(cfg.num_samples):
            sample_seed = cfg.random_seed + i
            sample_rng = random.Random(sample_seed)

            shape = sample_rng.choice(cfg.shapes)
            color = sample_rng.choice(cfg.colors)
            background = sample_rng.choice(cfg.backgrounds)

            image = _render_sample(cfg, shape, color, background, sample_rng)
            prompt = _make_prompt(cfg, shape, color, background, sample_rng)

            image_file = f"sample_{i:05d}.png"
            image.save(RAW_IMAGES_DIR / image_file, format="PNG")

            record = {
                "id": i,
                "image_file": image_file,
                "prompt": prompt,
                "shape": shape,
                "color": color,
                "background": background,
                "width": cfg.image_width,
                "height": cfg.image_height,
                "channels": cfg.image_channels,
                "source": cfg.source_type,
                "license": cfg.license,
                "seed": sample_seed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            meta_f.write(json.dumps(record) + "\n")

            shape_counts[shape] += 1
            color_counts[color] += 1

    summary = {
        "dataset_name": "custom_model_step2_synthetic_shapes",
        "source_type": cfg.source_type,
        "total_samples": cfg.num_samples,
        "image_size": [cfg.image_width, cfg.image_height],
        "channels": cfg.image_channels,
        "unique_shapes": cfg.shapes,
        "unique_colors": cfg.colors,
        "unique_backgrounds": cfg.backgrounds,
        "shape_distribution": shape_counts,
        "color_distribution": color_counts,
        "generation_seed": cfg.random_seed,
        "license": cfg.license,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "storage": {
            "images_dir": str(RAW_IMAGES_DIR.relative_to(RAW_DATA_DIR.parent.parent)),
            "metadata_file": str(RAW_METADATA_PATH.relative_to(RAW_DATA_DIR.parent.parent)),
        },
        "notes": "Step 2 raw collection only -- no cleaning, preprocessing, splitting, or training performed on this data yet.",
    }
    with open(DATASET_SUMMARY_PATH, "w", encoding="utf-8") as summary_f:
        json.dump(summary, summary_f, indent=2)

    return summary


def run_step2_data_collection() -> None:
    """CLI entry point: runs collection and prints a report, matching the
    style of Step 1's problem_definition.py."""
    print("=" * 70)
    print("  AI TEXT-TO-IMAGE GENERATOR: CUSTOM MODEL FROM SCRATCH")
    print("  STEP 2: DATA COLLECTION REPORT")
    print("=" * 70)

    cfg = DEFAULT_DATA_COLLECTION_CONFIG
    print("\n1. DATA SOURCE:")
    print(f"   • Source type: {cfg.source_type}")
    print(f"   • License: {cfg.license}")
    print(f"   • Random seed (reproducibility): {cfg.random_seed}")

    print("\n2. COLLECTING SAMPLES...")
    summary = collect_dataset(cfg)
    print(f"   • Generated {summary['total_samples']} paired (image, prompt) samples")
    print(f"   • Image size: {summary['image_size'][0]}x{summary['image_size'][1]}x{summary['channels']}")

    print("\n3. STORAGE LAYOUT:")
    print(f"   • Images:   {RAW_IMAGES_DIR}")
    print(f"   • Metadata: {RAW_METADATA_PATH}")
    print(f"   • Summary:  {DATASET_SUMMARY_PATH}")

    print("\n4. VOCABULARY COVERAGE:")
    print(f"   • Shapes ({len(cfg.shapes)}): {', '.join(cfg.shapes)}")
    print(f"   • Colors ({len(cfg.colors)}): {', '.join(cfg.colors)}")
    print(f"   • Backgrounds ({len(cfg.backgrounds)}): {', '.join(cfg.backgrounds)}")

    print("\n5. PER-SHAPE SAMPLE COUNT:")
    for shape, count in summary["shape_distribution"].items():
        print(f"   • {shape:<10} {count}")

    print("\n" + "=" * 70)
    print("Step 2 (Data Collection) complete. Data is RAW -- not cleaned,")
    print("preprocessed, split, or used for training yet (Steps 3-4).")
    print("=" * 70)


if __name__ == "__main__":
    run_step2_data_collection()
