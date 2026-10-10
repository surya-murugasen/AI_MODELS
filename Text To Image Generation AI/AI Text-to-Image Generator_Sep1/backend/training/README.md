# Training data pipeline (separate, optional)

This folder prepares a fine-tuning-ready image+caption dataset. It is
completely separate from the live SD-Turbo generation app: nothing in
`backend/main.py` or `backend/model.py` imports anything from here, and
nothing here imports from them. Installing its dependencies or running it
has **no effect** on the running app — this is offline tooling you run
yourself from the command line, whenever you want.

**No training happens yet.** This stage only downloads, validates,
preprocesses, and stores a dataset. `finetune.py` is a documented stub for
the next stage, not a working trainer.

## What it does

1. **Load** — downloads an image+caption dataset from the Hugging Face
   Hub using the `datasets` library (default:
   [`sayakpaul/naruto-blip-captions`](https://huggingface.co/datasets/sayakpaul/naruto-blip-captions),
   a small (~1200 image) captioned dataset used in Hugging Face's own
   diffusion fine-tuning examples). **Dataset access on the Hub can change
   at any time** — a dataset can become gated, renamed, or removed after
   this was written. If loading fails, see "If a dataset stops working"
   below.
2. **Validate** — every example is checked before it's kept:

   | Rejection reason | Meaning |
   |---|---|
   | `missing_image` | The image field was empty |
   | `unreadable_image` | The image couldn't be decoded |
   | `image_too_small` | Smaller than 64px on its shortest side (usually a broken/placeholder image) |
   | `missing_caption` | The caption field was empty |
   | `caption_too_short` | Fewer than 3 characters after trimming |
   | `caption_too_long` | More than 500 characters |

3. **Preprocess** — kept images are converted to RGB and resized +
   center-cropped to a fixed square size (512×512 by default, matching
   Stable Diffusion's native training resolution). Captions are trimmed
   and have repeated whitespace collapsed.
4. **Store** — written to `backend/training/data/`:

   ```
   backend/training/data/
   ├── images/
   │   ├── 000000.jpg
   │   ├── 000001.jpg
   │   └── ...
   ├── metadata.jsonl       # {"file_name": "images/000000.jpg", "text": "a cartoon...", ...} per line
   └── dataset_info.json    # summary: counts kept/rejected, columns used, timestamp
   ```

   This layout is deliberately the Hugging Face `datasets` library's
   ["imagefolder" format](https://huggingface.co/docs/datasets/image_dataset#image-captioning) —
   `dataset.py` in this folder reads it straight back with
   `load_dataset("imagefolder", ...)`, ready to hand to a training loop
   later.

## Running it

From the project root, with the main app's virtual environment active:

```bash
pip install -r requirements-training.txt   # one-time — adds the `datasets` library
python -m backend.training.prepare_dataset
```

Useful options:

```bash
# Try it fast first, on just 50 examples:
python -m backend.training.prepare_dataset --max-samples 50

# Use a different Hugging Face dataset:
python -m backend.training.prepare_dataset --dataset your-org/your-dataset

# Change the output image size:
python -m backend.training.prepare_dataset --image-size 768
```

Or configure via environment variables instead of flags (see
`config.py` for the full list): `TRAINING_DATASET_ID`,
`TRAINING_DATASET_CONFIG`, `TRAINING_DATASET_SPLIT`,
`TRAINING_MAX_SAMPLES`, `TRAINING_IMAGE_SIZE`.

## Using a different dataset

Any Hugging Face Hub dataset with an image column and a text/caption
column should work automatically — the pipeline looks for common column
names (`image`/`img`/`photo` and `text`/`caption`/`captions`/`prompt`/
`alt_text`). If your dataset uses a different name, add it to the
`*_COLUMN_CANDIDATES` lists in `config.py`.

## If a dataset stops working

The Hub is not static — a dataset that's public today can become gated,
be renamed, or disappear later, and the reverse also happens. If you see
`DatasetNotFoundError` (including "is a gated dataset"), the pipeline
prints suggested next steps and the same two options apply:

1. **Point at a different dataset.** A few small, commonly-used
   image+caption datasets worth trying (verify current access status on
   huggingface.co, since this can change):

   ```bash
   python -m backend.training.prepare_dataset --dataset lambdalabs/pokemon-blip-captions
   python -m backend.training.prepare_dataset --dataset diffusers/pokemon-gpt4-captions
   python -m backend.training.prepare_dataset --dataset reach-vb/pokemon-blip-captions
   ```

   Or search [huggingface.co/datasets](https://huggingface.co/datasets?task_categories=task_categories:text-to-image)
   for any dataset tagged `text-to-image`.

2. **Authenticate, if you specifically want a gated dataset you've
   accepted the terms for.** Run `huggingface-cli login` once (it'll ask
   for an access token from your Hugging Face account settings), or set
   the `HF_TOKEN` environment variable before running
   `prepare_dataset.py` — the `datasets` library picks either up
   automatically.

## Files in this folder

| File | Purpose |
|---|---|
| `config.py` | All settings (dataset id, thresholds, output paths), overridable via env vars |
| `validation.py` | Column auto-detection + per-example validation |
| `preprocess.py` | Pure image/caption transforms (no I/O — easy to unit test) |
| `dataset_pipeline.py` | Orchestrates load → validate → preprocess → write |
| `prepare_dataset.py` | CLI entry point (`python -m backend.training.prepare_dataset`) |
| `dataset.py` | Reads the prepared data back as a `datasets.Dataset`, for a future trainer |
| `finetune.py` | **Stub only.** Documents the intended fine-tuning approach; raises if called |
| `data/` | Output of `prepare_dataset.py` — not checked into git, safe to delete and regenerate |
| `checkpoints/` | Reserved for future fine-tuned model weights (nothing writes here yet) |

## What's next (not built yet)

`finetune.py`'s docstring lays out the intended approach: load this
dataset, load the base `stabilityai/sd-turbo` pipeline, fine-tune it with
LoRA (recommended over a full fine-tune — far less VRAM), and save
checkpoints to `checkpoints/`. None of that is implemented — this stage
only gets the data ready.
