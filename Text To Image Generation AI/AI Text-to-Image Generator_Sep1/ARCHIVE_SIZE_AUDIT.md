# Archive Size Audit

What `frontend/` and `backend/` actually contain, and what a zip of each would weigh if made separately.

- **Project**: AI Text-to-Image Generator
- **Measured**: 2026-08-28, via `du -sh`
- **Free disk at time of audit**: 8.9 GB / 228 GB
- **Note**: no zip files were actually created for this audit (measured folder sizes only, per request). Sizes below are on-disk usage from `du -sh`, which rounds — treat every figure as ±a few percent, not an exact byte count. Zip-size estimates use typical DEFLATE ratios per file type (60–70% reduction on JS/JSON/text, 5–10% on already-dense binary tensor data).

## At a glance

| | Frontend | Backend |
|---|---|---|
| Total (on disk) | 164 MB | 2.7 GB |
| Dominant contents | node_modules (97%) | model checkpoint weights (97%) |
| Est. zip, everything | ≈ 50–60 MB | ≈ 2.4–2.6 GB |
| Est. zip, source only | ≈ 1.5–2 MB | ≈ 150–200 KB |

Combined: **≈ 2.5 GB** zipped with everything included, or **≈ 2 MB** zipped as source-only code.

---

## Frontend — 164 MB

| Path | Category | Size | Share |
|---|---|---:|---:|
| `node_modules/` (8,168 files — installed dependencies) | Regenerable | 159 MB | 96.9% |
| `dist/` (production build output) | Build output | 2.8 MB | 1.7% |
| `public/` (static assets served as-is) | Static | 2.1 MB | 1.3% |
| `src/` (components/ 400K · lib/ 112K · App.tsx 12K · main.tsx, index.css 8K) | Source | 540 KB | 0.3% |
| `package-lock.json` | Source | 112 KB | 0.07% |
| Config files (package.json, 3× tsconfig, vite/tailwind/postcss config, index.html) | Source | 28 KB | 0.02% |

**Zip everything**: 164 MB → ≈ 50–60 MB. JS/JSON/text in `node_modules` is what DEFLATE is good at — typically 60–65% smaller.

**Zip source only** (excl. `node_modules`): 5.6 MB → ≈ 1.5–2 MB. `src/`, `dist/`, `public/`, config — everything you'd actually want to hand someone.

---

## Backend — 2.7 GB

| Path | Category | Size | Share |
|---|---|---:|---:|
| `custom_model/experiments/` (8 run folders × ~275 MB, plus `best_experiment_checkpoint.pt` at 137 MB) | Weights | 2.3 GB | 83.2% |
| `custom_model/checkpoints/` (`checkpoint_best.pt` + `checkpoint_latest.pt`, 137 MB each) | Weights | 275 MB | 9.9% |
| `custom_model/optimized/` (best 23 MB + fp16 23 MB + pruned 46 MB variants) | Weights | 92 MB | 3.3% |
| `custom_model/data/` (processed 27 MB · raw 2.1 MB · splits 216 KB) | Training data | 30 MB | 1.1% |
| `custom_model/model_store/` (`model_v1.pt` 23 MB + manifest/config JSON) | Weights | 23 MB | 0.8% |
| `custom_model` source, `evaluation/`, README (model.py, train.py, evaluate.py, etc. — 12 files) | Source | 460 KB | 0.02% |
| `training/` (pipeline, finetune, validation scripts) | Source | 76 KB | <0.01% |
| `main.py`, `model.py` (FastAPI entry point) | Source | 16 KB | <0.01% |
| `__pycache__/` (3 locations, compiled bytecode) | Regenerable | 448 KB | 0.02% |

**Zip everything**: 2.7 GB → ≈ 2.4–2.6 GB. 23 `.pt` checkpoint files are raw float32 tensors — high-entropy binary that DEFLATE barely touches (5–10% off, if that).

**Zip source only** (excl. weights & cache): ≈ 600 KB → ≈ 150–200 KB. Every `.py` file, README, and JSON config — no checkpoints, no raw training data.

---

## Where the 2.3 GB in `experiments/` actually goes

Every one of the 8 experiment folders (`baseline`, `lr_high`, `lr_low`, `batch_large`, `batch_small`, `weight_decay_high`, `weight_decay_zero`, `diffusion_short`) holds the same shape: a 137 MB `checkpoint_best.pt` and a 137 MB `checkpoint_latest.pt`, plus a few KB of logs. That's **~275 MB of near-duplicate structure repeated 8 times** for what's fundamentally a hyperparameter sweep.

The actual comparison data — `training_summary.json`, `training_log.jsonl` — is a few KB per run. If these runs were already compared and a winner picked, the checkpoints for the other 7 are the single biggest thing that could be deleted in this whole project.

---

## Recommendations

1. **Never zip or back up `node_modules` or the sibling `.venv`.** `.venv` sits at the project root (1.1 GB, outside both folders) — both regenerate in seconds from `npm install` / `pip install -r requirements.txt`.
   → **−1.26 GB**

2. **Prune the 7 experiment runs not actively being compared.** Keep each run's `training_summary.json` for the record; delete the two 137 MB checkpoints per run once a winner is chosen.
   → **up to −2.1 GB**

3. **If only the code needs to be shared, zip source-only.** Exclude `node_modules`, `dist`, every `*.pt`, `__pycache__`. Two zips, ~2 MB total, both emailable.
   → **164 MB → 2 MB**

4. **If the trained weights are actually needed, budget disk before zipping `backend/`.** A full backend zip will land near 2.5 GB — with only 8.9 GB free, that's roughly 30% of headroom for the archive alone, before accounting for where it gets copied to.
   → **plan ahead**
