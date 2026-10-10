# Project Transfer Audit

Goal checked against: zip this project → upload → download on another laptop → unzip → install deps → run with identical functionality.

**Method**: read-only inspection (structure, `du -sh`, dependency files, `grep` across all source for imports/paths/URLs). Nothing in the project was modified.

---

## Summary verdict

**Minor fixes needed** — not a hard blocker, but two real gaps found (below). The *core* Image Generate app (the thing `README.md` documents) is genuinely **ready to transfer as-is** with the commands in Section 9. The gaps are in the newer, undocumented parts of the project.

| | |
|---|---|
| Required transfer size | **≈ 2–3 MB** (source + config only) |
| Full transfer size (incl. trained model archive) | **≈ 2.5–2.6 GB** |
| Blocking issues found | **0** |
| Non-blocking issues found | **3** (see §7–8) |

---

## 1. Required files/folders — must be in the ZIP

| Path | Why |
|---|---|
| `backend/main.py`, `backend/model.py`, `backend/__init__.py` | The entire FastAPI app + model wrapper — this is the whole backend that actually runs |
| `requirements.txt` | Backend's Python dependencies |
| `frontend/src/` | All React/TypeScript source |
| `frontend/index.html` | Vite entry point |
| `frontend/package.json` + `frontend/package-lock.json` | Frontend dependency manifest — **the lockfile matters**, it's what guarantees the same dependency versions resolve on the new laptop |
| `frontend/vite.config.ts`, `tsconfig.json`, `tsconfig.app.json`, `tsconfig.node.json`, `tailwind.config.js`, `postcss.config.js` | Build/type-check configuration |
| `frontend/public/` | Static assets served as-is: demo CSV/XLSX files, background images, favicons — the app references these directly |
| `README.md` | Setup instructions (see §9) |

## 2. Optional files/folders — useful, not required

| Path | Notes |
|---|---|
| `backend/custom_model/*.py`, `custom_model/README.md`, `custom_model/config.py` and JSON reports (not the `.pt` files) | Fully documents a real from-scratch-trained diffusion model — valuable to keep for reference, but **not required to run the app** (see §4) |
| `backend/training/` | A separate, optional dataset-prep pipeline; README explicitly states it "does not touch, import, or get imported by `backend/main.py` or `backend/model.py`" |
| `requirements-training.txt` | Only needed if you run `backend/training/` |
| `HOW_IT_WORKS.md`, `2026-08-25.md`, `2026-08-26.md`, `2026-08-28.md` | Project history/docs — no functional role |
| `Train_Arrival_Prediction_Data.csv/.xlsx`, `demo_railway_train_data.csv/.xlsx` (root copies) | These appear to be reference copies — the ones the app actually serves live in `frontend/public/` (and get copied into `frontend/dist/` at build time). The root-level copies aren't imported by any code. |
| `frontend/dist/` | Pre-built static output. Optional: including it lets the new laptop run the app immediately without Node/npm installed at all (skip straight to `uvicorn`). Excluding it just means `npm run build` regenerates it — a normal setup step, not a problem either way. |

## 3. Regenerable files/folders — should NOT be included

| Path | Size | Regenerate with |
|---|---|---|
| `frontend/node_modules/` | 159 MB | `npm install` |
| `.venv/` (project root) | 1.1 GB | `python3 -m venv .venv && pip install -r requirements.txt` |
| `backend/__pycache__/`, `backend/custom_model/__pycache__/`, `backend/training/__pycache__/` | ~450 KB total | Auto-created by Python on first run |
| `.DS_Store` (found at project root, `backend/`, `frontend/`, `frontend/src/`) | negligible | macOS Finder metadata — useless (and slightly confusing) on Windows/Linux |

## 4. Model files — full inventory + required-to-run status

| Model | Type | Stored locally as a file? | Required to run the app? |
|---|---|---|---|
| `stabilityai/sd-turbo` (default) | Pretrained (Hugging Face) | **No** — downloaded to `~/.cache/huggingface` on first `/api/generate` call | **Yes**, for the default Image Generate flow — but it downloads itself, doesn't need to be in the ZIP |
| `runwayml/stable-diffusion-v1-5` | Pretrained (Hugging Face) | No — same, downloads on demand | Only if selected from the model dropdown |
| `stabilityai/stable-diffusion-2-1-base` | Pretrained (Hugging Face) | No — same | Only if selected from the model dropdown |
| `backend/custom_model/checkpoints/*.pt`, `experiments/**/*.pt` (16 files), `experiments/best_experiment_checkpoint.pt` | **Locally trained from scratch** (0% pretrained weights — verified in code and `model_manifest.json`) | Yes, 2.7 GB total | **No** — `main.py`'s `custom-scratch-model` option currently just raises `RuntimeError("...checkpoints will be generated in upcoming steps")`. Nothing in the serving path loads these files yet. |
| `backend/custom_model/optimized/*.pt` (3 files), `model_store/model_v1.pt` | Derived from the local checkpoint above (FP16 cast / pruned / repackaged, not re-trained) | Yes | Same as above — **not wired into serving** |

**Bottom line**: none of the 22 local `.pt` files are required for the app to function identically on the new laptop. They're a fully-documented, independently-verified (SHA-256 checksummed) trained model that simply isn't connected to the UI yet.

## 5. Internet-downloaded dependencies — what auto-downloads on the new laptop

| What | From | When | Needs internet? |
|---|---|---|---|
| npm packages (`react`, `gsap`, `@radix-ui/*`, etc.) | registry.npmjs.org | `npm install` | Yes |
| pip packages (`torch`, `diffusers`, `transformers`, `accelerate`, `fastapi`, etc.) | PyPI | `pip install -r requirements.txt` | Yes |
| Diffusion model weights (2–4 GB per README) | huggingface.co | First `/api/generate` request (not at install time) | Yes — this is the one to budget time/data for |
| (Optional) training dataset via `datasets.load_dataset(...)` | Hugging Face Hub | Only if you run `backend/training/prepare_dataset.py` | Yes, but only if you use this optional pipeline |

## 6. Environment/dependency files — checked

- **No `.env` file exists anywhere in the project** — no secrets, API keys, or machine-specific env config to carry over. Clean.
- `requirements.txt`: `torch`/`diffusers`/`transformers`/`accelerate` are deliberately left version-floored, not pinned (commented in the file) — "pip will pick the latest compatible versions for your machine." This is intentional and fine, but means the new laptop won't get byte-identical package versions, just compatible ones.
- `requirements-training.txt`: only `datasets`, `tqdm` — for the optional pipeline.
- `frontend/package-lock.json` exists and pins exact versions — `npm install` will reproduce the same dependency tree.
- No `engines` field in `package.json`, no `.nvmrc`/`.python-version` file — Node/Python minimums are documented in `README.md` prose only ("Python 3.10+", "Node.js 18+"), not enforced by tooling. This machine currently runs Python 3.14.3 / Node v25.8.2 — well above the stated floor.
- `MODEL_ID`, `NUM_INFERENCE_STEPS`, `GUIDANCE_SCALE` are optional environment variable overrides (documented in README) — not required to run.

## 7. Hidden/missing dependencies

- ⚠️ **`numpy` is imported directly** in `backend/custom_model/optimize.py`, `preprocess.py`, and `evaluate.py`, but **is not listed in either `requirements.txt` or `requirements-training.txt`**. It currently works only because `torch` happens to pull it in as a transitive dependency on this machine — there's no explicit guarantee of that on a fresh install. This doesn't affect the core app (`main.py`/`model.py` never import numpy directly), only the optional `custom_model` scripts.
- No system binaries required — grepped for `ffmpeg`, `subprocess`, `os.system`, `shutil.which` across the whole project: zero hits. The "Story to Video" feature is entirely client-side (browser Canvas/JS), no server-side video encoding dependency.
- Device selection is automatic and safe: `model.py` checks for CUDA → Apple Silicon MPS → CPU fallback, in that order, so a laptop without a GPU will still work (just slower).

## 8. Paths and hardcoded references

- **No functional hardcoded paths found.** Every path in the actual application code (`main.py`, `model.py`, `custom_model/config.py`, `training/config.py`) is resolved via `Path(__file__).resolve().parent`-style relative resolution — fully portable to any machine/location.
- `vite.config.ts`'s dev-server proxy target is `http://127.0.0.1:8000` — this is machine-independent (always means "whatever's running locally"), not a problem.
- ⚠️ **Cosmetic only**: several JSON report files under `backend/custom_model/{logs,evaluation,data/processed,experiments,optimized}/` contain this machine's absolute path (e.g. `"/Users/administrator/AI models/AI Text-to-Image Generator/backend/custom_model/checkpoints/checkpoint_latest.pt"`). Verified via code search that **nothing reads these fields back** to locate a file — they're write-only historical log entries from whenever each script last ran. Zero functional risk; the only effect is that opening those JSON files reveals this machine's username. If that's a concern, these specific report files (not the checkpoints) could be excluded or regenerated by re-running the corresponding script on the new laptop.
- ⚠️ **README inconsistency**: the setup instructions say `cd text-to-image-generator`, but the actual project folder is named `AI Text-to-Image Generator`. Also, `README.md`'s "Project structure" section predates the `custom_model/` pipeline and several newer frontend views (Predictive AI, Story to Video, How Algorithms Work) — it only documents the original Generate/Gallery/History/Settings/About feature set. Not a functional blocker, just stale documentation that could confuse someone following it fresh.
- No hardcoded usernames, hostnames, or ports found in any `.ts`/`.tsx`/`.py` source file.

## 9. Startup commands — exact, verified against the actual code

**Backend:**
```bash
cd "AI Text-to-Image Generator"
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Frontend (build once):**
```bash
cd frontend
npm install
npm run build
cd ..
```

**Run (serves both API and built UI from one process):**
```bash
uvicorn backend.main:app --reload
```
Then open **http://127.0.0.1:8000**.

**Optional — active frontend development** (two terminals, hot reload):
```bash
# Terminal 1
uvicorn backend.main:app --reload
# Terminal 2
cd frontend && npm run dev   # opens http://127.0.0.1:5173, proxies /api to :8000
```

**Optional — the separate custom_model pipeline** (not needed to run the app):
```bash
python -m backend.custom_model.data_collection   # etc., per custom_model/README.md
```

## 10. Recommended ZIP contents — explicit include/exclude

```
AI Text-to-Image Generator/
├── backend/
│   ├── __init__.py                 ✅ include
│   ├── main.py                     ✅ include
│   ├── model.py                    ✅ include
│   ├── custom_model/                ⚙ optional — include .py/.md/.json, EXCLUDE *.pt (2.7 GB)
│   │   └── __pycache__/             ❌ exclude
│   └── training/                    ⚙ optional — include, it's small (76 KB)
│       └── __pycache__/             ❌ exclude
│   └── __pycache__/                 ❌ exclude
├── frontend/
│   ├── src/                        ✅ include
│   ├── public/                     ✅ include
│   ├── index.html                  ✅ include
│   ├── package.json                ✅ include
│   ├── package-lock.json           ✅ include
│   ├── vite.config.ts              ✅ include
│   ├── tsconfig*.json              ✅ include
│   ├── tailwind.config.js          ✅ include
│   ├── postcss.config.js           ✅ include
│   ├── dist/                        ⚙ optional (regenerable via npm run build)
│   └── node_modules/                ❌ exclude (159 MB, regenerable)
├── requirements.txt                ✅ include
├── requirements-training.txt        ⚙ optional
├── README.md                       ✅ include
├── HOW_IT_WORKS.md                  ⚙ optional
├── 2026-*.md                        ⚙ optional
├── ARCHIVE_SIZE_AUDIT.md            ⚙ optional
├── PROJECT_TRANSFER_AUDIT.md        ⚙ optional (this file)
├── Train_Arrival_Prediction_Data.*   ⚙ optional (root copies; live copies already in frontend/public)
├── demo_railway_train_data.*         ⚙ optional (same)
├── .venv/                          ❌ exclude (1.1 GB, regenerable)
└── .DS_Store (all locations)       ❌ exclude
```

## 11. Final transfer size estimate

| ZIP contents | Uncompressed | Zipped (est.) |
|---|---|---|
| **Required only** (backend source + frontend source/config/public + README) | ≈ 2.8 MB | **≈ 1–1.5 MB** |
| **Required + all optional except custom_model weights** (adds custom_model source, training/, docs) | ≈ 3.5 MB | **≈ 1.5–2 MB** |
| **Everything, including the 2.7 GB trained-model archive** | ≈ 2.7 GB | **≈ 2.5–2.6 GB** (`.pt` tensor files barely compress) |

**Recommendation**: zip the "required + optional except weights" set. You get 100% of the app's functionality, all the documentation, and the full custom-model pipeline *source code* (so it can still be run/retrained on the new laptop if wanted) — at roughly 2 MB instead of 2.7 GB. The trained checkpoints aren't used by the running app today, so there's nothing to lose functionally by leaving them behind; if you want the trained weights preserved too, that's the one line item that costs the other 2.5 GB.

## 12. Transfer readiness rating

**Minor fixes needed** (not a hard blocker):

1. Add `numpy` to `requirements.txt` or `requirements-training.txt` (§7) — currently works by transitive luck, not by declaration.
2. `README.md`'s folder name (`text-to-image-generator`) and "Project structure" section are out of date relative to the current project (§8) — cosmetic, but worth a pass before handing this to someone unfamiliar with it.

Neither issue affects the **core Image Generate app** — that path (`backend/main.py` + `backend/model.py` + the built frontend) is ready to transfer exactly as documented in §9, right now, with no changes needed.
