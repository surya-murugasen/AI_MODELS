# AI Text-to-Image Generator — Transfer & Setup Guide

Based on direct inspection of `AI-Text-to-Image-Generator-transfer.zip` (5.78 MB, 1,202 files) — every file/command below was confirmed to exist in that ZIP, not assumed.

**What's actually inside the ZIP** (confirmed via `unzip -l`): `backend/main.py`, `backend/model.py`, `backend/__init__.py`, `backend/custom_model/` (source + docs, no `.pt` weights), `backend/training/`, `requirements.txt`, `requirements-training.txt`, `frontend/src/`, `frontend/public/`, `frontend/dist/` (already built), `frontend/package.json` + `package-lock.json`, config files, `README.md`, and project docs. Confirmed **absent**: `node_modules/`, `.venv/`, `__pycache__/`, `.DS_Store`, any `.pt` file.

Because `frontend/dist/` is already built and included, you technically only need step 8 to see the app running — steps 6–7 (frontend install/build) are needed only if you plan to edit the React code or want a from-scratch build. Both paths are marked below.

---

## 1. Software that must be installed first

| Software | Required version (from `README.md` / `requirements.txt`) | Required for |
|---|---|---|
| **Python** | 3.10+ | Backend (FastAPI + model) |
| **pip** | Comes with Python | Installing backend dependencies |
| **Node.js** | 18+ | Only needed if rebuilding the frontend (step 6–7). Not required to just run the already-built `frontend/dist/`. |
| **npm** | Comes with Node.js | Same as above |

No GPU is required — `backend/model.py` auto-detects CUDA → Apple Silicon (MPS) → CPU, in that order, and falls back to CPU automatically.

---

## 2. Where to extract the ZIP

Extract it anywhere you have a few hundred MB free (the model download in step 11 needs 2–4 GB more). Extracting creates a folder named `AI Text-to-Image Generator/` — that's your project root for every command below.

**macOS/Linux** (example — adjust the destination path to wherever you put the ZIP):
```bash
cd ~/Downloads
unzip AI-Text-to-Image-Generator-transfer.zip
```

**Windows PowerShell:**
```powershell
cd ~\Downloads
Expand-Archive -Path AI-Text-to-Image-Generator-transfer.zip -DestinationPath .
```

---

## 3. Enter the project directory

**macOS/Linux:**
```bash
cd "AI Text-to-Image Generator"
```

**Windows PowerShell:**
```powershell
cd "AI Text-to-Image Generator"
```

(Quotes are required — the folder name contains spaces.)

---

## 4. Create and activate the Python virtual environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now show `(.venv)` at the start of the line — that confirms it's active. Every command in steps 5, 8, and troubleshooting assumes this is active.

---

## 5. Install backend dependencies

Same command on all platforms (with `.venv` active):

```bash
pip install -r requirements.txt
```

This installs `fastapi`, `uvicorn`, `torch`, `diffusers`, `transformers`, `accelerate`, `Pillow`, `python-multipart` (exact contents of `requirements.txt` in the ZIP). `torch`/`diffusers`/`transformers`/`accelerate` are intentionally unpinned in the file (comment in `requirements.txt` explains why: pinning breaks on newer Python versions) — pip will resolve the latest compatible versions for your machine, so exact installed versions may differ slightly from the original laptop. This is expected.

---

## 6. Install frontend dependencies — *optional* (only if rebuilding the UI)

Skip this and step 7 if you're fine using the already-built `frontend/dist/` in the ZIP.

**macOS/Linux and Windows (identical):**
```bash
cd frontend
npm install
```

---

## 7. Build the frontend — *optional* (only if you ran step 6)

```bash
npm run build
cd ..
```

This regenerates `frontend/dist/`, overwriting the one already in the ZIP with a fresh build from `frontend/src/`.

---

## 8. Start the backend

With `.venv` active, from the project root (`AI Text-to-Image Generator/`):

**macOS/Linux:**
```bash
uvicorn backend.main:app --reload
```

**Windows PowerShell (identical):**
```powershell
uvicorn backend.main:app --reload
```

This single process serves **both** the API and the built React app (`backend/main.py` mounts `frontend/dist/` as static files) — there is no separate backend server + frontend server needed for normal use.

---

## 9. Start the frontend separately — *optional* (only for active development)

Only needed if you want hot-reload while editing React code. Requires steps 6 done. Run in a **second terminal**, alongside step 8's backend still running in the first:

```bash
cd frontend
npm run dev
```

`vite.config.ts` (confirmed in the ZIP) proxies `/api/*` calls from this dev server to `http://127.0.0.1:8000`, so both pieces work together.

---

## 10. URL to open

- **Normal use** (step 8 only): **http://127.0.0.1:8000**
- **Frontend dev mode** (step 9 also running): **http://127.0.0.1:5173** (exact port printed by `npm run dev` — Vite may pick a different port if 5173 is busy)

---

## 11. When/why Hugging Face models download

`backend/model.py` loads a text-to-image model from the Hugging Face Hub **lazily** — nothing downloads at install time or when `uvicorn` starts. The download happens the **first time you click Generate image** in the UI (i.e., the first `POST /api/generate` request), and is cached afterward so every request after that reuses it.

- Default model: `stabilityai/sd-turbo` (confirmed in `backend/model.py`)
- Also selectable from the UI's model dropdown: `runwayml/stable-diffusion-v1-5`, `stabilityai/stable-diffusion-2-1-base`
- Size: a few hundred MB to a few GB depending on which model, per `README.md`
- Requires internet access on the new laptop the first time each model is used
- Cached to `~/.cache/huggingface` (outside the project folder) — not part of this ZIP, and won't need re-downloading on subsequent runs on that same laptop
- Switching models in the dropdown triggers a fresh download+load for that model too, the first time it's picked

---

## 12. `.env` / API keys / configuration required?

**None.** Confirmed by inspecting the entire ZIP: no `.env` file exists anywhere in it, and no API key or secret is read by any code in `backend/`. The Hugging Face model downloads in step 11 are public, unauthenticated downloads.

The only configuration surface is three **optional** environment variables (all have working defaults if you skip them), read in `backend/model.py`:

| Variable | Default | Effect |
|---|---|---|
| `MODEL_ID` | `stabilityai/sd-turbo` | Which model loads before anyone picks one from the dropdown |
| `NUM_INFERENCE_STEPS` | `1` | Denoising steps per image |
| `GUIDANCE_SCALE` | `0.0` | How closely the model follows the prompt |

Example (optional — only if you want a different default model):
```bash
MODEL_ID="runwayml/stable-diffusion-v1-5" NUM_INFERENCE_STEPS=25 GUIDANCE_SCALE=7.5 uvicorn backend.main:app --reload
```
(Windows PowerShell equivalent: `$env:MODEL_ID="runwayml/stable-diffusion-v1-5"; uvicorn backend.main:app --reload`)

---

## 13. Platform command summary

| Step | macOS/Linux | Windows PowerShell |
|---|---|---|
| Extract | `unzip AI-Text-to-Image-Generator-transfer.zip` | `Expand-Archive -Path AI-Text-to-Image-Generator-transfer.zip -DestinationPath .` |
| Enter directory | `cd "AI Text-to-Image Generator"` | `cd "AI Text-to-Image Generator"` |
| Create venv | `python3 -m venv .venv` | `python -m venv .venv` |
| Activate venv | `source .venv/bin/activate` | `.venv\Scripts\Activate.ps1` |
| Install backend deps | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Install frontend deps *(optional)* | `cd frontend && npm install` | `cd frontend; npm install` |
| Build frontend *(optional)* | `npm run build && cd ..` | `npm run build; cd ..` |
| Start backend | `uvicorn backend.main:app --reload` | `uvicorn backend.main:app --reload` |
| Start frontend dev *(optional)* | `cd frontend && npm run dev` | `cd frontend; npm run dev` |

---

## 14. Troubleshooting

**`python3: command not found` / `python: command not found`**
Python isn't installed or isn't on PATH. Install Python 3.10+ from python.org, then retry. On Windows, try `py -m venv .venv` instead of `python -m venv .venv` if `python` isn't recognized.

**`.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled`** (Windows)
PowerShell's execution policy is blocking it. Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry activation.

**`ModuleNotFoundError: No module named 'fastapi'` (or torch/diffusers/etc.)**
The virtual environment isn't active, or `pip install -r requirements.txt` wasn't run (or failed partway). Confirm `(.venv)` shows in your prompt, then re-run step 5.

**`pip install` fails on `torch`**
Usually a Python-version/platform mismatch — `torch` doesn't ship wheels for every Python version. Confirm `python --version` is 3.10+ (ideally matching what's reasonably current), and check the error message for which platform/version it couldn't find a wheel for.

**`npm install` fails or hangs**
Confirm `node --version` is 18+. Delete nothing in the project — just retry `npm install`; transient registry errors are common and usually resolve on retry.

**Port 8000 already in use**
Something else is using it. Either stop that process, or run on a different port: `uvicorn backend.main:app --reload --port 8001`, then open `http://127.0.0.1:8001` instead.

**Browser shows "no UI will be served" warning, or a blank/404 page at :8000**
`frontend/dist/` is missing or wasn't found. It's included in this ZIP, so this shouldn't happen — but if it does, run steps 6–7 to rebuild it, then restart `uvicorn`.

**Generate image hangs for a long time on first use, or fails with a network/timeout error**
Expected on first use — see §11. It's downloading the model (a few hundred MB to a few GB). Check internet connectivity; if it's genuinely stuck, stop the server (`Ctrl+C`) and retry — Hugging Face downloads resume partial progress.

**Image generation is very slow**
Normal on CPU-only machines (no CUDA/MPS GPU detected). This is expected behavior, not a bug — `sd-turbo` (the default model) is chosen specifically because it's usable on CPU, just not fast.

**`Custom Model (Trained from Scratch)` option in the dropdown fails with an error**
Expected — confirmed in `backend/model.py`: this option deliberately raises `RuntimeError("...checkpoints will be generated in upcoming steps")`. It isn't wired into serving yet in this project (see `PROJECT_TRANSFER_AUDIT.md` if that file is available to you). Use one of the other three models instead.

---

## 15. Final verification checklist

- [ ] `(.venv)` shows in the terminal prompt after activation
- [ ] `pip install -r requirements.txt` completed with no errors
- [ ] `uvicorn backend.main:app --reload` starts and prints `Application startup complete` with no traceback
- [ ] Opening **http://127.0.0.1:8000** shows the app UI (sidebar with Image Generate, Gallery, History, Predictive AI, Story to Video, How Algorithms Work, Settings, About)
- [ ] `GET http://127.0.0.1:8000/api/health` (visit directly in the browser) returns JSON with `"status": "ok"`
- [ ] Typing a prompt and clicking **Generate image** produces an image (first click will be slow — model download, see §11)
- [ ] Switching the model dropdown to a different model and generating again also works (triggers its own first-time download)
- [ ] Dark mode toggle in the sidebar works
- [ ] (If you ran steps 6–7) `npm run build` completed with no errors and regenerated `frontend/dist/`
- [ ] (If you ran step 9) `npm run dev` starts, and the printed URL also loads the app with working `/api` calls
