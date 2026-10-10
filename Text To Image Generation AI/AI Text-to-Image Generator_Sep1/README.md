# AI Text-to-Image Generator (Learning Project)

A beginner-friendly text-to-image app built to demonstrate a simple AI
system architecture end to end:

```
User → Web UI → FastAPI → Diffusion Model → Generated Image → UI
```

- **Web UI** (`frontend/`): React + TypeScript + Tailwind CSS, with hand-built [shadcn/ui](https://ui.shadcn.com)-style components. A sidebar dashboard with five views — **Generate** (prompt, model/settings, and result panels plus stat tiles), **Gallery**, **History**, **Settings**, and **About** — with a light/dark theme toggle.
- **Backend** (`backend/main.py`): Python + FastAPI. Two endpoints — `POST /api/generate` (runs the model, returns a base64 PNG) and `GET /api/health` (reports device/model/defaults for the UI) — plus serving the built React app.
- **AI model** (`backend/model.py`): a pre-trained Hugging Face [Diffusers](https://github.com/huggingface/diffusers) pipeline running on PyTorch. Defaults to [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo), a small/fast model that produces an image in a single inference step — this matters because it makes the demo usable on a laptop CPU, where a full Stable Diffusion model would take minutes per image.

There is intentionally **no database, no authentication, and no cloud
deployment** — just the boxes in the diagram above, so it's easy to see how
each piece talks to the next. See "Deliberate simplifications" below (also
shown in-app under **About**) for exactly where this cuts corners on purpose.

## How a request flows through the system

1. You type a prompt in the Generate view and click **Generate image**.
2. The browser sends `POST /api/generate` with the prompt plus steps, guidance scale, width, and height as JSON.
3. FastAPI validates the request (via a Pydantic model) and calls into `model.py`.
4. `model.py` loads the Diffusers pipeline (only once, on the first request) and runs inference, producing a PIL image at the requested size.
5. FastAPI encodes that image as base64 PNG and returns it as JSON.
6. React decodes it into an `<img>` in the Result panel, and adds it to the in-memory Gallery/History.

## Requirements

- Python 3.10+
- Node.js 18+ and npm (only needed to build the UI; not required at runtime)
- ~2-4 GB free disk space (for the model weights, downloaded automatically on first run)
- A GPU is *not* required. It runs on CPU (slow), Apple Silicon via MPS (faster), or CUDA GPU (fastest) — the code auto-detects what's available.

## Setup

```bash
cd text-to-image-generator

# 1. Backend: create a virtual environment and install Python dependencies
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Frontend: install Node dependencies and build the React app
cd frontend
npm install
npm run build
cd ..

# 3. Run the server (serves both the API and the built UI)
uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser, type a prompt, and click
**Generate image**.

The first generate request will download the model from the Hugging Face Hub
(a few hundred MB to a few GB depending on which model you use) and load it
into memory — this can take a minute or two. Every request after that reuses
the already-loaded model.

## Run command

Once setup above has been done once, this is the only command you need to
start the app again later (from the `text-to-image-generator` folder):

```bash
source .venv/bin/activate && uvicorn backend.main:app --reload
```

Then open **http://127.0.0.1:8000**. Press `Ctrl+C` in the terminal to stop
the server. You only need to redo the `npm run build` step if you change the
React code (see "Developing the UI" below).

### Developing the UI (optional)

If you want to edit the React code and see changes live, run the frontend's
own dev server alongside the backend instead of rebuilding every time:

```bash
# Terminal 1
uvicorn backend.main:app --reload

# Terminal 2
cd frontend
npm run dev
```

Open the URL Vite prints (typically **http://127.0.0.1:5173**) — it proxies
`/api/*` calls to the FastAPI server on port 8000, so both pieces work
together during development. When you're done, `npm run build` produces the
static files FastAPI serves in step 3 above.

## Using the app

- **Generate**: write a prompt, optionally click **Enhance prompt** (a local, rule-based helper — see below), pick a model from the dropdown, pick an image size, adjust inference steps / guidance scale, then **Generate image**. Download, Share, or copy the prompt from the Result panel's toolbar; click any thumbnail in the strip below the image to bring it back up.
- **Gallery**: every image generated this session, as a grid.
- **History**: every prompt generated this session, with its settings and timing, as a list.
- **Settings**: live backend configuration (model, device, defaults), and the dark-mode toggle.
- **About**: the architecture walkthrough and the simplifications below, in-app.

## Swapping the model

The Generate view's Model & settings panel has a dropdown of curated models
(`backend/model.py`'s `AVAILABLE_MODELS`) — pick one and it's used on your
next **Generate image** click. Only one model is ever kept loaded in memory
at a time, so picking a different one unloads the current model and loads
the new one on that next request, which takes a while (longer still the
first time that model is used, since it's also downloaded). `GET
/api/models` reports the curated list plus whichever model is currently
loaded; `GET /api/health` reflects the same.

To change the curated list itself, or the model that's loaded before anyone
picks one, edit `AVAILABLE_MODELS` in `backend/model.py`, or set the
`MODEL_ID` environment variable before starting the server:

```bash
# A larger, higher-quality (but much slower on CPU) classic model, loaded by default:
MODEL_ID="runwayml/stable-diffusion-v1-5" NUM_INFERENCE_STEPS=25 GUIDANCE_SCALE=7.5 \
  uvicorn backend.main:app --reload
```

Any text-to-image model on the Hugging Face Hub compatible with
`diffusers.AutoPipelineForText2Image` should work — add its id/label to
`AVAILABLE_MODELS` to make it selectable from the dropdown too.

## Deliberate simplifications

The reference design this UI follows looks like a full SaaS product (user
accounts, a model picker, "Upgrade to Pro," persistent history). This project
intentionally keeps the *look* but not all of that scope, to stay a
beginner-friendly, local, no-database learning project:

- **Gallery & History are session-only** — kept in the browser's memory (a React state array), not a database. Refreshing the page clears them.
- **Enhance prompt** is a local, rule-based helper (`frontend/src/lib/enhancePrompt.ts`) that appends a few quality keywords — it does **not** call another AI model.
- **Model is switchable from a small curated dropdown** (see Generate), but only one model is ever kept in memory at a time — picking a different one unloads the current model and loads the new one on your next Generate, which can take a while (longer still the first time that model is used, since it's also downloaded). This keeps memory use bounded on a laptop instead of holding every model in RAM/VRAM at once.
- **No user accounts, "Upgrade to Pro," authentication, rate limiting, or cloud deployment** — this runs on your own machine, for one person, by design.

## Project structure

```
text-to-image-generator/
├── backend/
│   ├── main.py                    # FastAPI app: routes, request/response models, static file serving
│   └── model.py                   # Diffusers pipeline wrapper: model loading + inference
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Top-level state (prompt, settings, generations, theme) + view switching
│   │   ├── components/
│   │   │   ├── Sidebar.tsx        # Nav + dark mode toggle
│   │   │   ├── StatTile.tsx       # Small stat card used in the Generate view
│   │   │   ├── ui/                # Hand-built shadcn/ui-style Button, Card, Textarea, Slider, Switch, DropdownMenu
│   │   │   └── views/             # GenerateView, GalleryView, HistoryView, SettingsView, AboutView
│   │   └── lib/
│   │       ├── utils.ts           # `cn()` class-merging helper (standard shadcn/ui pattern)
│   │       ├── theme.ts           # Dark mode toggle + localStorage persistence
│   │       ├── types.ts           # Shared TS types (HealthInfo, GenerationRecord, image size presets)
│   │       └── enhancePrompt.ts   # Local "Enhance prompt" keyword helper
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.js         # Bright accent palette (bento-blue/orange/aqua/yellow/violet/magenta)
│   └── dist/                      # Built static files (created by `npm run build`; served by FastAPI)
├── requirements.txt
├── requirements-training.txt      # Extra deps for the training pipeline below (optional, not needed to run the app)
└── README.md
```

## Training data pipeline (separate, optional)

`backend/training/` is a standalone pipeline that downloads a Hugging Face
image+caption dataset, validates and preprocesses it, and stores it under
`backend/training/data/` in a format ready for a future diffusion
fine-tune. It does not touch, import, or get imported by `backend/main.py`
or `backend/model.py` — installing its dependencies or running it has no
effect on the generation app above. **It does not train anything yet.**

```bash
pip install -r requirements-training.txt
python -m backend.training.prepare_dataset --max-samples 50   # quick first run
```

See `backend/training/README.md` for the full explanation (what gets
validated/rejected and why, the output format, how to point it at a
different dataset, and what `finetune.py`'s stub documents for later).

## Notes / things to try next (for learning)

- Wire the "Enhance prompt" button up to a real small LLM call, and compare against the local heuristic.
- Persist Gallery/History to a file or a tiny SQLite database, to see what a minimal persistence layer adds to this architecture.
- Cache more than one loaded model at once (trading RAM/VRAM for instant switching) instead of unloading on every switch.
- This app has no auth, rate limiting, or queueing — appropriate for local learning, not for exposing on the public internet as-is.
