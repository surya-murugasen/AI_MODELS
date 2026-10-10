# How This Project Works

This document explains the concepts behind the AI Text-to-Image Generator: what
was built, which model powers it, what data that model learned from, and
exactly how a text prompt turns into a picture. It's meant as a companion to
`README.md` (which covers setup and running the app) — this one is about the
ideas, not the commands.

## What we built

This project is a small full-stack application, not a new AI model. It's three
pieces wired together:

- A **React frontend** — the dashboard you interact with (Generate, Gallery,
  History, Settings, About).
- A **FastAPI backend** — a thin Python web server that receives your prompt,
  validates it, and calls into the AI layer.
- A **model wrapper** (`backend/model.py`) — a small piece of Python that loads
  a pre-trained Hugging Face Diffusers pipeline and runs it with PyTorch.

Nothing here was trained by this project. The very first time you click
Generate, the app downloads an already-trained model from the Hugging Face
Hub and simply runs it. Building this app is like building a car around an
engine someone else designed and built — the interesting engineering here is
the plumbing (the UI, the API, the request/response flow), not the engine
itself.

## The model

By default, the app uses [`stabilityai/sd-turbo`](https://huggingface.co/stabilityai/sd-turbo),
a distilled version of Stable Diffusion 2.1 released by Stability AI.

"Distilled" means Stability AI took the original, slower Stable Diffusion
model — which needs roughly 20 to 50 denoising steps to produce an image — and
trained a smaller, faster *student* model to reproduce similar results in as
few as **1 step**, using a technique called Adversarial Diffusion Distillation
(ADD). That's why this app can generate an image in a couple of seconds on a
laptop CPU or an Apple Silicon GPU, instead of the minute-plus that full
Stable Diffusion would take without a high-end GPU.

The tradeoff is that a heavily distilled model like sd-turbo is slightly less
flexible and can be a little less faithful to very specific or unusual
prompts than the full model. The README shows how to switch to
[`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
(the original, slower model, using more steps) via the `MODEL_ID` environment
variable, if you want to compare quality directly.

## The dataset

Neither this project nor sd-turbo's own training involved us collecting any
data — that work was done by Stability AI when they trained Stable Diffusion
2.1, the model sd-turbo was distilled from. That training used filtered
subsets of **LAION-5B**, a public dataset of roughly 5 billion image-text
pairs scraped from the public web — images paired with their alt text,
captions, or other nearby text. Stable Diffusion 2.x specifically used
further-filtered subsets of that (such as LAION-2B-en, filtered for
aesthetic quality and to remove explicit content).

That's where the model "learned" what a fox, a watercolor painting, or a
cyberpunk city look like: by seeing an enormous number of image-caption pairs
during training and learning the statistical relationship between words and
pixels well enough to generate new images it had never seen before.

## How a prompt becomes an image

This is the part that feels like magic but is really a specific, well-defined
pipeline. Every time you click **Generate**, the same sequence of steps runs:

### 1. Your text becomes numbers

Your prompt is first split into tokens (roughly word- or sub-word-sized
pieces) and then passed through a **CLIP text encoder** — a separate neural
network, already trained, whose only job is to turn a sequence of tokens into
a sequence of numerical vectors that capture their meaning. This is the same
kind of encoder used in a lot of text-image matching systems, and it isn't
specific to diffusion models.

### 2. Generation happens in a compressed "latent space," not in pixels

Working directly on full-resolution pixels would be slow, so the model
doesn't do that. A **VAE (variational autoencoder)** has separately learned
to compress a 512×512 image down to a much smaller grid of numbers (roughly
64×64×4) and decompress it back again with minimal loss. Image generation
starts from pure random noise in that small, compressed space — not from
random pixel noise.

### 3. A neural network removes noise, step by step, guided by your words

A **U-Net** (a convolutional network shaped like an hourglass — it shrinks the
input down and then expands it back up) looks at the current noisy latent and
your text embedding together, and predicts what noise to subtract to make the
image slightly more like something matching your prompt. It does this through
**cross-attention**: at each layer, the partially-formed image "attends to"
the relevant words in your prompt. This is the mechanism that makes "a red
fox" actually paint something red and fox-shaped, rather than a generic
animal in a random color.

This denoising step repeats for however many steps you've set (the
**Inference Steps** slider in the UI) — sd-turbo can do this in as little as
1 step; full Stable Diffusion typically needs 20 to 50.

### 4. Guidance scale controls how literally the model follows your prompt

At each step, the model actually runs its prediction twice: once "blind"
(unconditioned on your prompt) and once conditioned on it. The **Guidance
Scale** slider controls how strongly the final prediction is pulled toward
the prompt-conditioned version. Higher guidance means the image follows your
words more literally, at some cost to naturalness and diversity; lower
guidance (sd-turbo is tuned to work well at 0) lets the model be more free.

### 5. The final latent is decoded back into a real image

Once denoising finishes, the same VAE from step 2 — now running in reverse —
decodes the final latent back into a full-resolution RGB image. That image is
what `model.py` returns, what FastAPI encodes as a base64 PNG, and what your
browser displays.

## The short version

Your prompt never generates pixels directly. It steers a repeated
noise-removal process that happens in a compressed mathematical space, and a
separate decoder turns the end result into the picture you actually see. The
"AI" in this app is really three specialized, separately-trained networks
working together: a text encoder, a denoising U-Net, and a VAE — all wrapped
up for you by the Diffusers library.

## Where each concept lives in the code

| Concept | Where |
|---|---|
| Loading the pre-trained pipeline | `backend/model.py`, `ImageGenerator._load()` |
| Sending your prompt/steps/guidance/size to the model | `backend/model.py`, `ImageGenerator.generate()` |
| Choosing which model to download | `MODEL_ID` environment variable (see `README.md`) |
| Turning the result into something the browser can show | `backend/main.py`, `/api/generate` (base64 PNG encoding) |
| The sliders that control steps/guidance/size | `frontend/src/components/views/GenerateView.tsx` |
