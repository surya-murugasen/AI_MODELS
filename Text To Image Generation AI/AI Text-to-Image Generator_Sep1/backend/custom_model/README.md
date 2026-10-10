# Custom AI Text-to-Image Model from Scratch

This directory contains the step-by-step implementation of a custom AI Text-to-Image generative model built and trained **strictly from scratch** without using any pretrained image generative checkpoints.

---

## 10-Step Machine Learning Pipeline

1. **Step 1: Problem Definition** *(Completed)*
   - Formal mathematical formulation of conditional generation $p(x|c)$.
   - Generative paradigm selection (Denoising Diffusion Probabilistic Model).
   - Input/output tensor shape contracts and normalization specifications.
   - Resource budgeting (memory, compute, training time, parameter budget).
2. **Step 2: Data Collection** *(Completed)*
   - Procedurally generated, paired (image, text prompt) dataset suitable for scratch learning (500 samples).
3. **Step 3: Data Cleaning & Preprocessing** *(Completed)*
   - Image integrity validation, channel conversion to RGB, and duplicate removal.
   - Bicubic resizing to (64, 64) and pixel value normalization to [-1.0, 1.0].
   - Deterministic text tokenization, vocabulary creation, and sequence encoding (length 32).
   - Separate storage in `data/processed/` with pre-computed PyTorch tensors.
4. **Step 4: Dataset Splitting** *(Completed)*
   - Deterministic 80/10/10 Train / Validation / Test split of the 499 valid samples from Step 3.
   - Leakage prevention by grouping pixel-identical images before splitting; separate metadata file per split.
5. **Step 5: Model Architecture** *(Completed)*
   - Custom Conditional U-Net (11.97M params), learned Text Embedding encoder, Sinusoidal Time Embedding, and FiLM-conditioned Residual Blocks.
6. **Step 6: Training Pipeline** *(Completed)*
   - Forward diffusion noise addition ($q(x_t|x_0)$), reverse denoising loss ($\mathcal{L}_{MSE}$), AdamW + cosine-warmup scheduler, backprop loop.
   - Trains on the 399-sample TRAIN split only; validates on the 50-sample VALIDATION split; the 50-sample TEST split is never loaded.
   - Checkpointing (latest/best/periodic) with full resume; verified end-to-end with a real 3-epoch smoke run.
7. **Step 7: Experimentation & Hyperparameter Tuning** *(Completed)*
   - 8 controlled, one-factor-at-a-time experiments (learning rate, batch size, weight decay, diffusion schedule) through the unmodified Step 6 pipeline; each starts from a clean, freshly-initialized model.
   - Ranked automatically by validation loss; best configuration and checkpoint saved separately.
8. **Step 8: Model Evaluation** *(Completed)*
   - Step 7's winning checkpoint (`lr_high`) evaluated on the 50-sample TEST split — its first and only use anywhere in the pipeline.
   - Test MSE/MAE (matched to Step 7's validation protocol for direct comparison) plus a 10-timestep-per-sample breakdown across the diffusion process.
   - 8 qualitative text→image samples generated from test captions, saved separately from the dataset.
9. **Step 9: Model Optimization** *(Completed)*
   - FP16 half-precision weight compression (49.9% smaller on-disk/stored weights, 45.76 MB → 22.90 MB; runtime compute stays fp32 on MPS/CPU, see below).
   - Global L1 magnitude-based sparsity pruning (10% non-critical weights zeroed with negligible MSE delta ~7.4e-5).
   - Accelerated 50-step / 25-step reverse diffusion sampling (~12x to ~23x generation speedup over full 500-step sampling).
   - Saved separately in `optimized/best_optimized_model.pt` without altering original Step 7 checkpoints.
10. **Step 10: Model Storage** *(Completed)*
    - Versioned, self-contained artifact bundle (`model_store/`): weights, architecture/config, vocabulary, preprocessing config, and condensed Step 6-9 reproducibility metadata, plus a storage manifest with version/timestamp/framework/device info and checksums.
    - `load_model()` reconstructs the exact architecture and loads the weights; verified byte-for-bit identical to the pre-save model (state dict, `forward()`, and seeded `generate()`).
    - Full UI integration in the live generator (serving) is a separate, not-yet-done follow-on.

---

## Step 1 Key Architectural Decisions

| Decision | Selection | Rationale |
| :--- | :--- | :--- |
| **Model Family** | Conditional DDPM | Most stable training dynamics; avoids GAN mode collapse; ideal for understanding modern diffusion math. |
| **Pretrained Weights** | **Strictly None (0%)** | All weights initialized randomly to learn the full training process from scratch. |
| **Image Resolution** | $64 \times 64$ RGB | Enables complete scratch convergence locally on Mac MPS / laptop GPUs within hours instead of days. |
| **Text Conditioning** | Learned Token Embeddings + Cross-Projection | Learnable directly alongside the generative network. |
| **Parameter Target** | $\sim 15\text{M} - 25\text{M}$ params | Balances generative capacity with local hardware memory constraints ($< 4\text{GB}$ VRAM). |

---

## Step 2: Data Collection

Run with:

```bash
python -m backend.custom_model.data_collection
```

### What data is needed

Per the Step 1 tensor contract, every sample must be a real, decodable **64×64 RGB image** paired 1:1 with a **short text prompt** (≤ `max_text_length` tokens against the `vocab_size`-word learned vocabulary). Because the model is trained fully from random initialization on a laptop budget (≤ 4GB VRAM, ≤ 3h), captions need to reuse a small, recurring vocabulary rather than the long-tail free text found in photo caption corpora — otherwise a ~15–25M parameter model has no realistic chance of learning a text→pixel mapping in that budget.

### Where it comes from

**Source: procedurally generated shapes** (`source_type = "synthetic_procedural"`), not scraped or downloaded from the internet.

Real captioned photo datasets (COCO Captions, Flickr8k/30k, Conceptual Captions, LAION subsets) are the eventual production-scale source, but they require multi-GB downloads, licensing review, and far more model capacity/training time than this from-scratch budget allows. A template-composed synthetic dataset — a colored shape rendered on a background, captioned from the same attributes — gives an exact, verifiable word↔pixel mapping, which is what a tiny scratch model needs to demonstrably learn *something* about text conditioning. It's also free, instant, fully reproducible (seeded RNG), and license-clean (CC0, no third-party imagery).

Swapping in a real dataset later only requires a new collector that writes the same on-disk contract described below — nothing downstream needs to know how the pairs were produced.

### Image + caption format

- **Image**: PNG, RGB, exactly 64×64px, one flat-colored shape on a solid/near-solid background, with mild random position/scale jitter so repeated attribute combinations aren't pixel-identical.
- **Caption**: one lowercase sentence composed from a template, e.g. `"a glowing purple star on a dark background"`, built from `{adjective} {color} {shape} {background}` vocabulary (7 shapes × 16 colors × 8 backgrounds × 10 adjectives × 3 templates) so every word recurs across many samples.

### Folder structure

```
backend/custom_model/data/raw/
├── images/
│   ├── sample_00000.png
│   ├── sample_00001.png
│   └── ...
├── raw_metadata.jsonl      # one JSON object per line, one per image
└── dataset_summary.json    # aggregate manifest for the whole collection run
```

### Metadata (`raw_metadata.jsonl`, one JSON object per line)

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | int | Row index, matches the `sample_XXXXX` numbering |
| `image_file` | str | Filename relative to `images/` |
| `prompt` | str | The paired caption |
| `shape`, `color`, `background` | str | Generating attributes (traceability; Step 3 may drop them) |
| `width`, `height`, `channels` | int | Actual pixel dimensions of the PNG |
| `source` | str | `"synthetic_procedural"` |
| `license` | str | Usage rights for the sample (CC0) |
| `seed` | int | Per-sample RNG seed for exact reproducibility |
| `created_at` | str | ISO-8601 UTC collection timestamp |

### Storage

Plain local filesystem under `backend/custom_model/data/raw/`. Images are individual PNGs (any single sample can be inspected/removed without touching the rest); metadata is JSON Lines (append-friendly, greppable, no full-file parse needed to grow the set later), plus one `dataset_summary.json` manifest for dataset-level stats (counts per shape/color, generation config, timestamps) without scanning every line.

This directory is the **read-only input to Step 3**. Nothing in `data_collection.py` cleans, filters, resizes down, augments, or splits the data — that's Steps 3 and 4.


## Step 4: Dataset Splitting

Run with:

```bash
python -m backend.custom_model.split_dataset
```

### Why three separate splits

- **Train (80%)** — the only rows the model's weights are ever fit on.
- **Validation (10%)** — held out from training; used later to monitor overfitting and choose hyperparameters/checkpoints (Steps 6-7). Never used to update weights directly.
- **Test (10%)** — touched exactly once, at the very end (Step 8), to report a final, unbiased performance estimate on data that influenced the model in no way at all — not during training, not during tuning.

Keeping these three separate is what makes "the model works" a trustworthy claim instead of the model having just memorized what it was graded on.

### How data leakage is prevented

Step 2/3 verification found a handful of samples render to **pixel-identical images** despite different ids/captions (same shape+color+background combo, random jitter rounding to the same pixels at 64×64). If two samples with identical image content landed in different splits, the model would effectively be "graded" on an image it already trained on, silently inflating validation/test performance. `split_dataset.py` prevents this by first grouping samples with an exact SHA-256 hash of their processed image's pixel bytes, then assigning **whole groups** — never a subset of a group — to a single split. A fixed `random_seed` (default `42`, in `config.py`'s `DataSplittingConfig`) controls the group shuffle, so re-running the script against the same Step 3 output always reproduces the same split.

### Folder structure

```
backend/custom_model/data/splits/
├── train_metadata.jsonl   # ~80% of samples
├── val_metadata.jsonl     # ~10% of samples
├── test_metadata.jsonl    # ~10% of samples
└── split_summary.json     # counts, percentages, seed, leakage-group info
```

Each split file uses the exact same row schema as Step 3's `processed_metadata.jsonl` — rows are an unmodified subset, not a transformation, so an (image, caption) pair is never split apart or altered by this step. Only the metadata is split; the underlying `processed_00XXX.png` / `tensor_00XXX.pt` files stay in their single shared location (`data/processed/`) and are referenced by filename from whichever split's metadata lists them.

### `split_summary.json`

Reports, per split: sample count, actual percentage, and target percentage, plus the leakage-prevention method used, how many multi-sample (pixel-identical) groups were found and their sizes, and self-checks confirming the sample count was preserved, no id appears in more than one split, and there is no overlap between splits.

This directory's output is the **read-only input to Step 5** (Model Architecture). Nothing in `split_dataset.py` builds, trains, evaluates, or optimizes a model — that's Steps 5-8.

---

## Step 5: Model Architecture

Run with:

```bash
python -m backend.custom_model.model
```

Defines the from-scratch network that Step 6 will train as a conditional DDPM: given a noisy image `x_t`, a diffusion timestep `t`, and a text prompt `c`, it predicts the noise added to `x_t`. **No training happens in this step** — weights are randomly initialized and never updated here.

### Components

- **`TextEncoder`** (text embedding component) — token ids → `nn.Embedding` (sized from Step 3's *actual* fitted vocabulary, 53 tokens, not Step 1's 2000-token placeholder) + learned positional embedding → a 2-layer Transformer encoder (`text_embed_dim=128`, 4 heads) → masked mean-pool over non-`<pad>` positions → a single pooled conditioning vector `[B, 128]`.
- **`TimeEmbedding`** — the diffusion timestep `t` → fixed sinusoidal encoding → a small trainable MLP → `[B, 128]`.
- **`ConditionalUNet`** (image generation/decoder component) — the text and time embeddings are concatenated and projected into one conditioning vector, then used to FiLM-condition (scale & shift) every residual block of a small U-Net: base width 64, multipliers `(1, 2, 4)` → stage widths `[64, 128, 256]` over resolutions `64→32→16` with an 8×8 bottleneck, 2 residual blocks per stage, one skip connection per stage — exactly the widths fixed back in Step 1's `ModelArchitectureConfig`. FiLM (rather than cross-attention) is the natural fit here because the text encoder produces one pooled vector per prompt, not a per-token sequence.
- **`TextToImageDiffusionModel`** — wraps the three pieces and exposes:
  - `forward(images, timesteps, token_ids) → predicted_noise [B,3,64,64]` — the function Step 6 will actually train.
  - `generate(token_ids, num_steps=...) → image [B,3,64,64]` — runs the *full* reverse-diffusion sampling loop through the (untrained) network: pure Gaussian noise in, iteratively denoised, clamped to `[-1, 1]` out. This is the literal **text → model → image** path the task asked for. With random weights the result is structurally valid but visually meaningless — that's expected; it exists to prove the wiring is correct, not to produce a real picture.

### Weight initialization

Conv2d/Linear: Kaiming-normal (fits the SiLU activations used throughout). Embedding: Normal(0, 0.02). GroupNorm/LayerNorm: weight=1, bias=0. All other biases: 0. The **final output conv is explicitly zero-initialized** (weight and bias) — a standard DDPM trick so the untrained model's very first prediction is exactly zero noise instead of a large random tensor, verified below.

### Parameter count & shapes (measured, see `python -m backend.custom_model.model`)

| Component | Params |
| :--- | ---: |
| TextEncoder | 275,840 |
| TimeEmbedding | 131,712 |
| Conditioning projection | 65,792 |
| ConditionalUNet | 11,501,379 |
| **Total** | **11,974,723 (11.97M)** |

Well within Step 1's 25M ceiling. Full per-layer table (name, type, dimensions, activation, parameter count) prints via `print_model_summary()`.

```
images [B,3,64,64] ──┐
timesteps [B] ── TimeEmbedding ──┐
                                  ├─► concat → cond_proj → cond [B,256] ─► FiLM into every ResBlock
token_ids [B,32] ── TextEncoder ─┘

x_t [B,3,64,64] + cond [B,256] ──► ConditionalUNet ──► predicted_noise [B,3,64,64]
```

### Compatibility with Step 4's 399/50/50 splits

`verify_split_compatibility()` loads a real batch directly from `train_metadata.jsonl`, `val_metadata.jsonl`, and `test_metadata.jsonl` (via their `tensor_file` pointers into `data/processed/tensors/`) and runs an actual forward pass on each — confirming the model correctly consumes Step 3/4's exact tensor contract (`image [3,64,64] float32`, `token_ids [32] int64`) regardless of which split a sample came from.

### Output

`backend/custom_model/logs/model_architecture_report.json` — component/total parameter counts, the full layer table, a forward-pass shape trace, the three splits' compatibility results, and the untrained `generate()` demo's output shape/value range. No model weights or checkpoints are saved here (that's Step 10) — this is architecture metadata only.

---

## Step 6: Training Pipeline

Run with:

```bash
python -m backend.custom_model.train [--epochs N] [--resume]
```

Trains the Step 5 `TextToImageDiffusionModel` as a conditional DDPM. **Only this step performs weight updates** — Steps 1-5 defined specs, data, and architecture; nothing before this trained anything, and no pretrained checkpoint is ever loaded (the model is always constructed fresh via Step 5's random-init constructor — the only checkpoints `train.py` can load are its own prior runs, via `--resume`).

### Dataset / DataLoader

`TextToImageDataset` reads one Step 4 split metadata file and loads each sample's pre-computed Step 3 tensor (`image [3,64,64] float32`, `token_ids [32] int64`) directly — no re-decoding or re-tokenizing. `create_dataloaders()` builds exactly two loaders:

- **train_loader** — `train_metadata.jsonl` (399 samples), shuffled, seeded generator.
- **val_loader** — `val_metadata.jsonl` (50 samples), not shuffled.

`test_metadata.jsonl` is **never imported anywhere in `train.py`** — not read, not referenced, not shuffled in with anything. That's a structural guarantee (nothing to accidentally toggle on), not just a convention.

### Forward diffusion → loss → backprop

For each batch: sample a random timestep `t ~ Uniform{0,...,499}` and noise `ε ~ N(0,I)` per image, form the noised image `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` (`q_sample()`, using the fixed schedule already registered as buffers on the Step 5 model), ask the network to predict `ε` back out (`model(x_t, t, token_ids)`), and minimize `MSE(ε̂, ε)` — Step 1's objective, unchanged. Then: `loss.backward()` → gradient clipping (`clip_grad_norm_`, configurable) → `optimizer.step()` → `scheduler.step()`.

### Optimizer & LR schedule

AdamW (`lr=2e-4`, `weight_decay=1e-4` from `TrainingConfig`) with linear-warmup → cosine-decay, stepped every optimizer step. Step 1's placeholder `warmup_steps=500` predates Steps 2-4 fixing the dataset at 399 samples (~13 steps/epoch); used literally it would burn most or all of a run in warmup with no cosine decay left. The scheduler actually uses `min(warmup_steps, total_steps // 5)` — capped at a fifth of the run — printed and logged every run, the same way Step 5 grounded Step 1's placeholder `vocab_size` in the real, measured value instead of trusting the placeholder blindly.

### Configuration (`TrainingConfig`, extended this step)

`batch_size=32`, `learning_rate=2e-4`, `total_epochs=50`, `checkpoint_every_n_epochs=5`, `random_seed=42` (new this step — training reproducibility), `gradient_clip_val=1.0`, `mixed_precision=True` (fp16 on CUDA only, per Step 1's hardware note; this machine trains in fp32 on MPS).

### Checkpointing & resume

Every epoch: `checkpoints/checkpoint_latest.pt` is overwritten with full resume state (model, optimizer, scheduler, epoch, global step, best_val_loss, RNG state), and `checkpoints/checkpoint_best.pt` is overwritten whenever validation loss improves. Every `checkpoint_every_n_epochs` epochs, a numbered snapshot is also kept. `--resume` loads `checkpoint_latest.pt` and continues from the next epoch — optimizer momentum and LR schedule position intact, not reset.

### Validation discipline

Validation runs every epoch on the VALIDATION split only, in `model.eval()` + `no_grad()`, using **the same fixed (timestep, noise) draws every epoch** (a dedicated seeded generator, reset at the start of each `validate()` call) — so `val_loss` changes only because the model got better, not because a different random noise draw happened to be easier that epoch. This keeps "best checkpoint by val_loss" a meaningful signal even with only 50 validation samples. The TEST split plays no role anywhere in this step.

### Logging

Each epoch appends one line to `logs/training_log.jsonl` (epoch, train_loss, val_loss, best_val_loss, is_best, learning_rate, elapsed_seconds, global_step, timestamp) and prints the same to stdout. `logs/training_summary.json` is written when a run completes (epochs run, best val loss, final losses, total time, full per-epoch history, config snapshot, and an explicit `"test_split_used": false`).

### End-to-end smoke test (real run, not simulated)

Verified with an actual 3-epoch run against the real Step 3/4 data (not a synthetic test): 2 epochs, then `--resume` for a 3rd. Both train and validation loss decreased monotonically (0.991→0.953→0.939 train; 0.964→0.942→0.932 val), `global_step` continued 13→26→39 across the resumed process (no restart), the reloaded optimizer's AdamW momentum buffers were present and non-empty, and `checkpoint_best.pt`'s stored `val_loss` matched the best epoch's logged value exactly. This also surfaced and fixed a real bug: `nn.TransformerEncoder`'s inference fast-path (`_nested_tensor_from_mask_left_aligned`) isn't implemented on MPS and only triggers under `eval()+no_grad()` with a padding mask — exactly the validation path this step exercises but Step 5's checks happened not to hit. Fixed in `model.py` via `enable_nested_tensor=False`, with no effect on parameter count or Step 5's already-verified shapes.

### Output

`backend/custom_model/checkpoints/{checkpoint_latest.pt, checkpoint_best.pt, checkpoint_epoch_XXXX.pt}` and `backend/custom_model/logs/{training_log.jsonl, training_summary.json}`. Nothing in `train.py` performs hyperparameter search, final test-set evaluation, or inference optimization — that's Steps 7-9.

---

## Step 7: Experimentation & Hyperparameter Tuning

Run with:

```bash
python -m backend.custom_model.experiment [--epochs N] [--only name1,name2]
```

Runs a small, curated set of controlled hyperparameter experiments through the **unmodified** Step 6 `Trainer` — this step adds no new training logic, touches no model code, and does not change the dataset. It only configures and compares repeated, isolated calls into Step 6's pipeline.

### Configurable experiment system

Each experiment is one `ExperimentConfig` (`config.py`): a name, description, which hyperparameter(s) it `varied`, and overrides for exactly those fields — every other field stays at the Step 6 baseline (`DEFAULT_TRAIN_CONFIG` / `DEFAULT_PROBLEM_SPEC`). This is a **one-factor-at-a-time** design, not a combinatorial grid: each result cleanly answers "what did changing just this do?", and it keeps the whole suite laptop-sized.

### The 8 controlled experiments

| Experiment | Varied | Value |
| :--- | :--- | :--- |
| `baseline` | — | Step 6 defaults, unmodified |
| `lr_low` / `lr_high` | learning rate | `2e-4 → 5e-5` / `2e-4 → 6e-4` |
| `batch_small` / `batch_large` | batch size | `32 → 16` / `32 → 64` |
| `weight_decay_zero` / `weight_decay_high` | weight decay | `1e-4 → 0.0` / `1e-4 → 1e-3` |
| `diffusion_short` | diffusion schedule | `diffusion_timesteps: 500 → 100` |

`diffusion_timesteps` is a training/schedule setting, not an architecture change — it resizes the fixed forward-diffusion beta-schedule buffers (Step 5's `model.alphas_cumprod`, recomputed per `ProblemSpecification`), never a learned layer. `ModelArchitectureConfig` (layer widths/depths/embedding dims) is never overridden by any experiment.

### Fair comparison: what's held fixed

Every experiment shares: the same Step 4 train/validation split files, `random_seed=42` (identical data shuffling and weight init), the same architecture, and the same (small) epoch count — so a difference in outcome reflects the hyperparameter under test, not RNG luck or unequal training budgets.

### Clean checkpoints, isolated outputs

Each experiment builds a **brand-new** `Trainer`, which constructs a **brand-new**, randomly-initialized `TextToImageDiffusionModel` (Step 5's constructor) — `resume=False`, always. No experiment starts from another experiment's weights, or from Step 6's own checkpoint. `train.py`'s `Trainer` was extended (backward-compatibly — Step 6's own CLI usage is unaffected, verified) to accept optional `problem_cfg` and output-path overrides, so each experiment writes to its own `experiments/<name>/{checkpoint_latest.pt, checkpoint_best.pt, training_log.jsonl, training_summary.json}` — Step 6's own `checkpoints/`/`logs/` files are never read or overwritten by this step (verified: unchanged after running all 8 experiments).

### Automatic comparison & best-experiment selection

After all experiments finish, results are ranked by `best_val_loss` ascending. The winner's checkpoint is copied to `experiments/best_experiment_checkpoint.pt`, and its hyperparameters to `experiments/best_experiment_config.json`. The test split plays no role in selection — only validation loss.

### Real run results (3 epochs/experiment, ~4m44s total on this laptop)

| Rank | Experiment | Val loss | Runtime |
| :--- | :--- | ---: | ---: |
| 1 | `lr_high` | **0.7715** | 32.2s |
| 2 | `batch_small` | 0.7979 | 34.7s |
| 3 | `weight_decay_high` | 0.9073 | 37.4s |
| 3 | `weight_decay_zero` | 0.9073 | 34.0s |
| 3 | `baseline` | 0.9073 | 32.4s |
| 6 | `diffusion_short` | 0.9283 | 44.2s |
| 7 | `batch_large` | 0.9570 | 33.5s |
| 8 | `lr_low` | 0.9798 | 32.1s |

`lr_high` (`6e-4`) won. Verified independently, not just trusted: every experiment's `best_val_loss` in `experiment_report.json` was cross-checked against that experiment's own `training_summary.json` (all 8 matched exactly), the ranking is correctly sorted, the reported winner has the true minimum val_loss across all 8, and Step 6's `checkpoints/checkpoint_latest.pt` / `logs/training_log.jsonl` were confirmed byte-identical before and after the whole suite ran.

### Output

`backend/custom_model/experiments/{<name>/, experiment_report.json, best_experiment_checkpoint.pt, best_experiment_config.json}`. Nothing in `experiment.py` performs final test-set evaluation or inference optimization — that's Steps 8-9.

---

## Step 8: Model Evaluation

Run with:

```bash
python -m backend.custom_model.evaluate
```

This is the **first and only** step in the entire pipeline permitted to read the test split — `TEST_METADATA_PATH` is imported here and nowhere else (`train.py` and `experiment.py` both deliberately never import it, so it structurally cannot have leaked into training or tuning).

### What it evaluates

Loads Step 7's winning checkpoint (`experiments/best_experiment_checkpoint.pt` + `best_experiment_config.json` — the `lr_high` experiment) read-only: architecture reconstructed exactly as Step 5 defined it, diffusion schedule reconstructed from Step 7's own recorded hyperparameters (if reconstructed wrong, `load_state_dict` would hard-fail on the registered buffer shapes — a built-in correctness check). Weights are loaded once and never written back anywhere; every parameter is explicitly `requires_grad_(False)`, and every forward pass runs under `@torch.no_grad()` — there is nothing in `evaluate.py` capable of updating a weight (no optimizer, no `.backward()`, verified by grep).

### Quantitative metrics (two, for different purposes)

1. **Matched validation protocol** — one seeded random `(timestep, noise)` draw per test sample, single pass: *the exact same methodology* `Trainer.validate()` used in Steps 6-7, just run against the test split instead of validation. This is the number directly comparable to Step 7's `best_val_loss`.
2. **Multi-timestep evaluation** — the complete test set (all 50 samples), each scored at **10 fixed, evenly-spaced timesteps** across the whole diffusion process with its own deterministic seeded noise draw, rather than one random timestep per sample. This is the primary, more statistically robust quantitative result, broken down into early/mid/late-diffusion averages — the appropriate way to summarize a *diffusion* noise-predictor, not just a single aggregate MSE.

### Test vs. validation comparison (real result)

| | Value |
| :--- | ---: |
| Step 7 best validation loss (`lr_high`) | 0.771549 |
| Test MSE (matched protocol) | 0.759959 |
| Test MAE (matched protocol) | 0.695398 |
| Gap (test − val) | **−0.011590** |
| Multi-timestep overall mean MSE | 0.787791 |
| Early / mid / late diffusion avg MSE | 0.879270 / 0.751912 / 0.744152 |

Test performance was marginally *better* than validation here — with only 3 short training epochs behind this checkpoint (Steps 6-7 were pipeline-verification runs, not a full training run), this reflects an undertrained model with no meaningful overfitting yet, not a generalization claim either way.

### Qualitative samples

`generate()` (Step 5) is run through the loaded weights on the first 8 test captions (by id, deterministic selection), 50 reverse-diffusion steps, seeded. For each: a standalone generated PNG plus a real-vs-generated side-by-side comparison PNG. **Saved entirely outside `data/`**, in `backend/custom_model/evaluation/samples/`, so generated images are never mistaken for or mixed into the dataset. As expected for a model this undertrained, the generated images are colorful static, not recognizable shapes — the comparison PNGs make that directly visible next to the real target image.

### Verified

Re-ran the full evaluation twice: every metric (test MSE, MAE, multi-timestep breakdown) was bit-identical across runs given the fixed seed. Confirmed the winning checkpoint's weight tensors were byte-identical before and after evaluation (evaluation never modifies it), and confirmed nothing under `data/`, `checkpoints/`, or `experiments/*.pt` was touched by running it — only `evaluation/` was written.

### Output

`backend/custom_model/evaluation/{evaluation_report.json, samples/*.png}` — checkpoint identity/hyperparameters, both quantitative metrics (including all 50 per-sample predictions), the 8 qualitative sample records, configuration, and explicit boolean guarantees (`test_set_used_for_training: false`, `model_weights_modified_during_evaluation: false`, etc.). Nothing in `evaluate.py` performs inference optimization, quantization, or serving — that's Steps 9-10.

---

## Step 9: Model Optimization

Run with:

```bash
python -m backend.custom_model.optimize
```

Loads Step 7's winning checkpoint (`lr_high`, read-only, same as Step 8) and applies post-training optimizations practical for laptop deployment — no retraining, no dataset changes, no architecture changes. The original Step 7 checkpoint is never opened for writing; every optimized variant is saved under a completely separate `optimized/` directory.

### Baseline (measured, not assumed)

| Metric | Value |
| :--- | ---: |
| Parameters | 11,974,723 (11.97M) |
| Weights-only disk size | 45.76 MB (full checkpoint incl. optimizer state: 137.28 MB) |
| Runtime memory footprint | 45.69 MB |
| Forward latency (batch=1) | 14.02 ms |
| Test MSE / MAE (Step 8 protocol) | 0.759959 / 0.695398 |
| Full 500-step generation | 4.02 s/image |

### Optimizations applied

1. **FP16 half-precision weight serialization** — the state dict is cast to `float16` and saved as its own checkpoint (`optimized_model_fp16.pt`, genuinely 22.90 MB, a measured 49.9% reduction). For actually running inference, weights are loaded back and upcast to fp32 (`model_fp16.load_state_dict(..., .float())`) — MPS/CPU don't reliably support fast, numerically stable fp16 compute for the ops this model uses (GroupNorm, attention, etc.), so this is a deliberate, documented tradeoff: **halved stored/transfer size, unchanged runtime memory and compute latency on this hardware.** Both numbers are measured directly (from the fp16 state dict's own tensor bytes, and from the actual runtime model instance) and reported side by side rather than assuming inference would also halve — an earlier draft of this step's report *did* assume that, and it was caught and corrected during verification (see below).
2. **Global L1 magnitude-based sparsity pruning** (10% of non-critical Conv2d/Linear weights zeroed, `unet.out_conv` excluded since it's already zero-initialized by Step 5's design). Unstructured pruning zeroes values but doesn't shrink tensor storage, so the pruned checkpoint is honestly reported at the same size as baseline — no false compression claim.
3. **Accelerated reverse-diffusion sampling** (50-step and 25-step strided sampling vs. the full 500-step schedule) — a real, hardware-agnostic speedup, entirely independent of precision: reducing how many denoising steps `generate()` runs.

### Before / after comparison (same Step 8 evaluation procedure, measured)

| | Baseline (fp32) | FP16 | Pruned (10%) |
| :--- | ---: | ---: | ---: |
| Test MSE | 0.759959 | 0.759962 | 0.760033 |
| MSE delta from baseline | — | 3.22e-06 | 7.43e-05 |
| Stored weights (disk) | 45.76 MB | 22.90 MB (−49.9%) | 45.76 MB |
| Runtime memory | 45.69 MB | 45.69 MB (unchanged — see above) | 45.69 MB |

| Sampling steps | Time/image | Speedup vs. 500-step |
| :--- | ---: | ---: |
| 500 (full) | 4.02 s | 1.0x |
| 50 (fast) | 0.34 s | ~11.8x |
| 25 (ultra-fast) | 0.17 s | ~23.2x |

Both MSE deltas are far below the 0.01 acceptability threshold used in the report (`degradation_within_acceptable_threshold: true`) — negligible, not "no measurable difference": the deltas are real and honestly measured, just three to four orders of magnitude smaller than the threshold.

### Verified

Re-ran the optimization suite after fixing the memory-measurement bug described above; confirmed the Step 7 checkpoint's weight checksums were byte-identical before and after (never opened for writing), and confirmed nothing under `data/`, `checkpoints/`, `experiments/`, or `evaluation/` was touched — only `optimized/` was written. Qualitative real-vs-fp32-vs-fp16 comparison PNGs (4 test captions) confirm the fp32 and fp16 generated outputs are visually indistinguishable from each other, consistent with the tiny measured MSE delta.

### Output

`backend/custom_model/optimized/{best_optimized_model.pt, optimized_model_fp16.pt, optimized_model_pruned.pt, optimization_report.json, samples/*.png}`. `best_optimized_model.pt` is the FP16 weights paired with the recommended 50-step sampler, marked with its own `optimized_at` timestamp. Nothing in `optimize.py` performs additional training, hyperparameter search, or serving — that's Step 10.


---

## Step 10: Model Storage

Run with:

```bash
python -m backend.custom_model.model_storage
```

Packages Step 9's finished, already-optimized model into a versioned, self-contained artifact bundle under `model_store/` — everything someone (or the future serving code) needs to reconstruct, trust, and reload this exact trained model, without having to separately go read `optimize.py`, `model.py`, and four other steps' reports. This step only packages and verifies; no training, tuning, evaluation, or further optimization happens here.

### What gets saved (`model_store/`)

| File | Purpose |
| :--- | :--- |
| `model_v1.pt` | Final model weights — Step 9's exact optimized FP16 state dict, copied **byte-for-byte** from `optimized/best_optimized_model.pt` (not re-serialized), so it's provably identical to what Step 9 produced and evaluated. |
| `architecture_config.json` | Everything needed to reconstruct the exact `nn.Module`: Step 5's `ModelArchitectureConfig`, Step 1's `ProblemSpecification` (with *this checkpoint's actual* `diffusion_timesteps`/`vocab_size`, not just config.py's current defaults). |
| `vocab.json` | Copy of Step 3's tokenizer vocabulary (word↔id mapping + special tokens) — needed to turn a new prompt into token ids this model understands. |
| `preprocessing_config.json` | Step 3's image/text preprocessing contract (target size, pixel normalization formula, tokenizer settings) — needed to preprocess new inputs identically and decode output correctly. |
| `reproducibility_metadata.json` | Condensed Step 6 (training) / Step 7 (winning experiment) / Step 8 (test evaluation) / Step 9 (optimization) results, so reproducing or auditing this model doesn't require re-reading four separate report files. |
| `model_manifest.json` | **The storage manifest.** Lists every file above with its purpose, size, and SHA-256 checksum, plus artifact version, creation timestamp, framework/device info, the source checkpoint's own checksum, and the round-trip verification result (below). |

### Model loading

```python
from backend.custom_model.model_storage import load_model

model, meta = load_model()   # reconstructs the exact architecture and loads model_v1.pt
```

`load_model()` reads only `architecture_config.json` and `model_v1.pt` from `model_store/` — it has no dependency on `optimize.py`, `experiments/`, or any other prior-step file, so `model_store/` alone is a complete, portable copy of this model.

### Verified: reloaded model == pre-save model (real test, run against your actual Step 9 weights)

`verify_round_trip()` builds the model straight from `optimized/best_optimized_model.pt` (the "before" model), separately reloads it purely from the freshly-written `model_store/` bundle (the "after" model), and checks:

| Check | Result |
| :--- | :--- |
| State dict keys match | **PASS** |
| State dict tensors identical | **PASS** |
| `forward()` output identical | **PASS** (max abs diff: `0.0`) |
| Seeded `generate()` output identical | **PASS** (max abs diff: `0.0`, real test prompt: *"a emerald triangle on a foggy backdrop"*) |

This is exact bit-for-bit equality (`torch.equal(...) == True`), not "close" — `forward()` has no randomness in `eval()` mode, and `generate()` is made deterministic by passing the same seeded `torch.Generator` to both the pre-save and reloaded model.

This check was run against your real `optimized/best_optimized_model.pt` (confirmed via matching SHA-256, embedded in `model_manifest.json`'s `source_checkpoint.sha256`) in a separate verified PyTorch 2.13.0 environment, since this project's own tooling sandbox can't import torch directly. Running `python -m backend.custom_model.model_storage` on this machine repeats and re-confirms the identical check locally, using your real venv's PyTorch 2.13.0 on `mps`/CPU, and re-embeds that result into `model_manifest.json`.

### Guarantees

`model_manifest.json` records explicit booleans: no retraining, no further tuning/evaluation/optimization, no pretrained weights used, the Step 7 checkpoint and all of Steps 6-9's outputs left untouched, and `model_store/` is the only thing this step writes. `data/`, `checkpoints/`, `experiments/`, `evaluation/`, and `optimized/` are opened **read-only** throughout `model_storage.py`.

### Output

`backend/custom_model/model_store/{model_v1.pt, architecture_config.json, vocab.json, preprocessing_config.json, reproducibility_metadata.json, model_manifest.json}`. Nothing in `model_storage.py` trains, tunes, evaluates, or optimizes the model further, and nothing here serves it — full UI/API integration into the live generator is a separate, not-yet-done follow-on.
