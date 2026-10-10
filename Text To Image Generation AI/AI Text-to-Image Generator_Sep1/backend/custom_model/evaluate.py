"""
evaluate.py
-----------
STEP 8: MODEL EVALUATION

The FIRST and ONLY step in this pipeline permitted to read the test split.
Steps 6 (training) and 7 (tuning) were both built so the test split could
never be imported by them at all; this module is where it is finally used
-- exactly once, read-only, purely to report how the Step 7 winner
performs on data that has influenced it in no way whatsoever.

WHAT THIS STEP DOES
    1. Loads Step 7's winning checkpoint (`experiments/best_experiment_checkpoint.pt`
       + `best_experiment_config.json`) -- the exact weights and
       hyperparameters Step 7 selected by validation loss.
    2. Evaluates it on the 50-sample TEST split ONLY.
    3. Computes noise-prediction MSE/MAE two ways:
         (a) "matched validation protocol" -- one seeded random (t, noise)
             draw per sample, single pass -- IDENTICAL methodology to
             Step 6/7's `Trainer.validate()`, so the resulting number is
             directly, apples-to-apples comparable to Step 7's reported
             `best_val_loss`.
         (b) "multi-timestep evaluation" -- every one of the 50 test
             samples scored at 10 fixed, evenly-spaced timesteps across
             the whole diffusion process, each with its own deterministic
             seeded noise draw. A single random-t number depends on which
             timestep each sample happened to draw; averaging over a fixed
             timestep grid instead gives a fuller, lower-variance picture
             of noise-prediction accuracy across early/mid/late diffusion
             -- the appropriate way to summarize a *diffusion* model,
             not just a generic regressor.
    4. Generates a small number of real text-to-image samples from TEST
       captions (Step 5's `generate()`, run through the loaded weights) for
       qualitative inspection, saved as PNGs entirely outside `data/`.
    5. Writes one evaluation report with all of the above, plus the
       checkpoint's identity/hyperparameters and explicit guarantees that
       nothing here trained, tuned, or modified anything.

WHY TWO TEST METRICS INSTEAD OF ONE
    Step 6/7's val_loss is a single-random-timestep-per-sample number by
    construction (see train.py's `validate()`). Reporting only a
    differently-computed test number (e.g. averaged over many timesteps)
    would make the "test vs. validation" comparison apples-to-oranges. So
    (a) exists purely for that comparison, and (b) exists as the more
    thorough, primary quantitative result appropriate for a diffusion
    noise-prediction task.

GUARANTEES (enforced by construction, not just by convention)
    - No optimizer, no `.backward()`, no `loss.backward()`/`.step()` call
      anywhere in this file -- there is nothing here CAPABLE of updating a
      weight. Every forward pass runs under `@torch.no_grad()`.
    - The model is loaded once via `load_state_dict` from Step 7's
      checkpoint and never saved back to that checkpoint or anywhere else.
    - `ModelArchitectureConfig` is never touched; the model is
      reconstructed with Step 5's exact architecture.
    - The Step 3/4 dataset files (`data/raw`, `data/processed`, `data/splits`)
      are only ever opened for reading here, never written.
"""

import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.custom_model.config import (
    BEST_EXPERIMENT_CHECKPOINT_PATH,
    BEST_EXPERIMENT_CONFIG_PATH,
    DEFAULT_PROBLEM_SPEC,
    EVALUATION_DIR,
    EVALUATION_REPORT_PATH,
    EVALUATION_SAMPLES_DIR,
    PROCESSED_TENSORS_DIR,
    TEST_METADATA_PATH,
)
from backend.custom_model.model import TextToImageDiffusionModel, load_vocab_size
from backend.custom_model.train import TextToImageDataset, get_device, q_sample, set_seed

# NOTE: TEST_METADATA_PATH is imported here -- and ONLY here in the whole
# pipeline. train.py and experiment.py deliberately never import it.

EVAL_SEED = 42
NUM_TIMESTEPS_EVALUATED = 10   # evenly spaced across the diffusion process
NUM_QUALITATIVE_SAMPLES = 8
GENERATION_STEPS = 50          # reduced-stride reverse-diffusion for a quick qualitative look, not the full T-step schedule


# ---------------------------------------------------------------------------
# Load Step 7's winning checkpoint (read-only)
# ---------------------------------------------------------------------------
def load_best_checkpoint() -> Dict[str, Any]:
    if not BEST_EXPERIMENT_CHECKPOINT_PATH.exists() or not BEST_EXPERIMENT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Step 7 outputs not found ('{BEST_EXPERIMENT_CHECKPOINT_PATH}' / "
            f"'{BEST_EXPERIMENT_CONFIG_PATH}'). Run Step 7 (experiment.py) first."
        )
    with open(BEST_EXPERIMENT_CONFIG_PATH, "r", encoding="utf-8") as f:
        best_config = json.load(f)
    checkpoint = torch.load(BEST_EXPERIMENT_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    return {"best_config": best_config, "checkpoint": checkpoint}


def build_model_from_checkpoint(loaded: Dict[str, Any], device: torch.device):
    """Reconstructs the Step 5 architecture (untouched) with the exact
    problem/diffusion config the winning Step 7 experiment used, then loads
    its trained weights. If the diffusion schedule were reconstructed
    wrong, `load_state_dict` would fail outright on the registered
    `betas`/`alphas_cumprod` buffer shapes -- a built-in correctness check,
    not just an assumption."""
    hp = loaded["best_config"]["hyperparameters"]
    vocab_size = load_vocab_size(default=DEFAULT_PROBLEM_SPEC.vocab_size)
    problem_cfg = dataclasses.replace(DEFAULT_PROBLEM_SPEC, diffusion_timesteps=hp["diffusion_timesteps"])

    model = TextToImageDiffusionModel(vocab_size=vocab_size, problem_cfg=problem_cfg)
    model.load_state_dict(loaded["checkpoint"]["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # belt-and-suspenders: nothing here can accumulate gradients

    return model, problem_cfg, vocab_size


# ---------------------------------------------------------------------------
# Metric (a): matched validation protocol -- comparable to Step 7's val_loss
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_matched_protocol(model: TextToImageDiffusionModel, test_dataset: TextToImageDataset, device: torch.device, seed: int) -> Dict[str, Any]:
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    T = model.problem_cfg.diffusion_timesteps
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    total_mse, total_mae, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        token_ids = batch["token_ids"].to(device)
        B = images.shape[0]

        t = torch.randint(0, T, (B,), device=device, generator=gen)
        noise = torch.randn(images.shape, device=device, generator=gen)
        x_t = q_sample(model, images, t, noise)

        predicted_noise = model(x_t, t, token_ids)
        per_sample_mse = F.mse_loss(predicted_noise, noise, reduction="none").mean(dim=[1, 2, 3])
        per_sample_mae = F.l1_loss(predicted_noise, noise, reduction="none").mean(dim=[1, 2, 3])

        total_mse += per_sample_mse.sum().item()
        total_mae += per_sample_mae.sum().item()
        n += B

    return {"mse": total_mse / n, "mae": total_mae / n, "n_samples": n}


# ---------------------------------------------------------------------------
# Metric (b): multi-timestep evaluation across the complete test set
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_multi_timestep(model: TextToImageDiffusionModel, test_dataset: TextToImageDataset, device: torch.device, seed: int, num_timesteps: int) -> Dict[str, Any]:
    T = model.problem_cfg.diffusion_timesteps
    timesteps_to_eval = sorted({int(round(x)) for x in np.linspace(0, T - 1, num_timesteps)})

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    per_sample: List[Dict[str, Any]] = []
    per_timestep_mse: Dict[int, List[float]] = {t: [] for t in timesteps_to_eval}

    for rec in test_dataset.records:
        tensor_data = torch.load(PROCESSED_TENSORS_DIR / rec["tensor_file"], weights_only=False)
        image = tensor_data["image"].unsqueeze(0).to(device)
        token_ids = tensor_data["token_ids"].unsqueeze(0).to(device)

        sample_mses = []
        for t_val in timesteps_to_eval:
            t = torch.tensor([t_val], device=device)
            noise = torch.randn(image.shape, device=device, generator=gen)
            x_t = q_sample(model, image, t, noise)
            predicted_noise = model(x_t, t, token_ids)
            mse = F.mse_loss(predicted_noise, noise).item()
            sample_mses.append(mse)
            per_timestep_mse[t_val].append(mse)

        per_sample.append({
            "id": rec["id"],
            "prompt": rec["prompt"],
            "mean_mse_across_timesteps": float(np.mean(sample_mses)),
            "per_timestep_mse": {str(t): m for t, m in zip(timesteps_to_eval, sample_mses)},
        })

    mse_by_timestep = {str(t): float(np.mean(v)) for t, v in per_timestep_mse.items()}
    ordered_values = [mse_by_timestep[str(t)] for t in timesteps_to_eval]
    third = max(1, len(timesteps_to_eval) // 3)

    return {
        "num_test_samples": len(per_sample),
        "timesteps_evaluated": timesteps_to_eval,
        "overall_mean_mse": float(np.mean([r["mean_mse_across_timesteps"] for r in per_sample])),
        "mse_by_timestep": mse_by_timestep,
        "mse_early_diffusion_avg": float(np.mean(ordered_values[:third])),
        "mse_mid_diffusion_avg": float(np.mean(ordered_values[third:-third] or ordered_values)),
        "mse_late_diffusion_avg": float(np.mean(ordered_values[-third:])),
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Qualitative text -> image samples, saved outside data/
# ---------------------------------------------------------------------------
def denormalize_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Inverse of Step 3's normalization: [-1, 1] float CHW -> [0, 255] uint8 HWC."""
    arr = tensor.detach().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    return arr.permute(1, 2, 0).numpy()


@torch.no_grad()
def generate_qualitative_samples(
    model: TextToImageDiffusionModel,
    test_dataset: TextToImageDataset,
    device: torch.device,
    seed: int,
    num_samples: int,
    generation_steps: int,
) -> List[Dict[str, Any]]:
    EVALUATION_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # First N test records in id order -- a deterministic selection (not a
    # random subset), so the same test captions are illustrated every run.
    records = sorted(test_dataset.records, key=lambda r: r["id"])[:num_samples]

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    results = []
    for rec in records:
        tensor_data = torch.load(PROCESSED_TENSORS_DIR / rec["tensor_file"], weights_only=False)
        real_image = tensor_data["image"]
        token_ids = tensor_data["token_ids"].unsqueeze(0).to(device)

        generated = model.generate(token_ids, num_steps=generation_steps, generator=gen)  # [1,3,64,64]

        gen_arr = denormalize_to_uint8(generated[0])
        real_arr = denormalize_to_uint8(real_image)
        divider = np.full((real_arr.shape[0], 4, 3), 255, dtype=np.uint8)
        comparison_arr = np.concatenate([real_arr, divider, gen_arr], axis=1)

        gen_filename = f"test_{rec['id']:05d}_generated.png"
        compare_filename = f"test_{rec['id']:05d}_comparison_real_vs_generated.png"

        Image.fromarray(gen_arr).save(EVALUATION_SAMPLES_DIR / gen_filename)
        Image.fromarray(comparison_arr).save(EVALUATION_SAMPLES_DIR / compare_filename)

        results.append({
            "id": rec["id"],
            "prompt": rec["prompt"],
            "generation_steps": generation_steps,
            "generated_image_file": str(EVALUATION_SAMPLES_DIR / gen_filename),
            "comparison_image_file": str(EVALUATION_SAMPLES_DIR / compare_filename),
        })

    return results


# ---------------------------------------------------------------------------
# Report entry point
# ---------------------------------------------------------------------------
def run_step8_evaluation() -> Dict[str, Any]:
    print("=" * 70)
    print("  AI TEXT-TO-IMAGE GENERATOR: CUSTOM MODEL FROM SCRATCH")
    print("  STEP 8: MODEL EVALUATION")
    print("=" * 70)

    set_seed(EVAL_SEED)
    device = get_device()
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Loading Step 7's winning checkpoint...")
    loaded = load_best_checkpoint()
    best_config = loaded["best_config"]
    print(f"      Winning experiment: {best_config['name']} -- {best_config['description']}")
    print(f"      Step 7 best_val_loss: {best_config['best_val_loss']:.6f}")

    model, problem_cfg, vocab_size = build_model_from_checkpoint(loaded, device)
    print(f"      Model loaded on {device}, diffusion_timesteps={problem_cfg.diffusion_timesteps}, vocab_size={vocab_size}")

    print(f"\n[2/5] Loading the TEST split (never seen by training or tuning)...")
    test_dataset = TextToImageDataset(TEST_METADATA_PATH)
    print(f"      Test samples: {len(test_dataset)}")

    print(f"\n[3/5] Evaluating (matched validation protocol, for direct comparison to Step 7)...")
    matched = evaluate_matched_protocol(model, test_dataset, device, EVAL_SEED)
    print(f"      Test MSE (matched protocol): {matched['mse']:.6f}  |  Test MAE: {matched['mae']:.6f}  |  n={matched['n_samples']}")

    print(f"\n[4/5] Evaluating (multi-timestep, {NUM_TIMESTEPS_EVALUATED} timesteps x {len(test_dataset)} samples)...")
    multi = evaluate_multi_timestep(model, test_dataset, device, EVAL_SEED, NUM_TIMESTEPS_EVALUATED)
    print(f"      Overall mean MSE: {multi['overall_mean_mse']:.6f}")
    print(f"      Early/mid/late diffusion avg MSE: {multi['mse_early_diffusion_avg']:.6f} / {multi['mse_mid_diffusion_avg']:.6f} / {multi['mse_late_diffusion_avg']:.6f}")

    print(f"\n[5/5] Generating {NUM_QUALITATIVE_SAMPLES} qualitative text->image samples from TEST captions...")
    gen_start = time.time()
    qualitative = generate_qualitative_samples(model, test_dataset, device, EVAL_SEED, NUM_QUALITATIVE_SAMPLES, GENERATION_STEPS)
    gen_elapsed = round(time.time() - gen_start, 2)
    print(f"      Saved {len(qualitative)} generated images + side-by-side comparisons to: {EVALUATION_SAMPLES_DIR}  ({gen_elapsed}s)")

    val_loss = best_config["best_val_loss"]
    gap = matched["mse"] - val_loss

    report = {
        "step": "Step 8: Model Evaluation",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "checkpoint": {
            "source": "Step 7 best experiment",
            "experiment_name": best_config["name"],
            "experiment_description": best_config["description"],
            "checkpoint_path": str(BEST_EXPERIMENT_CHECKPOINT_PATH),
            "hyperparameters": best_config["hyperparameters"],
            "step7_best_val_loss": val_loss,
        },
        "test_set": {
            "test_metadata_path": str(TEST_METADATA_PATH),
            "num_test_samples": len(test_dataset),
            "note": "First and only use of the test split anywhere in this pipeline -- never used in Steps 6-7.",
        },
        "metrics": {
            "matched_validation_protocol": matched,
            "comparison_to_step7_validation": {
                "step7_best_val_loss": val_loss,
                "test_mse_matched_protocol": matched["mse"],
                "gap_test_minus_val": gap,
                "note": (
                    "Both numbers use the identical single-random-timestep-per-sample "
                    "protocol (Trainer.validate()'s methodology), so this gap isolates "
                    "the validation-set-to-test-set generalization difference, not a "
                    "difference in how the metric itself was computed."
                ),
            },
            "multi_timestep_evaluation": {k: v for k, v in multi.items() if k != "per_sample"},
        },
        "per_sample_test_predictions": multi["per_sample"],
        "qualitative_samples": qualitative,
        "configuration": {
            "eval_seed": EVAL_SEED,
            "vocab_size": vocab_size,
            "diffusion_timesteps": problem_cfg.diffusion_timesteps,
            "num_timesteps_evaluated_per_sample": NUM_TIMESTEPS_EVALUATED,
            "num_qualitative_samples": NUM_QUALITATIVE_SAMPLES,
            "generation_steps_for_qualitative_demo": GENERATION_STEPS,
        },
        "guarantees": {
            "test_set_used_for_training": False,
            "test_set_used_for_hyperparameter_tuning_or_checkpoint_selection": False,
            "model_weights_modified_during_evaluation": False,
            "model_architecture_modified": False,
            "dataset_or_test_samples_modified": False,
            "pretrained_weights_used": False,
        },
        "notes": (
            "Step 8 evaluation only -- loaded Step 7's winning checkpoint read-only, "
            "evaluated it on the untouched test split, and generated qualitative "
            "samples. No gradients, optimizer steps, or weight updates occurred "
            "anywhere in this run. The model at this point has only had a handful "
            "of short (3-epoch) experiments run on it (Steps 6-7 were pipeline "
            "verification runs, not a full training run), so both the quantitative "
            "loss and the qualitative generated images are expected to reflect an "
            "undertrained model, not a finished one. Step 9 (optimization) has not "
            "been run."
        ),
    }

    with open(EVALUATION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("  STEP 8 SUMMARY:")
    print(f"  - Test samples evaluated:          {len(test_dataset)} / {len(test_dataset)} (complete test set)")
    print(f"  - Test MSE (matched protocol):      {matched['mse']:.6f}")
    print(f"  - Test MAE (matched protocol):      {matched['mae']:.6f}")
    print(f"  - Step 7 best validation loss:      {val_loss:.6f}")
    print(f"  - Test - Val gap:                   {gap:+.6f}")
    print(f"  - Multi-timestep overall mean MSE:  {multi['overall_mean_mse']:.6f}")
    print(f"  - Qualitative samples generated:    {len(qualitative)}")
    print(f"  - Evaluation report:                {EVALUATION_REPORT_PATH}")
    print(f"  - Generated samples directory:      {EVALUATION_SAMPLES_DIR}")
    print("=" * 70)
    print("Step 8 (Model Evaluation) complete. No weights were modified.")
    print("Step 9 (optimization) has not been run.")
    print("=" * 70)

    return report


if __name__ == "__main__":
    run_step8_evaluation()
