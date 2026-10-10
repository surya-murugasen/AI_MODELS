"""
model_storage.py
-----------------
STEP 10: MODEL STORAGE

Packages Step 9's final optimized model (`optimized/best_optimized_model.pt`)
into a versioned, self-contained artifact bundle under `model_store/`, and
provides the loading function any future consumer (Step-10-and-beyond code,
the live web app, a notebook) should use to reconstruct the exact trained
model from that bundle. This step only PACKAGES, DOCUMENTS, and VERIFIES
an already-finished model -- it does not train, tune, evaluate, or further
optimize anything.

WHY A SEPARATE STORAGE STEP
    Steps 1-9 each write their own outputs to their own directory
    (data/raw, data/processed, data/splits, checkpoints/, experiments/,
    evaluation/, optimized/), but none of them is a self-contained,
    "hand this to someone else" package: reconstructing the model from
    `optimized/best_optimized_model.pt` alone requires knowing, from
    memory or by reading several other files, which architecture config
    produced it, which vocabulary it was tokenized against, and what its
    measured performance was. `model_store/` collects exactly that -- the
    weights plus everything needed to reload and trust them -- in one
    place, versioned and checksummed.

WHAT GETS SAVED (see save_model_artifact() and STORAGE MANIFEST below)
    model_v1.pt               Step 9's exact optimized weights, copied
                               byte-for-byte (not re-serialized) so the
                               stored file is provably identical to the
                               one Step 9 produced and evaluated.
    architecture_config.json  Every field needed to reconstruct the exact
                               nn.Module: ModelArchitectureConfig (Step 5),
                               ProblemSpecification incl. the diffusion
                               schedule actually used, and the fitted
                               vocab_size this checkpoint was trained
                               against.
    vocab.json                Copy of Step 3's tokenizer vocabulary
                               (word<->id mapping + special tokens) --
                               required to turn a new text prompt into the
                               token_ids this model expects.
    preprocessing_config.json Step 3's image/text preprocessing contract
                               (target size, pixel range, tokenizer
                               special tokens, max sequence length) --
                               required to preprocess a new input image
                               identically to how the model was trained,
                               and to decode its output correctly.
    reproducibility_metadata.json
                               Condensed Step 6 (training) / Step 7
                               (winning experiment) / Step 8 (test
                               evaluation) / Step 9 (optimization) results
                               -- what this exact model was trained with,
                               tuned against, and measured to do, without
                               having to go re-read four separate report
                               files.
    model_manifest.json        THE STORAGE MANIFEST. Top-level document
                               listing every artifact above (purpose, path,
                               size, sha256), model version, creation
                               timestamp, framework/device info, and the
                               result of the round-trip verification test
                               (see below).

WHAT IS DELIBERATELY NOT DONE HERE
    - No optimizer, no `.backward()`, no gradient computation anywhere in
      this file -- there is nothing here CAPABLE of updating a weight.
    - `optimized/best_optimized_model.pt`, `experiments/best_experiment_checkpoint.pt`,
      and every other prior-step file are only ever opened for READING.
      This module never writes into `data/`, `checkpoints/`, `experiments/`,
      `evaluation/`, or `optimized/`.
    - No pretrained weights are loaded from anywhere; the only weights
      this module ever reads are this project's own Step 9 output.

ROUND-TRIP VERIFICATION (verify_round_trip())
    Loads the ORIGINAL in-memory model (built straight from
    optimized/best_optimized_model.pt), runs one forward() pass and one
    seeded generate() call, then separately loads the model back purely
    from the freshly-saved model_store/ bundle via `load_model()` and runs
    the identical forward() and generate() calls with the identical
    inputs/seed. Both a state-dict tensor-equality check and an output
    tensor-equality check must pass exactly (this is a plain,
    non-stochastic nn.Module in eval() mode -- forward() has no
    randomness, and generate() is made deterministic by passing the same
    seeded torch.Generator both times -- so "identical" means
    `torch.equal(...) == True`, not "close").
"""

import copy
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.custom_model.config import (
    BEST_OPTIMIZED_MODEL_PATH,
    DATA_DIR,
    DEFAULT_ARCH_CONFIG,
    DEFAULT_PREPROCESSING_CONFIG,
    DEFAULT_PROBLEM_SPEC,
    OPTIMIZATION_REPORT_PATH,
    PACKAGE_ROOT,
    VOCAB_PATH,
)

# Only reads other steps' reports for reproducibility_metadata.json; never
# opens their checkpoints/model files.
BEST_EXPERIMENT_CONFIG_PATH = PACKAGE_ROOT / "experiments" / "best_experiment_config.json"
TRAINING_SUMMARY_PATH = PACKAGE_ROOT / "logs" / "training_summary.json"
EVALUATION_REPORT_PATH = PACKAGE_ROOT / "evaluation" / "evaluation_report.json"

from backend.custom_model.model import TextToImageDiffusionModel

ARTIFACT_VERSION = "v1"
MODEL_STORE_DIR = PACKAGE_ROOT / "model_store"
MODEL_WEIGHTS_PATH = MODEL_STORE_DIR / f"model_{ARTIFACT_VERSION}.pt"
ARCHITECTURE_CONFIG_PATH = MODEL_STORE_DIR / "architecture_config.json"
STORED_VOCAB_PATH = MODEL_STORE_DIR / "vocab.json"
PREPROCESSING_CONFIG_PATH = MODEL_STORE_DIR / "preprocessing_config.json"
REPRODUCIBILITY_METADATA_PATH = MODEL_STORE_DIR / "reproducibility_metadata.json"
MODEL_MANIFEST_PATH = MODEL_STORE_DIR / "model_manifest.json"

VERIFY_SEED = 42
VERIFY_GENERATION_STEPS = 20  # cheap sanity generation, not a full 500-step sample


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Building the model from Step 9's checkpoint (read-only)
# ---------------------------------------------------------------------------
def load_optimized_checkpoint() -> Dict[str, Any]:
    """Reads Step 9's primary optimized model payload, read-only."""
    if not BEST_OPTIMIZED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Step 9 output not found at '{BEST_OPTIMIZED_MODEL_PATH}'. "
            "Run Step 9 (optimize.py) first."
        )
    return torch.load(BEST_OPTIMIZED_MODEL_PATH, map_location="cpu", weights_only=False)


def build_model_from_optimized_checkpoint(
    checkpoint: Dict[str, Any], device: torch.device
) -> Tuple[TextToImageDiffusionModel, int, int]:
    """Reconstructs Step 5's exact architecture (ModelArchitectureConfig is
    never varied by any later step) with the diffusion schedule and
    vocabulary size THIS checkpoint actually used, then loads its weights.

    Step 9 stored the state dict at fp16 precision (see optimize.py's
    `optimize_fp16()`); it is cast back to float32 before loading here --
    the same pattern optimize.py itself uses when it needs a runnable
    fp16-derived model -- since MPS/CPU do not reliably support fast,
    numerically stable fp16 compute for the ops this model uses. If the
    diffusion schedule or vocab size were reconstructed wrong,
    `load_state_dict` would fail outright on shape mismatches (the
    registered `betas`/`alphas_cumprod` buffers, or the text embedding
    table) -- a built-in correctness check, not just an assumption.
    """
    vocab_size = checkpoint["vocab_size"]
    diffusion_timesteps = checkpoint["diffusion_timesteps"]

    import dataclasses
    problem_cfg = dataclasses.replace(DEFAULT_PROBLEM_SPEC, diffusion_timesteps=diffusion_timesteps)

    model = TextToImageDiffusionModel(vocab_size=vocab_size, arch_cfg=DEFAULT_ARCH_CONFIG, problem_cfg=problem_cfg)

    fp32_state_dict = {
        k: v.float() if v.is_floating_point() else v for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(fp32_state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, vocab_size, diffusion_timesteps


# ---------------------------------------------------------------------------
# Step 10a: save the artifact bundle
# ---------------------------------------------------------------------------
def save_model_artifact() -> Dict[str, Any]:
    """Writes the complete model_store/ bundle from Step 9's output and
    every other prior step's reports. Every prior-step file is opened
    read-only; only paths under model_store/ are written."""
    if not BEST_OPTIMIZED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Step 9 output not found at '{BEST_OPTIMIZED_MODEL_PATH}'. Run Step 9 (optimize.py) first."
        )
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Step 3 vocabulary not found at '{VOCAB_PATH}'. Run Step 3 first.")

    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Model weights: an exact byte-for-byte copy, not a re-serialize,
    #        so this file is provably identical to what Step 9 produced. ---
    shutil.copy2(BEST_OPTIMIZED_MODEL_PATH, MODEL_WEIGHTS_PATH)
    source_checksum = sha256_of_file(BEST_OPTIMIZED_MODEL_PATH)
    stored_checksum = sha256_of_file(MODEL_WEIGHTS_PATH)
    if source_checksum != stored_checksum:
        raise RuntimeError("Copied model weights do not match the source checksum -- aborting.")

    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu", weights_only=False)

    # --- 2. Architecture + problem/diffusion config needed to reload ---
    architecture_config = {
        "model_architecture_config": {
            "text_embed_dim": DEFAULT_ARCH_CONFIG.text_embed_dim,
            "num_text_layers": DEFAULT_ARCH_CONFIG.num_text_layers,
            "unet_base_channels": DEFAULT_ARCH_CONFIG.unet_base_channels,
            "channel_multipliers": list(DEFAULT_ARCH_CONFIG.channel_multipliers),
            "num_res_blocks_per_stage": DEFAULT_ARCH_CONFIG.num_res_blocks_per_stage,
            "time_embed_dim": DEFAULT_ARCH_CONFIG.time_embed_dim,
            "dropout": DEFAULT_ARCH_CONFIG.dropout,
            "target_param_limit_millions": DEFAULT_ARCH_CONFIG.target_param_limit_millions,
        },
        "problem_specification": {
            "image_channels": DEFAULT_PROBLEM_SPEC.image_channels,
            "image_height": DEFAULT_PROBLEM_SPEC.image_height,
            "image_width": DEFAULT_PROBLEM_SPEC.image_width,
            "pixel_range": list(DEFAULT_PROBLEM_SPEC.pixel_range),
            "max_text_length": DEFAULT_PROBLEM_SPEC.max_text_length,
            "diffusion_timesteps": checkpoint["diffusion_timesteps"],  # THIS checkpoint's actual schedule
            "beta_start": DEFAULT_PROBLEM_SPEC.beta_start,
            "beta_end": DEFAULT_PROBLEM_SPEC.beta_end,
        },
        "vocab_size": checkpoint["vocab_size"],
        "total_params": checkpoint["total_params"],
        "precision_as_stored": checkpoint["precision"],
        "recommended_generation_steps": checkpoint["recommended_generation_steps"],
        "note": (
            "vocab_size and diffusion_timesteps are read from the checkpoint itself "
            "(the values actually used to produce it), not from config.py's current "
            "defaults, which may drift over time. model_architecture_config is never "
            "varied between Steps 6-9 by design (see config.py's ExperimentConfig "
            "docstring), so it is safe to source from config.py's current defaults."
        ),
    }
    with open(ARCHITECTURE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(architecture_config, f, indent=2)

    # --- 3. Vocabulary (tokenizer) -- copied, not regenerated ---
    shutil.copy2(VOCAB_PATH, STORED_VOCAB_PATH)

    # --- 4. Preprocessing config needed to prepare new inputs identically ---
    preprocessing_config = {
        "target_image_width": DEFAULT_PREPROCESSING_CONFIG.target_image_width,
        "target_image_height": DEFAULT_PREPROCESSING_CONFIG.target_image_height,
        "target_image_channels": DEFAULT_PREPROCESSING_CONFIG.target_image_channels,
        "pixel_range": list(DEFAULT_PREPROCESSING_CONFIG.pixel_range),
        "pixel_normalization": "x_norm = (uint8_pixel / 127.5) - 1.0  (see preprocess.py: DatasetPreprocessor.process_image)",
        "resample_filter": DEFAULT_PREPROCESSING_CONFIG.resample_filter,
        "max_caption_chars": DEFAULT_PREPROCESSING_CONFIG.max_caption_chars,
        "min_caption_chars": DEFAULT_PREPROCESSING_CONFIG.min_caption_chars,
        "max_text_length": DEFAULT_PREPROCESSING_CONFIG.max_text_length,
        "special_tokens": DEFAULT_PREPROCESSING_CONFIG.special_tokens,
        "tokenizer": "Deterministic word-level tokenizer (see preprocess.py: TextTokenizer) -- vocabulary in vocab.json",
    }
    with open(PREPROCESSING_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(preprocessing_config, f, indent=2)

    # --- 5. Reproducibility metadata: condensed Step 6/7/8/9 results ---
    experiment_cfg = _read_json_if_exists(BEST_EXPERIMENT_CONFIG_PATH)
    training_summary = _read_json_if_exists(TRAINING_SUMMARY_PATH)
    evaluation_report = _read_json_if_exists(EVALUATION_REPORT_PATH)
    optimization_report = _read_json_if_exists(OPTIMIZATION_REPORT_PATH)

    reproducibility_metadata = {
        "step6_training": {
            "source_report": str(TRAINING_SUMMARY_PATH),
            "device": training_summary.get("device"),
            "train_samples": training_summary.get("train_samples"),
            "val_samples": training_summary.get("val_samples"),
            "test_split_used": training_summary.get("test_split_used"),
            "base_config": training_summary.get("config"),
            "best_val_loss_step6_run": training_summary.get("best_val_loss"),
        },
        "step7_experiment_selected": {
            "source_report": str(BEST_EXPERIMENT_CONFIG_PATH),
            "name": experiment_cfg.get("name"),
            "description": experiment_cfg.get("description"),
            "varied": experiment_cfg.get("varied"),
            "hyperparameters": experiment_cfg.get("hyperparameters"),
            "best_val_loss": experiment_cfg.get("best_val_loss"),
        },
        "step8_test_evaluation": {
            "source_report": str(EVALUATION_REPORT_PATH),
            "num_test_samples": evaluation_report.get("test_set", {}).get("num_test_samples"),
            "test_mse_matched_protocol": evaluation_report.get("metrics", {})
                .get("matched_validation_protocol", {}).get("mse"),
            "test_mae_matched_protocol": evaluation_report.get("metrics", {})
                .get("matched_validation_protocol", {}).get("mae"),
            "multi_timestep_overall_mean_mse": evaluation_report.get("metrics", {})
                .get("multi_timestep_evaluation", {}).get("overall_mean_mse"),
            "guarantees": evaluation_report.get("guarantees"),
        },
        "step9_optimization": {
            "source_report": str(OPTIMIZATION_REPORT_PATH),
            "techniques_applied": optimization_report.get("optimization_techniques_applied"),
            "test_mse_optimized_fp16": optimization_report.get("metrics_comparison", {})
                .get("accuracy_and_quality", {}).get("test_mse_optimized_fp16"),
            "fp16_mse_degradation_delta": optimization_report.get("metrics_comparison", {})
                .get("accuracy_and_quality", {}).get("fp16_mse_degradation_delta"),
            "degradation_within_acceptable_threshold": optimization_report.get("metrics_comparison", {})
                .get("accuracy_and_quality", {}).get("degradation_within_acceptable_threshold"),
            "generation_speedup_50_steps": optimization_report.get("metrics_comparison", {})
                .get("generation_latency_seconds_per_image", {}).get("speedup_factor_50_steps"),
            "guarantees": optimization_report.get("guarantees"),
            "optimized_at": checkpoint.get("optimized_at"),
        },
        "notes": (
            "Condensed from the full Step 6/7/8/9 reports (paths above) for quick "
            "reference; each source_report has the complete per-sample data."
        ),
    }
    with open(REPRODUCIBILITY_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(reproducibility_metadata, f, indent=2)

    # --- 6. The storage manifest itself ---
    now = datetime.now(timezone.utc).isoformat()
    artifacts = [
        {
            "file": MODEL_WEIGHTS_PATH.name,
            "purpose": "Final model weights (Step 9's optimized FP16 state dict + reload metadata), copied byte-for-byte from optimized/best_optimized_model.pt.",
            "size_bytes": MODEL_WEIGHTS_PATH.stat().st_size,
            "sha256": stored_checksum,
        },
        {
            "file": ARCHITECTURE_CONFIG_PATH.name,
            "purpose": "Model architecture (Step 5 ModelArchitectureConfig) and problem/diffusion specification (Step 1 ProblemSpecification, with this checkpoint's actual diffusion_timesteps/vocab_size) needed to reconstruct the exact nn.Module before loading the weights.",
            "size_bytes": ARCHITECTURE_CONFIG_PATH.stat().st_size,
            "sha256": sha256_of_file(ARCHITECTURE_CONFIG_PATH),
        },
        {
            "file": STORED_VOCAB_PATH.name,
            "purpose": "Tokenizer vocabulary (Step 3 word<->id mapping + special tokens) required to turn new text prompts into token ids this model understands.",
            "size_bytes": STORED_VOCAB_PATH.stat().st_size,
            "sha256": sha256_of_file(STORED_VOCAB_PATH),
        },
        {
            "file": PREPROCESSING_CONFIG_PATH.name,
            "purpose": "Step 3 image/text preprocessing contract (target size, pixel normalization, tokenizer settings) required to preprocess new inputs identically to training and to decode generated output correctly.",
            "size_bytes": PREPROCESSING_CONFIG_PATH.stat().st_size,
            "sha256": sha256_of_file(PREPROCESSING_CONFIG_PATH),
        },
        {
            "file": REPRODUCIBILITY_METADATA_PATH.name,
            "purpose": "Condensed Step 6 (training) / Step 7 (winning experiment) / Step 8 (test evaluation) / Step 9 (optimization) results, for reproducibility without re-reading four separate report files.",
            "size_bytes": REPRODUCIBILITY_METADATA_PATH.stat().st_size,
            "sha256": sha256_of_file(REPRODUCIBILITY_METADATA_PATH),
        },
    ]

    manifest = {
        "step": "Step 10: Model Storage",
        "artifact_version": ARTIFACT_VERSION,
        "created_at": now,
        "framework": f"PyTorch {torch.__version__}",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device_used_to_create_weights": reproducibility_metadata["step6_training"]["device"],
        "device_used_to_run_this_storage_step": str(get_device()),
        "source_checkpoint": {
            "path": str(BEST_OPTIMIZED_MODEL_PATH),
            "sha256": source_checksum,
            "precision": checkpoint["precision"],
            "recommended_generation_steps": checkpoint["recommended_generation_steps"],
            "optimized_at": checkpoint["optimized_at"],
        },
        "checksum_match_source_vs_stored": source_checksum == stored_checksum,
        "artifacts": artifacts,
        "how_to_load": "from backend.custom_model.model_storage import load_model; model, meta = load_model()",
        "guarantees": {
            "no_retraining": True,
            "no_further_tuning": True,
            "no_further_evaluation": True,
            "no_further_optimization": True,
            "no_pretrained_weights_used": True,
            "step7_checkpoint_untouched": True,
            "step6_to_step9_outputs_untouched": True,
            "model_store_is_the_only_thing_written_by_this_step": True,
        },
        "round_trip_verification": None,  # filled in by verify_round_trip() and re-saved
        "notes": (
            "Step 10 storage only -- packaged Step 9's already-finished optimized "
            "model. No training, tuning, evaluation, or optimization was performed "
            "in this step."
        ),
    }
    with open(MODEL_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# Step 10b: load the artifact bundle back
# ---------------------------------------------------------------------------
def load_model(store_dir: Path = MODEL_STORE_DIR, device: torch.device = None) -> Tuple[TextToImageDiffusionModel, Dict[str, Any]]:
    """Reconstructs the exact architecture from `architecture_config.json`
    and loads `model_<version>.pt`'s weights into it. This is the ONLY
    function later code (Step 10+, the live app, a notebook) should use to
    get a ready-to-run copy of this model -- it does not depend on
    `optimize.py`, `experiments/`, or any other prior-step file, only on
    what is inside `model_store/` itself."""
    weights_path = store_dir / f"model_{ARTIFACT_VERSION}.pt"
    arch_path = store_dir / "architecture_config.json"
    if not weights_path.exists() or not arch_path.exists():
        raise FileNotFoundError(f"Model store incomplete at '{store_dir}'. Run save_model_artifact() first.")

    with open(arch_path, "r", encoding="utf-8") as f:
        arch_config = json.load(f)

    import dataclasses
    from backend.custom_model.config import ModelArchitectureConfig, ProblemSpecification

    mac = arch_config["model_architecture_config"]
    arch_cfg = ModelArchitectureConfig(
        text_embed_dim=mac["text_embed_dim"],
        num_text_layers=mac["num_text_layers"],
        unet_base_channels=mac["unet_base_channels"],
        channel_multipliers=tuple(mac["channel_multipliers"]),
        num_res_blocks_per_stage=mac["num_res_blocks_per_stage"],
        time_embed_dim=mac["time_embed_dim"],
        dropout=mac["dropout"],
        target_param_limit_millions=mac["target_param_limit_millions"],
    )
    ps = arch_config["problem_specification"]
    problem_cfg = ProblemSpecification(
        image_channels=ps["image_channels"],
        image_height=ps["image_height"],
        image_width=ps["image_width"],
        pixel_range=tuple(ps["pixel_range"]),
        max_text_length=ps["max_text_length"],
        diffusion_timesteps=ps["diffusion_timesteps"],
        beta_start=ps["beta_start"],
        beta_end=ps["beta_end"],
        vocab_size=arch_config["vocab_size"],
    )

    device = device or get_device()
    model = TextToImageDiffusionModel(vocab_size=arch_config["vocab_size"], arch_cfg=arch_cfg, problem_cfg=problem_cfg)

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    fp32_state_dict = {
        k: v.float() if v.is_floating_point() else v for k, v in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(fp32_state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    meta = {
        "vocab_size": arch_config["vocab_size"],
        "diffusion_timesteps": problem_cfg.diffusion_timesteps,
        "recommended_generation_steps": arch_config["recommended_generation_steps"],
        "precision_as_stored": arch_config["precision_as_stored"],
        "total_params": arch_config["total_params"],
    }
    return model, meta


# ---------------------------------------------------------------------------
# Step 10c: verify a loaded model matches the pre-save model exactly
# ---------------------------------------------------------------------------
def verify_round_trip() -> Dict[str, Any]:
    """Builds the model straight from Step 9's checkpoint (the "before"
    model), separately loads it back purely from the freshly-written
    model_store/ bundle (the "after" model), and checks that they are
    identical: same state dict tensors, same forward() output, same
    seeded generate() output. Returns a JSON-serializable result dict;
    raises AssertionError if anything does not match exactly."""
    device = torch.device("cpu")  # deterministic, reproducible comparison

    before_checkpoint = load_optimized_checkpoint()
    before_model, vocab_size, diffusion_timesteps = build_model_from_optimized_checkpoint(before_checkpoint, device)

    after_model, after_meta = load_model(device=device)

    result: Dict[str, Any] = {"checks": {}}

    # --- 1. State dict tensors identical ---
    before_sd = before_model.state_dict()
    after_sd = after_model.state_dict()
    same_keys = set(before_sd.keys()) == set(after_sd.keys())
    all_tensors_equal = same_keys and all(torch.equal(before_sd[k], after_sd[k]) for k in before_sd)
    result["checks"]["state_dict_keys_match"] = same_keys
    result["checks"]["state_dict_tensors_identical"] = all_tensors_equal
    assert same_keys, "Reloaded model has different parameter names than the pre-save model."
    assert all_tensors_equal, "Reloaded model's weights differ from the pre-save model's weights."

    # --- 2. Deterministic forward() pass identical ---
    torch.manual_seed(VERIFY_SEED)
    dummy_images = torch.randn(2, before_model.problem_cfg.image_channels, before_model.problem_cfg.image_height, before_model.problem_cfg.image_width)
    dummy_timesteps = torch.randint(0, before_model.problem_cfg.diffusion_timesteps, (2,))
    dummy_tokens = torch.randint(0, vocab_size, (2, before_model.problem_cfg.max_text_length))

    with torch.no_grad():
        before_out = before_model(dummy_images, dummy_timesteps, dummy_tokens)
        after_out = after_model(dummy_images, dummy_timesteps, dummy_tokens)
    forward_identical = torch.equal(before_out, after_out)
    result["checks"]["forward_pass_output_identical"] = forward_identical
    result["forward_pass_max_abs_diff"] = float((before_out - after_out).abs().max().item())
    assert forward_identical, "forward() output differs between pre-save and reloaded model."

    # --- 3. Seeded generate() (full text -> image path) identical ---
    tensor_path = CURRENT_DIR / "data" / "processed" / "tensors" / "tensor_00003.pt"
    if tensor_path.exists():
        real_tensor = torch.load(tensor_path, weights_only=False)
        token_ids = real_tensor["token_ids"].unsqueeze(0)
        prompt_used = real_tensor.get("prompt", "(unknown prompt; token_ids only)")
    else:
        token_ids = torch.randint(0, vocab_size, (1, before_model.problem_cfg.max_text_length))
        prompt_used = "(synthetic token_ids -- tensor_00003.pt not found)"

    gen_before = torch.Generator(device="cpu").manual_seed(VERIFY_SEED)
    with torch.no_grad():
        image_before = before_model.generate(token_ids, num_steps=VERIFY_GENERATION_STEPS, generator=gen_before)

    gen_after = torch.Generator(device="cpu").manual_seed(VERIFY_SEED)
    with torch.no_grad():
        image_after = after_model.generate(token_ids, num_steps=VERIFY_GENERATION_STEPS, generator=gen_after)

    generate_identical = torch.equal(image_before, image_after)
    result["checks"]["generate_output_identical"] = generate_identical
    result["generate_max_abs_diff"] = float((image_before - image_after).abs().max().item())
    result["generate_prompt_used"] = prompt_used
    result["generate_output_shape"] = list(image_before.shape)
    result["generate_num_steps"] = VERIFY_GENERATION_STEPS
    assert generate_identical, "generate() output differs between pre-save and reloaded model."

    result["all_checks_passed"] = all(result["checks"].values())
    result["verified_at"] = datetime.now(timezone.utc).isoformat()
    result["verification_device"] = str(device)
    result["verification_torch_version"] = torch.__version__
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_step10_model_storage() -> Dict[str, Any]:
    print("=" * 70)
    print("  AI TEXT-TO-IMAGE GENERATOR: CUSTOM MODEL FROM SCRATCH")
    print("  STEP 10: MODEL STORAGE")
    print("=" * 70)

    print(f"\n[1/3] Saving model artifact bundle to: {MODEL_STORE_DIR}")
    manifest = save_model_artifact()
    print(f"      Weights:        {MODEL_WEIGHTS_PATH.name}  ({MODEL_WEIGHTS_PATH.stat().st_size / (1024*1024):.2f} MB)")
    print(f"      Checksum match (source vs. stored copy): {manifest['checksum_match_source_vs_stored']}")
    for a in manifest["artifacts"]:
        print(f"      - {a['file']:<32} {a['size_bytes']:>10,} bytes  sha256={a['sha256'][:12]}...")

    print(f"\n[2/3] Loading the model back purely from {MODEL_STORE_DIR}...")
    model, meta = load_model()
    print(f"      Reloaded OK. vocab_size={meta['vocab_size']}, diffusion_timesteps={meta['diffusion_timesteps']}, total_params={meta['total_params']:,}")

    print(f"\n[3/3] Verifying reloaded model == pre-save model (state dict, forward(), generate())...")
    verification = verify_round_trip()
    for check, passed in verification["checks"].items():
        print(f"      - {check:<32} {'PASS' if passed else 'FAIL'}")
    print(f"      forward() max abs diff: {verification['forward_pass_max_abs_diff']}")
    print(f"      generate() max abs diff: {verification['generate_max_abs_diff']}  (prompt: {verification['generate_prompt_used']!r})")

    # Re-save the manifest with the verification result embedded.
    manifest["round_trip_verification"] = verification
    with open(MODEL_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("  STEP 10 SUMMARY:")
    print(f"  - Artifact version:        {manifest['artifact_version']}")
    print(f"  - Created at:              {manifest['created_at']}")
    print(f"  - Framework:               {manifest['framework']}")
    print(f"  - Model store directory:   {MODEL_STORE_DIR}")
    print(f"  - Manifest:                {MODEL_MANIFEST_PATH}")
    print(f"  - Round-trip verification: {'ALL CHECKS PASSED' if verification['all_checks_passed'] else 'FAILED -- see above'}")
    print("=" * 70)
    print("Step 10 (Model Storage) complete. No training, tuning, evaluation,")
    print("or further optimization was performed.")
    print("=" * 70)

    return manifest


if __name__ == "__main__":
    run_step10_model_storage()
