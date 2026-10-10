"""
experiment.py
-------------
STEP 7: EXPERIMENTATION & HYPERPARAMETER TUNING

Runs a small, curated set of controlled hyperparameter experiments through
the EXISTING Step 6 training pipeline (`train.Trainer`), compares them by
validation loss, and keeps the best one. This module does not define any
new training logic, model code, or dataset handling -- it only configures
and orchestrates repeated, isolated calls into Step 6's `Trainer`.

WHAT "CONTROLLED" MEANS HERE
    Each experiment changes exactly the hyperparameter(s) named in its
    `varied` list and leaves everything else at the Step 6 baseline
    (`DEFAULT_TRAIN_CONFIG` / `DEFAULT_PROBLEM_SPEC`) -- a one-factor-at-a-
    time design, not a combinatorial grid. That keeps the comparison
    readable (each result answers "what did changing just this do?") and
    keeps the whole suite laptop-sized: 8 short experiments instead of a
    multi-way grid.

WHAT IS HELD FIXED ACROSS EVERY EXPERIMENT (for a fair comparison)
    - The train/validation split files (Step 4's 399/50 split) -- untouched.
    - `random_seed=42` -- identical data shuffling and weight
      initialization in every experiment, so a difference in outcome
      reflects the hyperparameter under test, not RNG luck.
    - `ModelArchitectureConfig` -- never overridden. Step 7 tunes how the
      Step 5 network is trained, never what it is.
    - Epoch count per experiment (default 3) -- short but identical, so
      runtime/loss are comparable across experiments.

WHAT IS VARIED (one controlled experiment group per hyperparameter)
    - learning_rate     (lr_low, lr_high)
    - batch_size        (batch_small, batch_large)
    - weight_decay      (weight_decay_zero, weight_decay_high)
    - diffusion_timesteps -- a training/schedule setting, not an
      architecture change: it resizes the fixed forward-diffusion beta
      schedule buffers, not any learned layer (diffusion_short)
    plus one `baseline` experiment (Step 6's own defaults, unmodified) that
    every other experiment is compared against.

ISOLATION FROM CLEAN CHECKPOINTS
    Every experiment builds a brand-new `Trainer`, which constructs a
    brand-new `TextToImageDiffusionModel` from Step 5's random-init
    constructor (`resume=False`, always) -- no experiment ever starts from
    another experiment's, or Step 6's own, weights. Each experiment also
    writes to its own isolated directory
    (`experiments/<name>/checkpoint_*.pt`, `training_log.jsonl`,
    `training_summary.json`) so Step 6's own `checkpoints/`/`logs/` files
    are never read or overwritten by this step.

TEST SPLIT
    `train.py`'s `Trainer`/`create_dataloaders` never import or reference
    the test split (see train.py's own module docstring); this module adds
    no new data loading of its own, so that guarantee is inherited
    unchanged. Model/hyperparameter SELECTION here is by validation loss
    only.

OUTPUT
    `experiments/experiment_report.json` -- every experiment's
    hyperparameters, metrics (final/best train & val loss), and runtime,
    ranked by best validation loss, with the winner marked.
    `experiments/best_experiment_checkpoint.pt` -- a copy of the winning
    experiment's best checkpoint.
    `experiments/best_experiment_config.json` -- the winning
    hyperparameter configuration alone, for easy reuse.
"""

import argparse
import dataclasses
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.custom_model.config import (
    BEST_EXPERIMENT_CHECKPOINT_PATH,
    BEST_EXPERIMENT_CONFIG_PATH,
    DEFAULT_PROBLEM_SPEC,
    DEFAULT_TRAIN_CONFIG,
    EXPERIMENT_REPORT_PATH,
    EXPERIMENTS_DIR,
    ExperimentConfig,
    ProblemSpecification,
    TrainingConfig,
)
from backend.custom_model.train import Trainer

DEFAULT_EXPERIMENT_EPOCHS = 3  # small on purpose -- a comparison signal, not a full training run


def build_experiment_configs(epochs: int = DEFAULT_EXPERIMENT_EPOCHS) -> List[ExperimentConfig]:
    """The curated, one-factor-at-a-time controlled experiment suite."""
    base = DEFAULT_TRAIN_CONFIG
    base_T = DEFAULT_PROBLEM_SPEC.diffusion_timesteps

    return [
        ExperimentConfig(
            name="baseline",
            description=f"Step 6 defaults, unmodified (lr={base.learning_rate}, batch={base.batch_size}, "
                        f"weight_decay={base.weight_decay}, diffusion_timesteps={base_T})",
            varied=[],
            epochs=epochs,
        ),
        ExperimentConfig(
            name="lr_low",
            description=f"Lower learning rate ({base.learning_rate} -> 5e-05)",
            varied=["learning_rate"],
            epochs=epochs,
            learning_rate=5e-5,
        ),
        ExperimentConfig(
            name="lr_high",
            description=f"Higher learning rate ({base.learning_rate} -> 6e-04)",
            varied=["learning_rate"],
            epochs=epochs,
            learning_rate=6e-4,
        ),
        ExperimentConfig(
            name="batch_small",
            description=f"Smaller batch size ({base.batch_size} -> 16)",
            varied=["batch_size"],
            epochs=epochs,
            batch_size=16,
        ),
        ExperimentConfig(
            name="batch_large",
            description=f"Larger batch size ({base.batch_size} -> 64)",
            varied=["batch_size"],
            epochs=epochs,
            batch_size=64,
        ),
        ExperimentConfig(
            name="weight_decay_zero",
            description=f"No weight decay ({base.weight_decay} -> 0.0)",
            varied=["weight_decay"],
            epochs=epochs,
            weight_decay=0.0,
        ),
        ExperimentConfig(
            name="weight_decay_high",
            description=f"Stronger weight decay ({base.weight_decay} -> 1e-03)",
            varied=["weight_decay"],
            epochs=epochs,
            weight_decay=1e-3,
        ),
        ExperimentConfig(
            name="diffusion_short",
            description=f"Shorter/coarser diffusion schedule ({base_T} -> 100 timesteps)",
            varied=["diffusion_timesteps"],
            epochs=epochs,
            diffusion_timesteps=100,
        ),
    ]


def apply_overrides(exp: ExperimentConfig) -> Tuple[TrainingConfig, ProblemSpecification]:
    """Builds this experiment's TrainingConfig/ProblemSpecification by
    replacing only the fields it names, off of the shared Step 6 defaults
    -- every unset field (including random_seed) stays identical to every
    other experiment."""
    train_overrides = {
        k: v
        for k, v in {
            "learning_rate": exp.learning_rate,
            "batch_size": exp.batch_size,
            "weight_decay": exp.weight_decay,
            "gradient_clip_val": exp.gradient_clip_val,
        }.items()
        if v is not None
    }
    train_cfg = dataclasses.replace(DEFAULT_TRAIN_CONFIG, **train_overrides) if train_overrides else DEFAULT_TRAIN_CONFIG

    problem_cfg = DEFAULT_PROBLEM_SPEC
    if exp.diffusion_timesteps is not None:
        problem_cfg = dataclasses.replace(DEFAULT_PROBLEM_SPEC, diffusion_timesteps=exp.diffusion_timesteps)

    return train_cfg, problem_cfg


class ExperimentRunner:
    def __init__(self, experiments: List[ExperimentConfig]):
        self.experiments = experiments
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    def run_one(self, exp: ExperimentConfig) -> Dict[str, Any]:
        train_cfg, problem_cfg = apply_overrides(exp)
        exp_dir = EXPERIMENTS_DIR / exp.name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"  EXPERIMENT: {exp.name}  --  {exp.description}")
        print(f"{'=' * 70}")
        print(f"  varied: {exp.varied if exp.varied else '(none -- baseline)'}")
        print(
            f"  lr={train_cfg.learning_rate}  batch_size={train_cfg.batch_size}  "
            f"weight_decay={train_cfg.weight_decay}  diffusion_timesteps={problem_cfg.diffusion_timesteps}  "
            f"epochs={exp.epochs}  seed={train_cfg.random_seed}"
        )

        # Fresh Trainer -> fresh Step 5 random-init model every time (never
        # resumed from another experiment or from Step 6's own checkpoint),
        # writing into this experiment's own isolated directory.
        trainer = Trainer(
            cfg=train_cfg,
            problem_cfg=problem_cfg,
            num_epochs=exp.epochs,
            checkpoint_dir=exp_dir,
            latest_checkpoint_path=exp_dir / "checkpoint_latest.pt",
            best_checkpoint_path=exp_dir / "checkpoint_best.pt",
            log_path=exp_dir / "training_log.jsonl",
            summary_path=exp_dir / "training_summary.json",
            verbose=False,
        )

        start = time.time()
        summary = trainer.fit(resume=False)
        runtime = round(time.time() - start, 2)

        result = {
            "name": exp.name,
            "description": exp.description,
            "varied": exp.varied,
            "hyperparameters": {
                "learning_rate": train_cfg.learning_rate,
                "batch_size": train_cfg.batch_size,
                "weight_decay": train_cfg.weight_decay,
                "gradient_clip_val": train_cfg.gradient_clip_val,
                "diffusion_timesteps": problem_cfg.diffusion_timesteps,
                "epochs": exp.epochs,
                "random_seed": train_cfg.random_seed,
            },
            "train_samples": summary["train_samples"],
            "val_samples": summary["val_samples"],
            "test_split_used": summary["test_split_used"],
            "final_train_loss": summary["final_train_loss"],
            "final_val_loss": summary["final_val_loss"],
            "best_val_loss": summary["best_val_loss"],
            "runtime_seconds": runtime,
            "per_epoch_history": summary["history"],
            "checkpoint_best": str(exp_dir / "checkpoint_best.pt"),
            "checkpoint_latest": str(exp_dir / "checkpoint_latest.pt"),
        }
        print(f"  -> best_val_loss={result['best_val_loss']:.4f}  final_train_loss={result['final_train_loss']:.4f}  runtime={runtime}s")
        return result

    def run_all(self) -> Dict[str, Any]:
        print("=" * 70)
        print("  AI TEXT-TO-IMAGE GENERATOR: CUSTOM MODEL FROM SCRATCH")
        print("  STEP 7: EXPERIMENTATION & HYPERPARAMETER TUNING")
        print("=" * 70)
        print(f"\nRunning {len(self.experiments)} controlled experiments (train split only, validation for selection):")
        for e in self.experiments:
            print(f"   - {e.name:<20} {e.description}")

        suite_start = time.time()
        results = [self.run_one(exp) for exp in self.experiments]
        suite_elapsed = round(time.time() - suite_start, 2)

        ranked = sorted(results, key=lambda r: r["best_val_loss"])
        for rank, r in enumerate(ranked, start=1):
            r["rank_by_val_loss"] = rank
        winner = ranked[0]

        # Save the winning configuration + a standalone copy of its best checkpoint.
        shutil.copy2(winner["checkpoint_best"], BEST_EXPERIMENT_CHECKPOINT_PATH)
        with open(BEST_EXPERIMENT_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": winner["name"],
                    "description": winner["description"],
                    "varied": winner["varied"],
                    "hyperparameters": winner["hyperparameters"],
                    "best_val_loss": winner["best_val_loss"],
                    "source_checkpoint": winner["checkpoint_best"],
                },
                f,
                indent=2,
            )

        report = {
            "step": "Step 7: Experimentation & Hyperparameter Tuning",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "num_experiments": len(results),
            "total_suite_runtime_seconds": suite_elapsed,
            "baseline_config": {
                "learning_rate": DEFAULT_TRAIN_CONFIG.learning_rate,
                "batch_size": DEFAULT_TRAIN_CONFIG.batch_size,
                "weight_decay": DEFAULT_TRAIN_CONFIG.weight_decay,
                "diffusion_timesteps": DEFAULT_PROBLEM_SPEC.diffusion_timesteps,
                "random_seed": DEFAULT_TRAIN_CONFIG.random_seed,
            },
            "results_ranked_by_val_loss": ranked,
            "winner": {
                "name": winner["name"],
                "description": winner["description"],
                "varied": winner["varied"],
                "hyperparameters": winner["hyperparameters"],
                "best_val_loss": winner["best_val_loss"],
                "saved_checkpoint": str(BEST_EXPERIMENT_CHECKPOINT_PATH),
                "saved_config": str(BEST_EXPERIMENT_CONFIG_PATH),
            },
            "notes": (
                "Step 7 tuning only -- every experiment trained on the TRAIN split, "
                "selected by VALIDATION loss, and never touched the TEST split. The "
                "dataset (Steps 2-4) and model architecture (Step 5) were not modified "
                "by this step; only training/schedule hyperparameters were varied. No "
                "pretrained weights were used -- every experiment started from a fresh, "
                "randomly initialized model. Step 8 (final test-set evaluation) and "
                "Step 9 (optimization) have not been run."
            ),
        }
        with open(EXPERIMENT_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("\n" + "=" * 70)
        print("  STEP 7 SUMMARY (ranked by validation loss):")
        for r in ranked:
            marker = "  <== BEST" if r["name"] == winner["name"] else ""
            print(f"  {r['rank_by_val_loss']}. {r['name']:<20} val_loss={r['best_val_loss']:.4f}  ({r['runtime_seconds']}s){marker}")
        print(f"\n  Winner: {winner['name']} -- {winner['description']}")
        print(f"  Total suite runtime: {suite_elapsed}s")
        print(f"  Report:            {EXPERIMENT_REPORT_PATH}")
        print(f"  Best checkpoint:   {BEST_EXPERIMENT_CHECKPOINT_PATH}")
        print(f"  Best config:       {BEST_EXPERIMENT_CONFIG_PATH}")
        print("=" * 70)
        print("Step 7 (Experimentation & Hyperparameter Tuning) complete.")
        print("Test split was never used. Steps 8-9 not run.")
        print("=" * 70)

        return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7: Experimentation & Hyperparameter Tuning")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EXPERIMENT_EPOCHS, help="Epochs per experiment (kept small)")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated experiment names to run (default: all)")
    args = parser.parse_args()

    configs = build_experiment_configs(epochs=args.epochs)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",")}
        configs = [c for c in configs if c.name in wanted]
        missing = wanted - {c.name for c in configs}
        if missing:
            raise ValueError(f"Unknown experiment name(s): {sorted(missing)}")

    runner = ExperimentRunner(configs)
    runner.run_all()


if __name__ == "__main__":
    main()
