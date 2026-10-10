"""
train.py
--------
STEP 6: TRAINING PIPELINE

Trains the Step 5 `TextToImageDiffusionModel` as a conditional DDPM on the
Step 4 TRAIN split only, using the Step 4 VALIDATION split solely to track
generalization and pick the best checkpoint. The Step 4 TEST split is never
imported, referenced, or read anywhere in this module -- by construction,
not just by convention, so it cannot leak into training or model selection.

WHAT "TRAINING" MEANS HERE (DDPM noise-prediction objective)
    For each real image x_0 in a batch:
      1. Sample a random diffusion timestep t ~ Uniform{0, ..., T-1}.
      2. Sample Gaussian noise  epsilon ~ N(0, I),  same shape as x_0.
      3. Forward diffusion (fixed, not learned): produce the noised image
             x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * epsilon
         using the beta/alpha schedule already registered on the Step 5
         model (`model.alphas_cumprod`).
      4. Ask the network to predict the noise that was added:
             epsilon_hat = model(x_t, t, token_ids)
      5. Loss = MSE(epsilon_hat, epsilon)  (Step 1's objective, verbatim).
      6. Backprop + gradient clipping + an AdamW step update the network's
         weights to make that prediction better next time.
    Steps 1-3 are pure math on the fixed schedule -- nothing there is
    learned. Only step 4 (the network) has trainable parameters, and it is
    the SAME `TextToImageDiffusionModel.forward` Step 5 already defined and
    verified; this file does not modify the architecture.

LEARNING RATE
    AdamW with a linear-warmup -> cosine-decay schedule (Step 1's
    `TrainingConfig.lr_scheduler = "cosine_with_warmup"`), stepped once per
    optimizer step (not per epoch). Step 1's placeholder `warmup_steps=500`
    was sized before Step 2-4 fixed the dataset at 399 training samples
    (~13 steps/epoch); taken literally it would consume most or all of a
    run in pure warmup with no cosine decay left. So the warmup actually
    used is `min(warmup_steps, total_steps // 5)` -- at most a fifth of the
    run -- and this adaptation is printed and recorded in the training log,
    the same way Step 5 grounded Step 1's placeholder `vocab_size` in the
    real, measured value instead of silently trusting the placeholder.

CHECKPOINTING & RESUME
    Every epoch: `checkpoints/checkpoint_latest.pt` is overwritten (full
    resume state: model, optimizer, scheduler, epoch, best_val_loss, RNG
    state), and `checkpoints/checkpoint_best.pt` is overwritten whenever
    validation loss improves. Every `checkpoint_every_n_epochs` epochs, a
    numbered snapshot `checkpoints/checkpoint_epoch_XXXX.pt` is also kept.
    `--resume` loads `checkpoint_latest.pt` and continues from the next
    epoch with the optimizer/scheduler/RNG state intact, rather than
    restarting.

LOGGING
    Each epoch appends one JSON line to `logs/training_log.jsonl` (epoch,
    train_loss, val_loss, learning_rate, elapsed_seconds, is_best) and
    prints the same to stdout. A final `logs/training_summary.json` is
    written when `fit()` completes.

NO PRETRAINED WEIGHTS
    The model is constructed fresh via `TextToImageDiffusionModel(...)`
    (Step 5's random-init constructor) every run; the only state ever
    loaded from disk is this pipeline's OWN prior checkpoints via
    `--resume`, never any external/pretrained checkpoint.
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.custom_model.config import (
    BEST_CHECKPOINT_PATH,
    CHECKPOINTS_DIR,
    DEFAULT_TRAIN_CONFIG,
    LATEST_CHECKPOINT_PATH,
    LOGS_DIR,
    PROCESSED_TENSORS_DIR,
    TRAIN_METADATA_PATH,
    TRAINING_LOG_PATH,
    TRAINING_SUMMARY_PATH,
    VAL_METADATA_PATH,
    DEFAULT_PROBLEM_SPEC,
    ProblemSpecification,
    TrainingConfig,
)
from backend.custom_model.model import TextToImageDiffusionModel, load_vocab_size

# NOTE: TEST_METADATA_PATH is deliberately NOT imported anywhere in this
# module. Training and validation must never see the test split.


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------
class TextToImageDataset(Dataset):
    """Reads one Step 4 split metadata file and loads each sample's
    pre-computed Step 3 tensor (image [3,64,64] float32, token_ids [32]
    int64). No re-decoding, re-resizing, or re-tokenizing happens here --
    that was all done once in Step 3."""

    def __init__(self, split_metadata_path: Path):
        if not split_metadata_path.exists():
            raise FileNotFoundError(
                f"Split metadata not found at '{split_metadata_path}'. "
                "Ensure Step 4: Dataset Splitting has been run first."
            )
        self.records = [
            json.loads(line) for line in open(split_metadata_path, "r", encoding="utf-8") if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        tensor_data = torch.load(PROCESSED_TENSORS_DIR / rec["tensor_file"], weights_only=False)
        return {
            "id": rec["id"],
            "image": tensor_data["image"],
            "token_ids": tensor_data["token_ids"],
        }


def create_dataloaders(cfg: TrainingConfig) -> Tuple[DataLoader, DataLoader, int, int]:
    """Train loader (TRAIN split only, shuffled) + val loader (VALIDATION
    split only, not shuffled). Never touches the test split."""
    train_dataset = TextToImageDataset(TRAIN_METADATA_PATH)
    val_dataset = TextToImageDataset(VAL_METADATA_PATH)

    gen = torch.Generator()
    gen.manual_seed(cfg.random_seed)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0, generator=gen, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0, drop_last=False,
    )
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


# ---------------------------------------------------------------------------
# Forward diffusion
# ---------------------------------------------------------------------------
def q_sample(model: TextToImageDiffusionModel, x0: torch.Tensor, t: torch.LongTensor, noise: torch.Tensor) -> torch.Tensor:
    """Forward diffusion: x_t = sqrt(alpha_cumprod_t) * x0 + sqrt(1 - alpha_cumprod_t) * noise.
    Fixed math on the schedule registered on the Step 5 model -- not learned."""
    alphas_cumprod_t = model.alphas_cumprod[t]                      # [B]
    sqrt_ac = alphas_cumprod_t.sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_ac = (1.0 - alphas_cumprod_t).sqrt().view(-1, 1, 1, 1)
    return sqrt_ac * x0 + sqrt_one_minus_ac * noise


# ---------------------------------------------------------------------------
# LR schedule: linear warmup -> cosine decay
# ---------------------------------------------------------------------------
def build_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        cfg: TrainingConfig = DEFAULT_TRAIN_CONFIG,
        problem_cfg: ProblemSpecification = DEFAULT_PROBLEM_SPEC,
        num_epochs: Optional[int] = None,
        log_every_n_steps: int = 5,
        checkpoint_dir: Optional[Path] = None,
        latest_checkpoint_path: Optional[Path] = None,
        best_checkpoint_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
        summary_path: Optional[Path] = None,
        verbose: bool = True,
    ):
        """`problem_cfg` and the five path overrides exist so Step 7 can run
        many independent, isolated experiments (different diffusion
        schedules, separate checkpoint/log locations per experiment)
        through this SAME Trainer/training loop, without touching Step 6's
        own checkpoints/logs. Calling Trainer() with no arguments (Step 6's
        own CLI usage) is unaffected -- every override defaults to Step 6's
        original fixed paths and the original ProblemSpecification."""
        self.cfg = cfg
        self.num_epochs = num_epochs or cfg.total_epochs
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose
        self.device = get_device()

        self.checkpoint_dir = checkpoint_dir or CHECKPOINTS_DIR
        self.latest_checkpoint_path = latest_checkpoint_path or LATEST_CHECKPOINT_PATH
        self.best_checkpoint_path = best_checkpoint_path or BEST_CHECKPOINT_PATH
        self.log_path = log_path or TRAINING_LOG_PATH
        self.summary_path = summary_path or TRAINING_SUMMARY_PATH

        set_seed(cfg.random_seed)

        self.train_loader, self.val_loader, self.n_train, self.n_val = create_dataloaders(cfg)
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.steps_per_epoch * self.num_epochs

        vocab_size = load_vocab_size(default=DEFAULT_PROBLEM_SPEC.vocab_size)
        self.model = TextToImageDiffusionModel(vocab_size=vocab_size, problem_cfg=problem_cfg).to(self.device)

        optimizer_cls = getattr(torch.optim, cfg.optimizer)
        self.optimizer = optimizer_cls(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        self.effective_warmup_steps = min(cfg.warmup_steps, max(1, self.total_steps // 5))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=build_lr_lambda(self.effective_warmup_steps, self.total_steps)
        )

        # Mixed precision: fp16 only on CUDA (per Step 1's HardwareBudgetConfig
        # comment); MPS/CPU always train in fp32.
        self.use_amp = cfg.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Fixed validation RNG: every epoch's validation uses the SAME
        # (timestep, noise) draws per sample, so val_loss changes only
        # because the model got better/worse -- not because a different
        # random noise realization happened to be easier or harder that
        # epoch. This makes "best checkpoint by val_loss" a meaningful,
        # low-variance signal even with only 50 validation samples.
        self.val_generator = torch.Generator(device="cpu")
        self.val_generator.manual_seed(cfg.random_seed)

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)

    # -- one training epoch -------------------------------------------------
    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_n = 0
        T = self.model.problem_cfg.diffusion_timesteps

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            token_ids = batch["token_ids"].to(self.device)
            B = images.shape[0]

            t = torch.randint(0, T, (B,), device=self.device)
            noise = torch.randn_like(images)
            x_t = q_sample(self.model, images, t, noise)

            self.optimizer.zero_grad(set_to_none=True)
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    predicted_noise = self.model(x_t, t, token_ids)
                    loss = F.mse_loss(predicted_noise, noise)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predicted_noise = self.model(x_t, t, token_ids)
                loss = F.mse_loss(predicted_noise, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.gradient_clip_val)
                self.optimizer.step()

            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item() * B
            total_n += B

            if self.verbose and (step % self.log_every_n_steps == 0 or step == self.steps_per_epoch - 1):
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"   epoch {epoch + 1:>3}/{self.num_epochs}  "
                    f"step {step + 1:>3}/{self.steps_per_epoch}  "
                    f"loss {loss.item():.4f}  lr {lr:.2e}"
                )

        return total_loss / max(1, total_n)

    # -- validation (VALIDATION split only) ---------------------------------
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_n = 0
        T = self.model.problem_cfg.diffusion_timesteps

        # Reset to the same seed every call so every epoch is scored on
        # identical (t, noise) draws -- see the note in __init__.
        self.val_generator.manual_seed(self.cfg.random_seed)

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            token_ids = batch["token_ids"].to(self.device)
            B = images.shape[0]

            t = torch.randint(0, T, (B,), generator=self.val_generator).to(self.device)
            noise = torch.randn(images.shape, generator=self.val_generator).to(self.device)
            x_t = q_sample(self.model, images, t, noise)

            predicted_noise = self.model(x_t, t, token_ids)
            loss = F.mse_loss(predicted_noise, noise)

            total_loss += loss.item() * B
            total_n += B

        return total_loss / max(1, total_n)

    # -- checkpointing --------------------------------------------------
    def _checkpoint_payload(self, epoch: int, train_loss: float, val_loss: float) -> Dict[str, Any]:
        return {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "torch_rng_state": torch.get_rng_state(),
            "config": {
                "batch_size": self.cfg.batch_size,
                "learning_rate": self.cfg.learning_rate,
                "weight_decay": self.cfg.weight_decay,
                "optimizer": self.cfg.optimizer,
                "lr_scheduler": self.cfg.lr_scheduler,
                "warmup_steps_configured": self.cfg.warmup_steps,
                "warmup_steps_effective": self.effective_warmup_steps,
                "total_epochs": self.num_epochs,
                "gradient_clip_val": self.cfg.gradient_clip_val,
                "random_seed": self.cfg.random_seed,
            },
        }

    def save_checkpoint(self, epoch: int, train_loss: float, val_loss: float, is_best: bool) -> None:
        payload = self._checkpoint_payload(epoch, train_loss, val_loss)

        torch.save(payload, self.latest_checkpoint_path)

        if is_best:
            torch.save(payload, self.best_checkpoint_path)

        if (epoch + 1) % self.cfg.checkpoint_every_n_epochs == 0:
            numbered_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            torch.save(payload, numbered_path)

    def load_checkpoint(self, path: Path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        self.global_step = payload["global_step"]
        self.best_val_loss = payload["best_val_loss"]
        self.start_epoch = payload["epoch"] + 1
        torch.set_rng_state(payload["torch_rng_state"].to(torch.uint8).cpu())
        print(f"Resumed from '{path}': starting at epoch {self.start_epoch + 1}, best_val_loss={self.best_val_loss:.4f}")

    # -- full training loop --------------------------------------------
    def fit(self, resume: bool = False) -> Dict[str, Any]:
        if self.verbose:
            print("=" * 70)
            print("  AI TEXT-TO-IMAGE GENERATOR: CUSTOM MODEL FROM SCRATCH")
            print("  STEP 6: TRAINING PIPELINE")
            print("=" * 70)
            print(f"\nDevice: {self.device}")
            print(f"Train samples: {self.n_train}  |  Validation samples: {self.n_val}  (TEST split never loaded)")
            print(f"Batch size: {self.cfg.batch_size}  |  Steps/epoch: {self.steps_per_epoch}  |  Epochs: {self.num_epochs}")
            print(f"Optimizer: {self.cfg.optimizer}  lr={self.cfg.learning_rate}  weight_decay={self.cfg.weight_decay}")
            print(
                f"LR schedule: {self.cfg.lr_scheduler}  "
                f"(warmup_steps: configured={self.cfg.warmup_steps}, effective={self.effective_warmup_steps} "
                f"of {self.total_steps} total steps)"
            )
            print(f"Mixed precision (fp16): {self.use_amp}  |  Gradient clip: {self.cfg.gradient_clip_val}")
            print(f"Random seed: {self.cfg.random_seed}")

        if resume and self.latest_checkpoint_path.exists():
            self.load_checkpoint(self.latest_checkpoint_path)
        elif resume:
            print(f"--resume requested but no checkpoint found at '{self.latest_checkpoint_path}'; starting fresh.")

        history = []
        run_start = time.time()

        for epoch in range(self.start_epoch, self.num_epochs):
            epoch_start = time.time()
            if self.verbose:
                print(f"\n--- Epoch {epoch + 1}/{self.num_epochs} ---")

            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            elapsed = round(time.time() - epoch_start, 2)
            lr = self.scheduler.get_last_lr()[0]

            self.save_checkpoint(epoch, train_loss, val_loss, is_best)

            log_entry = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "best_val_loss": round(self.best_val_loss, 6),
                "is_best": is_best,
                "learning_rate": lr,
                "elapsed_seconds": elapsed,
                "global_step": self.global_step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            history.append(log_entry)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.num_epochs} done in {elapsed}s -- "
                    f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"best_val_loss={self.best_val_loss:.4f}  lr={lr:.2e}"
                    + ("  [BEST]" if is_best else "")
                )

        total_elapsed = round(time.time() - run_start, 2)

        summary = {
            "step": "Step 6: Training Pipeline",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "device": str(self.device),
            "epochs_run": self.num_epochs - self.start_epoch,
            "final_epoch": self.num_epochs,
            "train_samples": self.n_train,
            "val_samples": self.n_val,
            "test_split_used": False,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": history[-1]["train_loss"] if history else None,
            "final_val_loss": history[-1]["val_loss"] if history else None,
            "total_training_seconds": total_elapsed,
            "checkpoints": {
                "latest": str(self.latest_checkpoint_path),
                "best": str(self.best_checkpoint_path),
            },
            "config": {
                "batch_size": self.cfg.batch_size,
                "learning_rate": self.cfg.learning_rate,
                "weight_decay": self.cfg.weight_decay,
                "optimizer": self.cfg.optimizer,
                "lr_scheduler": self.cfg.lr_scheduler,
                "warmup_steps_configured": self.cfg.warmup_steps,
                "warmup_steps_effective": self.effective_warmup_steps,
                "checkpoint_every_n_epochs": self.cfg.checkpoint_every_n_epochs,
                "random_seed": self.cfg.random_seed,
                "gradient_clip_val": self.cfg.gradient_clip_val,
                "mixed_precision_used": self.use_amp,
            },
            "history": history,
            "notes": (
                "Step 6 training only -- trained on the TRAIN split, validated on the "
                "VALIDATION split, and never loaded the TEST split. No pretrained "
                "weights were used. Steps 7-9 (tuning, final evaluation, optimization) "
                "have not been run."
            ),
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if self.verbose:
            print("\n" + "=" * 70)
            print("  STEP 6 SUMMARY:")
            print(f"  - Epochs run this session: {summary['epochs_run']}")
            print(f"  - Best validation loss:    {self.best_val_loss:.4f}")
            print(f"  - Total training time:     {total_elapsed}s")
            print(f"  - Latest checkpoint:       {self.latest_checkpoint_path}")
            print(f"  - Best checkpoint:         {self.best_checkpoint_path}")
            print(f"  - Training log:            {self.log_path}")
            print(f"  - Summary report:          {self.summary_path}")
            print("=" * 70)
            print("Step 6 (Training Pipeline) complete. Test split was never")
            print("used. Steps 7-9 (tuning, evaluation, optimization) not run.")
            print("=" * 70)

        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: Training Pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Override TrainingConfig.total_epochs")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/checkpoint_latest.pt")
    args = parser.parse_args()

    trainer = Trainer(num_epochs=args.epochs)
    trainer.fit(resume=args.resume)


if __name__ == "__main__":
    main()
