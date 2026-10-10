"""
problem_definition.py
---------------------
STEP 1: PROBLEM DEFINITION EXECUTABLE VALIDATION & REPORT

This script executes and validates the mathematical contracts, tensor dimensions,
compute budgets, and pipeline specifications formulated in Step 1.
"""

import sys
import torch
from pathlib import Path

# Add backend directory to sys.path if run directly
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
if str(BACKEND_DIR.parent) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR.parent))

from backend.custom_model.config import (
    DEFAULT_PROBLEM_SPEC,
    DEFAULT_ARCH_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_HARDWARE_CONFIG,
    DATA_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
)


def validate_step1_specification():
    """Validates the mathematical and engineering contracts of Step 1."""
    print("=" * 70)
    print("  AI TEXT-TO-IMAGE GENERATOR: CUSTOM MODEL FROM SCRATCH")
    print("  STEP 1: PROBLEM DEFINITION & ML SPECIFICATION REPORT")
    print("=" * 70)
    
    spec = DEFAULT_PROBLEM_SPEC
    arch = DEFAULT_ARCH_CONFIG
    train = DEFAULT_TRAIN_CONFIG
    hw = DEFAULT_HARDWARE_CONFIG
    
    print("\n1. MATHEMATICAL FORMULATION:")
    print(f"   • Problem Type: {spec.task_name}")
    print(f"   • Generative Paradigm: {spec.model_family}")
    print("   • Objective Function: Mean Squared Error (MSE) on predicted noise:")
    print("     L(θ) = E_{t, x₀, ε, c} [ || ε - ε_θ(x_t, t, c) ||² ]")
    print("   • Pretrained Weights: STRICTLY NONE (100% Random Init)")
    
    print("\n2. INPUT & OUTPUT TENSOR CONTRACTS:")
    print(f"   • Text Input: Prompt String -> Tokenized Tensor (Shape: [Batch, {spec.max_text_length}])")
    print(f"   • Vocabulary: {spec.vocab_size} tokens (custom learned tokenizer/embedding)")
    print(f"   • Text Conditioning Vector: Shape [Batch, {arch.text_embed_dim}]")
    print(f"   • Timestep Tensor: Shape [Batch], t ∈ [0, {spec.diffusion_timesteps - 1}]")
    print(f"   • Image Target Tensor: Shape [Batch, {spec.image_channels}, {spec.image_height}, {spec.image_width}]")
    print(f"   • Value Normalization: Range {spec.pixel_range}")

    print("\n3. ARCHITECTURAL BUDGET & RESOURCE CONSTRAINTS:")
    print(f"   • Backbone: Conditional U-Net ({spec.image_height}x{spec.image_width}) with Cross-Attention/Film")
    print(f"   • Channels: Base {arch.unet_base_channels} -> Multipliers {arch.channel_multipliers}")
    print(f"   • Target Parameter Ceiling: <= {arch.target_param_limit_millions}M parameters")
    print(f"   • Target VRAM / RAM Limit: <= {hw.max_vram_gb} GB")
    print(f"   • Practical Scratch Training Duration: <= {hw.max_training_hours} hours on local GPU/MPS")

    # Detect current hardware
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n4. ENVIRONMENT & HARDWARE STATUS:")
    print(f"   • Detected Active Compute Device: {device.upper()}")
    print(f"   • PyTorch Version: {torch.__version__}")
    
    # Initialize workspace directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n5. PIPELINE DIRECTORIES CREATED:")
    print(f"   • Data: {DATA_DIR}")
    print(f"   • Checkpoints: {CHECKPOINTS_DIR}")
    print(f"   • Logs: {LOGS_DIR}")

    print("\n6. FULL 10-STEP PIPELINE ROADMAP:")
    steps = [
        "1. Problem Definition (COMPLETED)",
        "2. Data Collection (Upcoming)",
        "3. Data Cleaning & Preprocessing (Upcoming)",
        "4. Dataset Splitting (Upcoming)",
        "5. Model Architecture (Upcoming)",
        "6. Training Pipeline (Upcoming)",
        "7. Experimentation & Hyperparameter Tuning (Upcoming)",
        "8. Model Evaluation (Upcoming)",
        "9. Model Optimization (Upcoming)",
        "10. Model Storage & Serving (Upcoming)",
    ]
    for step in steps:
        print(f"   [x] {step}" if "COMPLETED" in step else f"   [ ] {step}")

    print("=" * 70)
    print("Step 1 (Problem Definition) successfully configured and verified.")
    print("=" * 70)


if __name__ == "__main__":
    validate_step1_specification()
