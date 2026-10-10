"""
config.py
---------
STEP 1: PROBLEM DEFINITION & ML SPECIFICATION

This module formally specifies the machine learning problem, mathematical
formulation, tensor shapes, architectural boundaries, resource constraints,
and target metrics for building and training a custom Text-to-Image AI model
from scratch (strictly NO pretrained generative checkpoints).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ProblemSpecification:
    """
    Formal definition of the Machine Learning Problem.
    
    1. Task Type: Conditional Generative Modeling
    2. Input Modality: Natural Language Text Prompt (String)
    3. Output Modality: Synthetic RGB Image Tensor (3 x H x W)
    4. Learning Paradigm: Denoising Diffusion Probabilistic Model (DDPM)
       trained from random weight initialization.
    5. Mathematical Objective:
       Given conditioning text c and true noise epsilon ~ N(0, I),
       minimize L(theta) = E_{t, x_0, epsilon, c} [ || epsilon - epsilon_theta(x_t, t, c) ||^2 ]
    """
    task_name: str = "Conditional Text-to-Image Generation (From Scratch)"
    model_family: str = "DDPM (Denoising Diffusion Probabilistic Model)"
    
    # Input Specification
    max_text_length: int = 32          # Maximum tokens per prompt
    vocab_size: int = 2000             # Compact learned vocabulary size (character/BPE/word-level)
    
    # Output Specification
    image_channels: int = 3            # RGB
    image_height: int = 64             # 64x64 resolution (balanced for local scratch training)
    image_width: int = 64              # 64x64 resolution
    pixel_range: Tuple[float, float] = (-1.0, 1.0)
    
    # Mathematical Diffusion Schedule
    diffusion_timesteps: int = 500     # T steps for forward/reverse diffusion
    beta_start: float = 1e-4           # Linear or cosine variance schedule beta_1
    beta_end: float = 0.02             # beta_T


@dataclass
class ModelArchitectureConfig:
    """
    Architectural hyperparameters for the Custom Conditional U-Net.
    Designed for fast convergence and local training from scratch.
    """
    # Text Conditioning Encoder
    text_embed_dim: int = 128          # Learned text embedding dimension
    num_text_layers: int = 2           # Transformer/MLP layers for prompt encoder
    
    # Conditional U-Net Backbone
    unet_base_channels: int = 64       # Channel multiplier [64, 128, 256]
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks_per_stage: int = 2  # Residual blocks per down/up level
    time_embed_dim: int = 128          # Sinusoidal timestep embedding dimension
    dropout: float = 0.1
    
    # Target Parameter Scale (~12M to 20M parameters for fast scratch training)
    target_param_limit_millions: float = 25.0


@dataclass
class TrainingConfig:
    """
    Training hyperparameters and optimization configuration.
    """
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"
    lr_scheduler: str = "cosine_with_warmup"
    warmup_steps: int = 500
    total_epochs: int = 50
    gradient_clip_val: float = 1.0
    mixed_precision: bool = True       # fp16 on CUDA / fp32 on MPS/CPU
    checkpoint_every_n_epochs: int = 5  # periodic checkpoint cadence (best/latest are saved every epoch regardless)
    random_seed: int = 42               # training reproducibility (data shuffling, init, noise sampling)


@dataclass
class HardwareBudgetConfig:
    """
    Operational and hardware constraints for local execution.
    """
    max_vram_gb: float = 4.0           # Maximum memory consumption target
    max_training_hours: float = 3.0    # Practical time budget for complete scratch run
    target_fps_generation: float = 2.0 # Images per second during inference
    supported_devices: List[str] = field(default_factory=lambda: ["mps", "cuda", "cpu"])


@dataclass
class DataCollectionConfig:
    """
    STEP 2: DATA COLLECTION SPECIFICATION.

    Defines where the (image, text) training pairs come from and how many
    are collected. Every image/prompt pair produced under this config must
    satisfy the tensor contract fixed in Step 1 (ProblemSpecification):
    64x64 RGB images, prompts short enough to fit `max_text_length` tokens
    against a `vocab_size`-sized learned vocabulary.

    Source choice: "synthetic_procedural"
        Real captioned photo corpora (COCO Captions, Flickr8k/30k,
        Conceptual Captions, LAION subsets) are the eventual production
        path, but they require multi-GB downloads, licensing review, and
        image resolutions/subject complexity far beyond what a ~15-25M
        parameter model trained from random init on a laptop (Step 1
        budget: <=4GB VRAM, <=3h) can realistically learn. For this
        from-scratch pipeline, data is instead generated procedurally:
        simple colored shapes rendered on plain backgrounds, paired with
        a template-composed caption. This guarantees an exact, verifiable
        mapping between the words in a prompt (color, shape, background)
        and the pixels in its image -- the property a tiny scratch model
        needs to demonstrate real text-conditioning -- while being free,
        instant, license-free (CC0, no scraped/third-party material), and
        fully reproducible via `random_seed`.
    Swapping in a real dataset later only requires writing a new collector
    that emits the same on-disk contract (images/ + raw_metadata.jsonl
    format below); nothing downstream depends on how the pairs were made.
    """
    source_type: str = "synthetic_procedural"
    random_seed: int = 42
    num_samples: int = 500

    # Must mirror ProblemSpecification.image_height / image_width / image_channels.
    image_width: int = 64
    image_height: int = 64
    image_channels: int = 3

    shapes: List[str] = field(default_factory=lambda: [
        "circle", "square", "triangle", "diamond", "star", "hexagon", "cross",
    ])
    colors: List[str] = field(default_factory=lambda: [
        "red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan",
        "magenta", "turquoise", "gold", "silver", "crimson", "emerald",
        "violet", "indigo",
    ])
    backgrounds: List[str] = field(default_factory=lambda: [
        "a dark background", "a deep sky", "a white background",
        "a soft gradient", "a starry night", "a pastel background",
        "a black canvas", "a foggy backdrop",
    ])
    style_adjectives: List[str] = field(default_factory=lambda: [
        "bright", "vibrant", "glowing", "minimalist", "neon", "soft",
        "bold", "radiant", "muted", "electric",
    ])
    caption_templates: List[str] = field(default_factory=lambda: [
        "a {adjective} {color} {shape} on {background}",
        "a {color} {shape} on {background}",
        "a {adjective} {shape} in {color} on {background}",
    ])

    license: str = "CC0 (procedurally generated; no scraped or third-party imagery)"


@dataclass
class DataPreprocessingConfig:
    """
    STEP 3: DATA CLEANING & PREPROCESSING SPECIFICATION.

    Defines the exact transformations, validation rules, tokenization scheme,
    and tensor serialization parameters used to prepare raw (image, text) pairs
    for the custom scratch model.
    """
    # Image constraints & target geometry
    target_image_width: int = 64
    target_image_height: int = 64
    target_image_channels: int = 3
    pixel_range: Tuple[float, float] = (-1.0, 1.0)
    allowed_image_extensions: List[str] = field(default_factory=lambda: [".png", ".jpg", ".jpeg", ".webp"])
    
    # Text validation & tokenization constraints
    min_caption_chars: int = 3
    max_caption_chars: int = 200
    max_text_length: int = 32          # Max sequence length (matching Step 1 ProblemSpecification)
    special_tokens: dict = field(default_factory=lambda: {
        "<pad>": 0,
        "<start>": 1,
        "<end>": 2,
        "<unk>": 3,
    })
    
    # Preprocessing operations
    resample_filter: str = "bicubic"
    save_precomputed_tensors: bool = True
    deduplicate_by_content: bool = True



@dataclass
class DataSplittingConfig:
    """
    STEP 4: DATASET SPLITTING SPECIFICATION.

    Defines how the valid (image, caption) pairs produced by Step 3 are
    partitioned into Train / Validation / Test subsets for the later
    training, tuning, and final evaluation stages (Steps 6-8).

    Why three separate splits:
      - Train (80%): the only data the model's weights are ever fit on.
      - Validation (10%): held out from training; used to monitor
        overfitting and choose hyperparameters/checkpoints (Steps 6-7) --
        never used to update weights directly.
      - Test (10%): touched exactly once, at the very end (Step 8), to
        report a final, unbiased estimate of how the model performs on
        data that influenced it in no way at all -- not during training,
        not during hyperparameter selection.

    Leakage prevention:
      Step 2/3 verification found that a few samples render to
      pixel-identical images despite different captions/ids (same
      shape+color+background combo, jitter rounding to the same pixels at
      64x64). If two samples with identical image content landed in
      different splits, the model would effectively be graded on an image
      it already trained on. To prevent this, samples are first grouped by
      an exact hash of their processed image's pixel bytes; splitting
      assigns whole groups -- never individual members of a group -- to a
      single split.
    """
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42               # Shuffle seed -- reproducible split
    prevent_image_content_leakage: bool = True  # Group pixel-identical images before splitting

    def __post_init__(self):
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if total != 1.0:
            raise ValueError(f"train_ratio + val_ratio + test_ratio must equal 1.0 (got {total})")


@dataclass
class ExperimentConfig:
    """
    STEP 7: ONE CONTROLLED HYPERPARAMETER EXPERIMENT.

    A name/description plus a small set of overrides applied on top of
    Step 6's defaults (`TrainingConfig`, and -- for the diffusion-schedule
    group only -- `ProblemSpecification.diffusion_timesteps`). Every field
    left as None keeps the Step 6 default for that hyperparameter, so each
    experiment isolates exactly the one (or few) variable(s) named in its
    `varied` list -- a controlled, one-factor-at-a-time comparison rather
    than a combinatorial grid search.

    `ModelArchitectureConfig` (layer widths, depths, embedding dims) is
    deliberately NOT a field here: Step 7 tunes how the Step 5 network is
    trained, never what it is. `random_seed` is likewise not a field --
    every experiment shares the base `TrainingConfig.random_seed` so
    differences in outcome reflect the hyperparameter under test, not
    which random data shuffle/init a run happened to get.
    """
    name: str
    description: str
    varied: List[str]                   # which hyperparameter(s) this experiment isolates, for the report
    epochs: int = 3                     # kept small -- Step 7 is a comparison signal, not a full training run
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    weight_decay: Optional[float] = None
    gradient_clip_val: Optional[float] = None
    diffusion_timesteps: Optional[int] = None


# Paths configuration
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent
DATA_DIR = PACKAGE_ROOT / "data"
CHECKPOINTS_DIR = PACKAGE_ROOT / "checkpoints"
LOGS_DIR = PACKAGE_ROOT / "logs"

# Step 2 raw data collection layout: backend/custom_model/data/raw/{images/, raw_metadata.jsonl}
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
RAW_METADATA_PATH = RAW_DATA_DIR / "raw_metadata.jsonl"
DATASET_SUMMARY_PATH = RAW_DATA_DIR / "dataset_summary.json"

# Step 3 processed data layout: backend/custom_model/data/processed/{images/, tensors/, processed_metadata.jsonl, vocab.json, preprocessing_report.json}
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
PROCESSED_TENSORS_DIR = PROCESSED_DATA_DIR / "tensors"
PROCESSED_METADATA_PATH = PROCESSED_DATA_DIR / "processed_metadata.jsonl"
VOCAB_PATH = PROCESSED_DATA_DIR / "vocab.json"
PREPROCESSING_REPORT_PATH = PROCESSED_DATA_DIR / "preprocessing_report.json"

# Step 4 dataset split layout: backend/custom_model/data/splits/{train,val,test}_metadata.jsonl + split_summary.json
SPLITS_DIR = DATA_DIR / "splits"
TRAIN_METADATA_PATH = SPLITS_DIR / "train_metadata.jsonl"
VAL_METADATA_PATH = SPLITS_DIR / "val_metadata.jsonl"
TEST_METADATA_PATH = SPLITS_DIR / "test_metadata.jsonl"
SPLIT_SUMMARY_PATH = SPLITS_DIR / "split_summary.json"

# Step 6 training outputs: backend/custom_model/{checkpoints/, logs/training_log.jsonl, logs/training_summary.json}
LATEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "checkpoint_latest.pt"
BEST_CHECKPOINT_PATH = CHECKPOINTS_DIR / "checkpoint_best.pt"
TRAINING_LOG_PATH = LOGS_DIR / "training_log.jsonl"
TRAINING_SUMMARY_PATH = LOGS_DIR / "training_summary.json"

# Step 7 experiment outputs: backend/custom_model/experiments/{<name>/checkpoint_*.pt, <name>/training_*.jsonl|json, experiment_report.json, best_experiment_*}
# Kept entirely separate from Step 6's own checkpoints/logs above -- each
# experiment is isolated, and Step 6's canonical run is never overwritten.
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"
EXPERIMENT_REPORT_PATH = EXPERIMENTS_DIR / "experiment_report.json"
BEST_EXPERIMENT_CHECKPOINT_PATH = EXPERIMENTS_DIR / "best_experiment_checkpoint.pt"
BEST_EXPERIMENT_CONFIG_PATH = EXPERIMENTS_DIR / "best_experiment_config.json"

# Step 8 evaluation outputs: backend/custom_model/evaluation/{evaluation_report.json, samples/*.png}
# `samples/` is deliberately outside data/ entirely -- generated images are
# evaluation artifacts, never mistaken for or mixed into the dataset itself.
EVALUATION_DIR = PACKAGE_ROOT / "evaluation"
EVALUATION_SAMPLES_DIR = EVALUATION_DIR / "samples"
EVALUATION_REPORT_PATH = EVALUATION_DIR / "evaluation_report.json"

# Step 9 optimization outputs: backend/custom_model/optimized/{*.pt, optimization_report.json, samples/*.png}
# Kept completely isolated from Step 6 & 7 checkpoints -- original weights remain unchanged.
OPTIMIZED_DIR = PACKAGE_ROOT / "optimized"
OPTIMIZED_MODEL_FP16_PATH = OPTIMIZED_DIR / "optimized_model_fp16.pt"
OPTIMIZED_MODEL_PRUNED_PATH = OPTIMIZED_DIR / "optimized_model_pruned.pt"
BEST_OPTIMIZED_MODEL_PATH = OPTIMIZED_DIR / "best_optimized_model.pt"
OPTIMIZATION_REPORT_PATH = OPTIMIZED_DIR / "optimization_report.json"
OPTIMIZED_SAMPLES_DIR = OPTIMIZED_DIR / "samples"


@dataclass
class ModelOptimizationConfig:
    """
    STEP 9: MODEL OPTIMIZATION SPECIFICATION.

    Defines model compression, precision conversion (FP16/half-precision),
    sparsity pruning, and fast inference sampler parameters practical for laptop deployment.
    """
    enable_fp16: bool = True
    enable_pruning: bool = True
    pruning_sparsity: float = 0.10      # 10% magnitude-based non-critical weight pruning
    fast_sampling_steps: int = 50       # 10x accelerated reverse diffusion (50 vs 500 steps)
    ultra_fast_sampling_steps: int = 25 # 20x accelerated reverse diffusion (25 vs 500 steps)


# Default instantiated specification
DEFAULT_PROBLEM_SPEC = ProblemSpecification()
DEFAULT_ARCH_CONFIG = ModelArchitectureConfig()
DEFAULT_TRAIN_CONFIG = TrainingConfig()
DEFAULT_HARDWARE_CONFIG = HardwareBudgetConfig()
DEFAULT_DATA_COLLECTION_CONFIG = DataCollectionConfig()
DEFAULT_PREPROCESSING_CONFIG = DataPreprocessingConfig()
DEFAULT_SPLITTING_CONFIG = DataSplittingConfig()
DEFAULT_OPTIMIZATION_CONFIG = ModelOptimizationConfig()


