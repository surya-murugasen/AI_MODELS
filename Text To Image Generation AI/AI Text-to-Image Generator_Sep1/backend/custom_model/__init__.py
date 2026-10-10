"""
custom_model package
--------------------
Contains the complete 10-stage machine learning pipeline for building,
training, evaluating, and serving a custom Text-to-Image AI model from scratch
(without using pretrained generative weights).
"""

from .config import (
    ProblemSpecification,
    ModelArchitectureConfig,
    TrainingConfig,
    HardwareBudgetConfig,
    DataCollectionConfig,
    DataPreprocessingConfig,
    DataSplittingConfig,
    DEFAULT_PROBLEM_SPEC,
    DEFAULT_DATA_COLLECTION_CONFIG,
    DEFAULT_PREPROCESSING_CONFIG,
    DEFAULT_SPLITTING_CONFIG,
    DEFAULT_TRAIN_CONFIG,
    ExperimentConfig,
    ModelOptimizationConfig,
    DEFAULT_OPTIMIZATION_CONFIG,
)
from .preprocess import TextTokenizer, DatasetPreprocessor
from .split_dataset import DatasetSplitter
from .model import TextEncoder, ConditionalUNet, TextToImageDiffusionModel
from .train import TextToImageDataset, Trainer
from .experiment import ExperimentRunner, build_experiment_configs
from .evaluate import run_step8_evaluation
from .optimize import ModelOptimizer
from .model_storage import save_model_artifact, load_model, verify_round_trip, run_step10_model_storage

__all__ = [
    "ProblemSpecification",
    "ModelArchitectureConfig",
    "TrainingConfig",
    "HardwareBudgetConfig",
    "DataCollectionConfig",
    "DataPreprocessingConfig",
    "DEFAULT_PROBLEM_SPEC",
    "DEFAULT_DATA_COLLECTION_CONFIG",
    "DEFAULT_PREPROCESSING_CONFIG",
    "TextTokenizer",
    "DatasetPreprocessor",
    "DataSplittingConfig",
    "DEFAULT_SPLITTING_CONFIG",
    "DatasetSplitter",
    "TextEncoder",
    "ConditionalUNet",
    "TextToImageDiffusionModel",
    "DEFAULT_TRAIN_CONFIG",
    "TextToImageDataset",
    "Trainer",
    "ExperimentConfig",
    "ExperimentRunner",
    "build_experiment_configs",
    "run_step8_evaluation",
    "ModelOptimizationConfig",
    "DEFAULT_OPTIMIZATION_CONFIG",
    "ModelOptimizer",
    "save_model_artifact",
    "load_model",
    "verify_round_trip",
    "run_step10_model_storage",
]
