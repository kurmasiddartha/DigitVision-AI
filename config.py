"""
Central Configuration Module for MNIST Handwritten Digit Recognition System.
Provides hyperparameters, directory paths, data shapes, and runtime settings.
"""

from pathlib import Path
from typing import Tuple

# Base Project Directories
BASE_DIR: Path = Path(__file__).resolve().parent
SRC_DIR: Path = BASE_DIR / "src"
MODELS_DIR: Path = BASE_DIR / "models"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
DATA_DIR: Path = BASE_DIR / "data"
GRADCAM_DIR: Path = ARTIFACTS_DIR / "gradcam_samples"
APP_DIR: Path = BASE_DIR / "app"

# Ensure runtime directories exist
for directory in [MODELS_DIR, ARTIFACTS_DIR, DATA_DIR, GRADCAM_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Parameters
IMG_ROWS: int = 28
IMG_COLS: int = 28
NUM_CHANNELS: int = 1
INPUT_SHAPE: Tuple[int, int, int] = (IMG_ROWS, IMG_COLS, NUM_CHANNELS)
NUM_CLASSES: int = 10
CLASS_LABELS: Tuple[str, ...] = tuple(str(i) for i in range(NUM_CLASSES))

# Training Hyperparameters
BATCH_SIZE: int = 64
EPOCHS: int = 15
LEARNING_RATE: float = 0.001
VALIDATION_SPLIT: float = 0.1
RANDOM_SEED: int = 42

# Model Persistence Paths
MODEL_SAVE_PATH: Path = MODELS_DIR / "cnn_model.keras"
TRAINING_HISTORY_PLOT_PATH: Path = ARTIFACTS_DIR / "training_history.png"
CONFUSION_MATRIX_PLOT_PATH: Path = ARTIFACTS_DIR / "confusion_matrix.png"
EVALUATION_METRICS_PATH: Path = ARTIFACTS_DIR / "metrics_summary.json"

# Web App Configuration
HOST: str = "127.0.0.1"
PORT: int = 5000
DEBUG: bool = True
