"""
Data Loader Module for Handwritten Digit Recognition.
Handles loading MNIST dataset, train/validation/test splitting, and sample visualization.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Add parent directory to sys.path to allow config import
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class MNISTDataLoader:
    """
    Data Loader class for fetching, splitting, and inspecting the MNIST dataset.
    """

    def __init__(self, validation_split: float = config.VALIDATION_SPLIT, random_seed: int = config.RANDOM_SEED) -> None:
        """
        Initialize the MNIST Data Loader.

        Args:
            validation_split (float): Ratio of training data allocated for validation.
            random_seed (int): Random seed for reproducibility.
        """
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.X_train_raw: np.ndarray = np.array([])
        self.y_train_raw: np.ndarray = np.array([])
        self.X_val_raw: np.ndarray = np.array([])
        self.y_val_raw: np.ndarray = np.array([])
        self.X_test_raw: np.ndarray = np.array([])
        self.y_test_raw: np.ndarray = np.array([])

    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Loads the raw MNIST dataset from keras.datasets and splits training set into train/validation sets.

        Returns:
            Tuple containing:
                - (X_train, y_train): Raw training samples and labels.
                - (X_val, y_val): Raw validation samples and labels.
                - (X_test, y_test): Raw test samples and labels.
        """
        print("[INFO] Fetching MNIST Dataset...")
        (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Split training set into Train and Validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=self.validation_split,
            random_state=self.random_seed,
            stratify=y_train_full
        )

        self.X_train_raw, self.y_train_raw = x_train, y_train
        self.X_val_raw, self.y_val_raw = x_val, y_val
        self.X_test_raw, self.y_test_raw = x_test, y_test

        print(f"[INFO] Dataset successfully loaded and split:")
        print(f"       - Train set: {self.X_train_raw.shape[0]} samples")
        print(f"       - Validation set: {self.X_val_raw.shape[0]} samples")
        print(f"       - Test set: {self.X_test_raw.shape[0]} samples")

        return (self.X_train_raw, self.y_train_raw), (self.X_val_raw, self.y_val_raw), (self.X_test_raw, self.y_test_raw)

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Generates statistical information regarding raw images and label distribution.

        Returns:
            Dict containing shapes, data types, pixel min/max values, and class distribution counts.
        """
        if self.X_train_raw.size == 0:
            self.load_data()

        unique, counts = np.unique(self.y_train_raw, return_counts=True)
        class_distribution = dict(zip([int(u) for u in unique], [int(c) for c in counts]))

        summary = {
            "train_shape": self.X_train_raw.shape,
            "val_shape": self.X_val_raw.shape,
            "test_shape": self.X_test_raw.shape,
            "pixel_dtype": str(self.X_train_raw.dtype),
            "min_pixel_val": int(np.min(self.X_train_raw)),
            "max_pixel_val": int(np.max(self.X_train_raw)),
            "class_distribution": class_distribution
        }
        return summary

    def visualize_samples(self, num_samples: int = 10, save_path: Path = config.ARTIFACTS_DIR / "sample_digits.png") -> Path:
        """
        Visualizes a grid of sample MNIST images along with their ground truth labels and saves the plot.

        Args:
            num_samples (int): Number of sample images to display.
            save_path (Path): File path to save output figure.

        Returns:
            Path: Path to saved figure.
        """
        if self.X_train_raw.size == 0:
            self.load_data()

        fig, axes = plt.subplots(2, num_samples // 2, figsize=(12, 5))
        axes = axes.flatten()

        # Randomly select sample indices
        indices = np.random.choice(len(self.X_train_raw), size=num_samples, replace=False)

        for i, idx in enumerate(indices):
            axes[i].imshow(self.X_train_raw[idx], cmap="gray")
            axes[i].set_title(f"Label: {self.y_train_raw[idx]}", fontsize=12, fontweight="bold")
            axes[i].axis("off")

        plt.suptitle("MNIST Dataset Sample Images", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[INFO] Sample images visualization saved to: {save_path}")
        return save_path


if __name__ == "__main__":
    loader = MNISTDataLoader()
    loader.load_data()
    summary = loader.get_dataset_summary()
    print("[SUMMARY]", summary)
    loader.visualize_samples()
