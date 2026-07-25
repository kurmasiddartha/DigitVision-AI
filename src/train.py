"""
Model Training Module for Handwritten Digit Recognition.
Executes end-to-end model training using Keras callbacks and exports training metrics.
"""

import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import MNISTDataLoader
from src.preprocess import ImagePreprocessor
from src.model import DigitCNNModel


class ModelTrainer:
    """
    Trainer class encapsulating the full training lifecycle including dataset loading,
    preprocessing, callback configuration, training execution, and metric visualization.
    """

    def __init__(self) -> None:
        """Initialize data loader, preprocessor, and model instances."""
        self.data_loader = MNISTDataLoader()
        self.preprocessor = ImagePreprocessor()
        self.model_builder = DigitCNNModel()
        self.history: Dict[str, Any] = {}

    def run_training_pipeline(self) -> tf.keras.callbacks.History:
        """
        Runs the complete model training workflow.

        Returns:
            tf.keras.callbacks.History: Keras training history object.
        """
        print("==================================================")
        print("           STARTING MODEL TRAINING PIPELINE       ")
        print("==================================================")

        # 1. Load Data
        (X_train_raw, y_train), (X_val_raw, y_val), _ = self.data_loader.load_data()

        # 2. Preprocess Images
        X_train = self.preprocessor.preprocess_dataset(X_train_raw)
        X_val = self.preprocessor.preprocess_dataset(X_val_raw)

        # 3. Build & Compile CNN Model
        model = self.model_builder.build_model()
        self.model_builder.compile_model()

        # 4. Configure Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(config.MODEL_SAVE_PATH),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # 5. Fit Model
        print(f"[INFO] Training CNN model for {config.EPOCHS} epochs with batch size {config.BATCH_SIZE}...")
        history = model.fit(
            X_train,
            y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )

        self.history = history.history

        # 6. Save final model instance
        self.model_builder.save_model(config.MODEL_SAVE_PATH)

        # 7. Plot and save training history curves
        self.plot_training_history(config.TRAINING_HISTORY_PLOT_PATH)

        print("==================================================")
        print("           TRAINING PIPELINE COMPLETED           ")
        print("==================================================")

        return history

    def plot_training_history(self, save_path: Path = config.TRAINING_HISTORY_PLOT_PATH) -> Path:
        """
        Generates and saves Training vs Validation Accuracy & Loss curves.

        Args:
            save_path (Path): Path where the figure will be saved.

        Returns:
            Path: Path to saved figure.
        """
        if not self.history:
            print("[WARN] No training history available to plot.")
            return save_path

        acc = self.history.get("accuracy", [])
        val_acc = self.history.get("val_accuracy", [])
        loss = self.history.get("loss", [])
        val_loss = self.history.get("val_loss", [])
        epochs_range = range(1, len(acc) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy Subplot
        ax1.plot(epochs_range, acc, "bo-", label="Training Accuracy", linewidth=2)
        ax1.plot(epochs_range, val_acc, "ro-", label="Validation Accuracy", linewidth=2)
        ax1.set_title("Training and Validation Accuracy", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epochs", fontsize=10)
        ax1.set_ylabel("Accuracy", fontsize=10)
        ax1.legend(loc="lower right")
        ax1.grid(True, linestyle="--", alpha=0.6)

        # Loss Subplot
        ax2.plot(epochs_range, loss, "bo-", label="Training Loss", linewidth=2)
        ax2.plot(epochs_range, val_loss, "ro-", label="Validation Loss", linewidth=2)
        ax2.set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epochs", fontsize=10)
        ax2.set_ylabel("Loss", fontsize=10)
        ax2.legend(loc="upper right")
        ax2.grid(True, linestyle="--", alpha=0.6)

        plt.suptitle("CNN Model Training Metrics", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[INFO] Training history plots saved to: {save_path}")
        return save_path


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run_training_pipeline()
