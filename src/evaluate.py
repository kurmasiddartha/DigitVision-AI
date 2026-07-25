"""
Model Evaluation Module for Handwritten Digit Recognition.
Computes test metrics (Loss, Accuracy, Precision, Recall, F1), confusion matrix,
and generates classification reports.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.data_loader import MNISTDataLoader
from src.preprocess import ImagePreprocessor
from src.model import DigitCNNModel


class ModelEvaluator:
    """
    Evaluator class for comprehensive performance validation of trained CNN models on test data.
    """

    def __init__(self, model_path: Path = config.MODEL_SAVE_PATH) -> None:
        """
        Initialize the evaluator with a saved model file path.

        Args:
            model_path (Path): Path to saved Keras model file.
        """
        self.model_path = model_path
        self.data_loader = MNISTDataLoader()
        self.preprocessor = ImagePreprocessor()
        self.model: tf.keras.Model = DigitCNNModel.load_model(model_path)

    def evaluate_test_set(self) -> Dict[str, Any]:
        """
        Runs evaluation on the test set, computing accuracy, loss, precision, recall, and F1 score.

        Returns:
            Dict[str, Any]: Structured summary dictionary of evaluation metrics.
        """
        print("==================================================")
        print("          STARTING MODEL EVALUATION PIPELINE      ")
        print("==================================================")

        # 1. Load test data
        _, _, (X_test_raw, y_test) = self.data_loader.load_data()

        # 2. Preprocess test images
        X_test = self.preprocessor.preprocess_dataset(X_test_raw)

        # 3. Model evaluation loss & accuracy
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=1)

        # 4. Predictions & Probability vectors
        y_probs = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_probs, axis=1)

        # 5. Compute Classification Metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

        report_str = classification_report(y_test, y_pred, target_names=config.CLASS_LABELS)
        report_dict = classification_report(y_test, y_pred, target_names=config.CLASS_LABELS, output_dict=True)

        print("\n[CLASSIFICATION REPORT]")
        print(report_str)

        # 6. Plot Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, config.CONFUSION_MATRIX_PLOT_PATH)

        metrics_summary = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_score_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_score_weighted": float(f1_weighted),
            "total_test_samples": int(len(y_test)),
            "per_class_metrics": report_dict
        }

        # 7. Save Metrics Summary to JSON
        with open(config.EVALUATION_METRICS_PATH, "w") as f:
            json.dump(metrics_summary, f, indent=4)

        print(f"[INFO] Evaluation metrics summary exported to: {config.EVALUATION_METRICS_PATH}")
        print("==================================================")
        print("          EVALUATION PIPELINE COMPLETED           ")
        print("==================================================")

        return metrics_summary

    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Path = config.CONFUSION_MATRIX_PLOT_PATH) -> Path:
        """
        Plots and saves a high-resolution heatmapped confusion matrix.

        Args:
            cm (np.ndarray): 10x10 confusion matrix array.
            save_path (Path): Target file path.

        Returns:
            Path: Path to saved matrix plot.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=config.CLASS_LABELS,
            yticklabels=config.CLASS_LABELS,
            cbar=True
        )
        plt.title("MNIST CNN Model - Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.ylabel("True Label", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[INFO] Confusion matrix plot saved to: {save_path}")
        return save_path


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_test_set()
