"""
Utility Functions and Helpers for MNIST Handwritten Digit Recognition.
Includes Grad-CAM visualizer, Base64 image codecs, and PDF summary report generator.
"""

import sys
import io
import json
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) Generator.
    Produces visual explanations highlighting regions of an input image that most influenced the CNN's digit prediction.
    """

    def __init__(self, model: tf.keras.Model, layer_name: Optional[str] = None) -> None:
        """
        Initialize Grad-CAM builder.

        Args:
            model (tf.keras.Model): Trained Keras model instance.
            layer_name (Optional[str]): Target convolutional layer name. Default picks last Conv2D layer.
        """
        self.model = model
        # Execute dummy pass to build Keras functional node tensors
        dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
        try:
            self.model(dummy)
        except Exception:
            pass
        self.layer_name = layer_name or self._find_last_conv_layer()
        self.grad_model = self._build_grad_model()

    def _find_last_conv_layer(self) -> str:
        """Finds the name of the final Conv2D layer in the Keras model graph."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("No Conv2D layer found in model architecture.")

    def _build_grad_model(self) -> tf.keras.Model:
        """Builds a dual-output gradient model for target layer outputs and top prediction logits."""
        try:
            target_layer = self.model.get_layer(self.layer_name)
            return tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )
        except Exception:
            # Fallback for Keras Sequential models: reconstruct functional graph from layers
            inputs = tf.keras.Input(shape=config.INPUT_SHAPE, name="gradcam_input")
            x = inputs
            target_output = None
            for layer in self.model.layers:
                x = layer(x)
                if layer.name == self.layer_name:
                    target_output = x
            return tf.keras.Model(inputs=inputs, outputs=[target_output, x])

    def generate_heatmap(self, input_tensor: np.ndarray, pred_index: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Computes the Grad-CAM activation heatmap for a single preprocessed input tensor (1, 28, 28, 1).

        Args:
            input_tensor (np.ndarray): Tensor batch of shape (1, 28, 28, 1).
            pred_index (Optional[int]): Target class index. Defaults to highest predicted class.

        Returns:
            Tuple[np.ndarray, np.ndarray, str]: (Heatmap array, Overlay image BGR array, Base64 image string).
        """
        input_tf = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(input_tf)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        # Calculate gradients of top predicted class score w.r.t conv output feature map
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature map channels by gradient importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU activation and normalize heatmap between [0.0, 1.0]
        heatmap = tf.maximum(heatmap, 0.0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        # Resize heatmap to input image dimensions (28x28)
        heatmap_resized = cv2.resize(heatmap, (config.IMG_COLS, config.IMG_ROWS))

        # Convert heatmap to uint8 color overlay (JET colormap)
        heatmap_colored = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

        # Create input image display array (28x28 3-channel RGB)
        raw_digit = np.uint8(input_tensor[0, :, :, 0] * 255)
        raw_digit_rgb = cv2.cvtColor(raw_digit, cv2.COLOR_GRAY2BGR)

        # Superimpose heatmap onto original digit
        overlay = cv2.addWeighted(raw_digit_rgb, 0.5, heatmap_colored, 0.5, 0)

        # Encode overlay image to Base64 data URL
        _, buffer = cv2.imencode(".png", overlay)
        base64_str = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_str}"

        return heatmap_resized, overlay, data_url


class ReportGenerator:
    """Utility class to export predictions and model evaluation metadata into downloadable reports."""

    @staticmethod
    def generate_json_report(prediction_data: Dict[str, Any]) -> str:
        """Formats prediction metadata into formatted JSON string."""
        return json.dumps(prediction_data, indent=4)

    @staticmethod
    def generate_pdf_report(prediction_data: Dict[str, Any], output_path: Path) -> Path:
        """
        Creates a PDF report summarizing prediction results, confidence scores, and probabilities.
        """
        try:
            from fpdf import FPDF
        except ImportError:
            # Fallback if fpdf2 is not installed
            output_path.write_text(json.dumps(prediction_data, indent=4))
            return output_path

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, "Handwritten Digit Recognition Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"Predicted Digit: {prediction_data.get('predicted_digit')}", ln=True)
        pdf.cell(0, 8, f"Confidence Score: {prediction_data.get('confidence_percentage')}%", ln=True)
        pdf.cell(0, 8, f"Processing Time: {prediction_data.get('processing_time_ms')} ms", ln=True)
        pdf.ln(10)

        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Probability Distribution Across Digits (0-9):", ln=True)
        pdf.set_font("Helvetica", "", 11)

        probs = prediction_data.get("probabilities", [])
        for digit, prob in enumerate(probs):
            pdf.cell(0, 6, f"Digit '{digit}': {prob * 100:.2f}%", ln=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(output_path))
        return output_path


if __name__ == "__main__":
    print("[INFO] Utilities module compiled successfully.")
