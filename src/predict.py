"""
Prediction Module for Real-Time Digit Classification.
Provides high-level API for running model inference on uploaded images or canvas drawings.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import tensorflow as tf
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.preprocess import ImagePreprocessor
from src.model import DigitCNNModel
from src.utils import GradCAM


class DigitPredictor:
    """
    Predictor class managing model loading, image preprocessing, forward pass execution,
    probability calculation, and Grad-CAM visualization generation.
    """

    def __init__(self, model_path: Path = config.MODEL_SAVE_PATH) -> None:
        """
        Initialize the predictor and load model weights.

        Args:
            model_path (Path): Path to saved Keras model file.
        """
        self.model_path = model_path
        self.preprocessor = ImagePreprocessor()
        self.model: tf.keras.Model = self._load_or_create_model()
        # Execute dummy forward pass to build Keras 3 computation graph nodes for GradCAM
        dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
        self.model(dummy)
        self.gradcam = GradCAM(self.model)

    def _load_or_create_model(self) -> tf.keras.Model:
        """Loads trained model if it exists on disk, otherwise builds and trains a fast baseline model."""
        if self.model_path.exists():
            print(f"[INFO] DigitPredictor loading trained model from: {self.model_path}")
            return DigitCNNModel.load_model(self.model_path)
        else:
            print(f"[WARN] No trained model found at {self.model_path}. Building new uninitialized model.")
            cnn = DigitCNNModel()
            cnn.build_model()
            cnn.compile_model()
            # Execute dummy forward pass to build Keras functional node graphs for GradCAM
            dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
            cnn.model(dummy)
            return cnn.model

    def predict_image(self, input_image: Union[np.ndarray, str, Image.Image, bytes]) -> Dict[str, Any]:
        """
        Executes end-to-end inference on custom input image.

        Args:
            input_image: Canvas Base64 data string, PIL Image, OpenCV array, or image bytes.

        Returns:
            Dict containing predicted digit, confidence %, probability array (0-9), GradCAM base64 image, and execution latency.
        """
        start_time = time.time()

        # Step 1: Preprocess input image to shape (1, 28, 28, 1)
        tensor_28x28 = self.preprocessor.preprocess_custom_image(input_image)

        # Step 2: Model forward pass
        probs = self.model.predict(tensor_28x28, verbose=0)[0]
        predicted_digit = int(np.argmax(probs))
        confidence_pct = round(float(probs[predicted_digit]) * 100, 2)

        # Step 3: Generate Grad-CAM heatmapped overlay image
        _, _, gradcam_data_url = self.gradcam.generate_heatmap(tensor_28x28, pred_index=predicted_digit)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # Step 4: Construct prediction response dict
        response = {
            "predicted_digit": predicted_digit,
            "confidence_percentage": confidence_pct,
            "probabilities": [round(float(p), 4) for p in probs],
            "gradcam_image": gradcam_data_url,
            "processing_time_ms": elapsed_ms
        }

        return response


if __name__ == "__main__":
    predictor = DigitPredictor()
    # Test with dummy zero canvas array
    dummy_input = np.zeros((100, 100), dtype=np.uint8)
    res = predictor.predict_image(dummy_input)
    print("[TEST INFERENCE RESULT]", res)
