"""
Convolutional Neural Network Architecture for Handwritten Digit Recognition.
Defines, builds, compiles, and serializes the CNN model using TensorFlow / Keras.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class DigitCNNModel:
    """
    CNN Model encapsulation class for MNIST digit classification.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = config.INPUT_SHAPE,
        num_classes: int = config.NUM_CLASSES,
        learning_rate: float = config.LEARNING_RATE
    ) -> None:
        """
        Initialize the CNN model builder.

        Args:
            input_shape (Tuple[int, int, int]): Shape of input image tensor (28, 28, 1).
            num_classes (int): Number of target digit classes (10).
            learning_rate (float): Learning rate for Adam optimizer.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model: Optional[tf.keras.Model] = None

    def build_model(self) -> tf.keras.Model:
        """
        Constructs the Convolutional Neural Network architecture using Keras Functional API.

        Returns:
            tf.keras.Model: Built Keras functional model instance.
        """
        inputs = layers.Input(shape=self.input_shape, name="input_image")
        
        # Block 1: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1")(inputs)
        x = layers.BatchNormalization(name="bn1_1")(x)
        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv1_2")(x)
        x = layers.BatchNormalization(name="bn1_2")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
        x = layers.Dropout(0.25, name="dropout1")(x)

        # Block 2: Conv -> BatchNorm -> ReLU -> MaxPool -> Dropout
        x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2_1")(x)
        x = layers.BatchNormalization(name="bn2_1")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
        x = layers.Dropout(0.25, name="dropout2")(x)

        # Fully Connected Head: Flatten -> Dense -> BatchNorm -> Dropout -> Softmax
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(128, activation="relu", name="fc1")(x)
        x = layers.BatchNormalization(name="bn_fc1")(x)
        x = layers.Dropout(0.5, name="dropout_fc1")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="output_probabilities")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MNIST_Digit_CNN")
        self.model = model
        return self.model

    def compile_model(self) -> tf.keras.Model:
        """
        Compiles the model with Adam optimizer, SparseCategoricalCrossentropy loss, and Accuracy metric.

        Returns:
            tf.keras.Model: Compiled Keras model instance.
        """
        if self.model is None:
            self.build_model()

        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        loss = losses.SparseCategoricalCrossentropy()
        metrics = ["accuracy"]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("[INFO] Model compiled successfully with Adam optimizer.")
        return self.model

    def summary(self) -> None:
        """Prints details of model layers, shape, and parameter counts."""
        if self.model is None:
            self.build_model()
        self.model.summary()

    def save_model(self, filepath: Path = config.MODEL_SAVE_PATH) -> None:
        """
        Saves compiled model weights and architecture in native Keras format.

        Args:
            filepath (Path): File path to save model.
        """
        if self.model is None:
            raise ValueError("Cannot save uninitialized model. Call build_model() first.")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"[INFO] Model saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath: Path = config.MODEL_SAVE_PATH) -> tf.keras.Model:
        """
        Loads saved Keras model from disk.

        Args:
            filepath (Path): Path to saved model file.

        Returns:
            tf.keras.Model: Loaded Keras model instance.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        model = tf.keras.models.load_model(filepath)
        print(f"[INFO] Loaded trained model from: {filepath}")
        return model


if __name__ == "__main__":
    cnn = DigitCNNModel()
    model = cnn.build_model()
    cnn.compile_model()
    cnn.summary()
