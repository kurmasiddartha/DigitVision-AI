"""
Image Preprocessing Module for Handwritten Digit Recognition.
Handles pixel normalization, tensor reshaping, and real-time user-input digit centering & bounding box transformations.
"""

import sys
import base64
import io
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import cv2
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


class ImagePreprocessor:
    """
    Preprocessor class providing methods for preparing MNIST arrays and real-world user drawings/images.
    """

    def __init__(self, target_shape: Tuple[int, int, int] = config.INPUT_SHAPE) -> None:
        """
        Initialize the Image Preprocessor.

        Args:
            target_shape (Tuple[int, int, int]): Target tensor shape (rows, cols, channels).
        """
        self.target_shape = target_shape
        self.img_rows, self.img_cols, self.num_channels = target_shape

    def preprocess_dataset(self, images: np.ndarray) -> np.ndarray:
        """
        Preprocesses a batch of raw MNIST dataset images.
        Normalizes pixels to [0.0, 1.0] and reshapes tensor to (N, 28, 28, 1).

        Args:
            images (np.ndarray): Input images with shape (N, 28, 28).

        Returns:
            np.ndarray: Preprocessed float32 array with shape (N, 28, 28, 1).
        """
        # Reshape tensor to include channel dimension
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=-1)

        # Normalize pixel values from [0, 255] to [0.0, 1.0]
        images_normalized = images.astype(np.float32) / 255.0

        return images_normalized

    def preprocess_custom_image(self, input_image: Union[np.ndarray, str, Image.Image, bytes]) -> np.ndarray:
        """
        Preprocesses custom input images (from canvas base64, uploaded file, or PIL Image).
        Applies grayscale conversion, color auto-inversion, bounding box cropping,
        aspect ratio scaling, center-of-mass alignment, normalization, and reshaping.

        Args:
            input_image: Image data as Base64 string, OpenCV array, PIL Image, or raw bytes.

        Returns:
            np.ndarray: Preprocessed image batch tensor with shape (1, 28, 28, 1).
        """
        # Step 1: Decode input format into a Grayscale OpenCV image
        img_gray = self._to_grayscale_numpy(input_image)

        # Step 2: Invert background if canvas background is light / digit is dark
        # MNIST dataset format requires: Background = Black (0), Digit = White (255)
        if np.mean(img_gray) > 127:
            img_gray = cv2.bitwise_not(img_gray)

        # Step 3: Apply light Gaussian blur to reduce aliasing noise before thresholding
        img_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # Step 4: Adaptive Otsu threshold — handles varying stroke brightness from anti-aliased canvas strokes
        # Hard threshold (e.g. 30) is too low for anti-aliased edges; Otsu computes the optimal cutoff automatically
        _, thresh = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 5: Morphological closing — reconnects thin broken strokes in digits like 4, 7, 1
        # after aggressive thresholding (e.g. top bar of "4" may become disconnected)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Step 6: Extract digit bounding box using contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            img_area = thresh.shape[0] * thresh.shape[1]
            # Filter contours:
            #   - Lower bound (>= 5 px²): removes tiny specks of noise
            #   - Upper bound (<= 80% image area): removes large background shapes such as the
            #     card border in icon-style images (e.g. white digit on green rounded rectangle).
            #     Without this, the bounding box spans the whole image and the digit is a tiny smudge.
            valid_contours = [
                c for c in contours
                if 5 <= cv2.contourArea(c) <= img_area * 0.80
            ]
            if not valid_contours:
                # Fallback: use the single largest contour if every contour was filtered out
                valid_contours = [max(contours, key=cv2.contourArea)]
            x_vals  = [cv2.boundingRect(c)[0] for c in valid_contours]
            y_vals  = [cv2.boundingRect(c)[1] for c in valid_contours]
            x2_vals = [cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] for c in valid_contours]
            y2_vals = [cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] for c in valid_contours]
            x, y   = min(x_vals), min(y_vals)
            x2, y2 = max(x2_vals), max(y2_vals)
            digit_crop = thresh[y:y2, x:x2]
        else:
            # Fallback if canvas is empty or no contour detected
            digit_crop = thresh

        # Step 6: Scale digit maintaining aspect ratio inside a 20x20 bounding box (MNIST spec)
        h_crop, w_crop = digit_crop.shape
        if h_crop > 0 and w_crop > 0:
            if h_crop > w_crop:
                scale = 20.0 / h_crop
                new_h = 20
                new_w = max(1, int(w_crop * scale))
            else:
                scale = 20.0 / w_crop
                new_w = 20
                new_h = max(1, int(h_crop * scale))

            resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_digit = cv2.resize(thresh, (20, 20), interpolation=cv2.INTER_AREA)

        # Step 7: Pad resized digit into 28x28 canvas using Center-of-Mass alignment
        padded_28x28 = np.zeros((28, 28), dtype=np.uint8)

        # Compute center of mass of the resized digit
        M = cv2.moments(resized_digit)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = resized_digit.shape[1] // 2, resized_digit.shape[0] // 2

        # Compute top-left placement coordinates targeting center (14, 14)
        start_x = max(0, min(28 - resized_digit.shape[1], 14 - cx))
        start_y = max(0, min(28 - resized_digit.shape[0], 14 - cy))

        padded_28x28[start_y:start_y + resized_digit.shape[0], start_x:start_x + resized_digit.shape[1]] = resized_digit

        # Step 8: Normalize to [0.0, 1.0] and reshape to (1, 28, 28, 1)
        normalized_img = padded_28x28.astype(np.float32) / 255.0
        final_tensor = np.expand_dims(normalized_img, axis=(0, -1))

        return final_tensor

    def _to_grayscale_numpy(self, input_image: Union[np.ndarray, str, Image.Image, bytes]) -> np.ndarray:
        """
        Converts diverse image data inputs into a 2D Grayscale numpy array.
        """
        if isinstance(input_image, str):
            # Strip base64 metadata header if present
            if "," in input_image:
                input_image = input_image.split(",")[1]
            img_bytes = base64.b64decode(input_image)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")
            return np.array(pil_img)

        elif isinstance(input_image, bytes):
            pil_img = Image.open(io.BytesIO(input_image)).convert("L")
            return np.array(pil_img)

        elif isinstance(input_image, Image.Image):
            return np.array(input_image.convert("L"))

        elif isinstance(input_image, np.ndarray):
            if len(input_image.shape) == 3:
                if input_image.shape[2] == 4:
                    # RGBA to Grayscale
                    return cv2.cvtColor(input_image, cv2.COLOR_RGBA2GRAY)
                elif input_image.shape[2] == 3:
                    # RGB to Grayscale
                    return cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            return input_image

        else:
            raise ValueError(f"Unsupported image input type: {type(input_image)}")


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    dummy_dataset = np.zeros((10, 28, 28), dtype=np.uint8)
    processed = preprocessor.preprocess_dataset(dummy_dataset)
    print(f"[TEST] Dataset shape after preprocessing: {processed.shape}, dtype: {processed.dtype}")
