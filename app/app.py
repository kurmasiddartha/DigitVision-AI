"""
Flask Web Application Server for Handwritten Digit Recognition System.
Provides RESTful APIs for real-time model inference, model metadata, session history, and report export.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any

from flask import Flask, render_template, request, jsonify, send_file

# Add project root directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import config
from src.predict import DigitPredictor
from src.utils import ReportGenerator

app = Flask(__name__, template_folder="templates", static_folder="static")

# Instantiate singleton Digit Predictor
predictor = DigitPredictor()

# In-memory session prediction history log
PREDICTION_HISTORY: List[Dict[str, Any]] = []


@app.route("/")
def index():
    """Renders the primary interactive web dashboard."""
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    API Endpoint for Digit Prediction.
    Accepts JSON containing Base64 image payload OR multipart Form-Data file upload.
    """
    try:
        image_input = None

        if request.is_json:
            data = request.get_json()
            image_input = data.get("image")
        elif "file" in request.files:
            file = request.files["file"]
            image_input = file.read()

        if not image_input:
            return jsonify({"error": "No valid image data provided"}), 400

        # Run model inference & Grad-CAM visualizer
        result = predictor.predict_image(image_input)

        # Log prediction entry into history
        history_entry = {
            "id": len(PREDICTION_HISTORY) + 1,
            "digit": result["predicted_digit"],
            "confidence": result["confidence_percentage"],
            "latency": result["processing_time_ms"],
            "timestamp": request.date or "Just now"
        }
        PREDICTION_HISTORY.insert(0, history_entry)
        if len(PREDICTION_HISTORY) > 20:
            PREDICTION_HISTORY.pop()

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def model_info():
    """API Endpoint serving architecture details, hyperparameter configuration, and test metrics."""
    metrics = {}
    if config.EVALUATION_METRICS_PATH.exists():
        with open(config.EVALUATION_METRICS_PATH, "r") as f:
            metrics = json.load(f)

    info = {
        "model_name": "MNIST Convolutional Neural Network (CNN)",
        "input_shape": list(config.INPUT_SHAPE),
        "num_classes": config.NUM_CLASSES,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "test_metrics": metrics
    }
    return jsonify(info)


@app.route("/api/history", methods=["GET"])
def get_history():
    """API Endpoint returning recent prediction history."""
    return jsonify({"history": PREDICTION_HISTORY})


@app.route("/api/export-report", methods=["POST"])
def export_report():
    """Generates and downloads a summary report of a prediction result."""
    try:
        data = request.get_json()
        report_path = config.ARTIFACTS_DIR / "digit_prediction_report.pdf"
        ReportGenerator.generate_pdf_report(data, report_path)
        return send_file(report_path, as_attachment=True, download_name="digit_prediction_report.pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"[INFO] Starting Handwritten Digit Recognition Web App on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
