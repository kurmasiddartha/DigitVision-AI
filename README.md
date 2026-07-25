# Handwritten Digit Recognition System (MNIST CNN)

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15+](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Flask 3.0+](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, production-grade Machine Learning repository for Handwritten Digit Recognition built with **TensorFlow / Keras**, **OpenCV**, **Scikit-Learn**, and **Flask**. 

Designed as an industry-standard portfolio project targeting **Machine Learning Engineer** placement interviews at top technology companies (NVIDIA, Qualcomm, Samsung, Oracle, HyperVerge, Walmart).

---

## 🌟 Features

- **Modular OOP Architecture**: Clean, modular Python library (`src/`) implementing separation of concerns, PEP8 compliance, and type hinting.
- **Advanced Preprocessing Pipeline**: Center-of-Mass image moment alignment, contour bounding-box cropping, and aspect-ratio scaling targeting domain shift mitigation.
- **Deep Convolutional Neural Network**: Multi-block CNN with Batch Normalization, Max Pooling, and Dropout regularization achieving **>99% Test Accuracy**.
- **Visual Explainability (Grad-CAM)**: Real-time Gradient-weighted Class Activation Mapping highlighting neural network attention regions.
- **Interactive Dark Dashboard**: Modern web client with HTML5 canvas drawing pad, stroke controls, preset digits, drag-and-drop file upload, live probability spectrums, and downloadable PDF reports.
- **Comprehensive Interview Preparation**: 12 detailed educational guides in `project_explanations/` covering math, code, and placement Q&A.

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------------------+
|                                 USER INPUT LAYER                                  |
|         HTML5 Drawing Canvas (280x280)    |    Uploaded Image Payload (PNG/JPG)    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           PREPROCESSING ENGINE (OpenCV)                           |
|   Auto-Inversion -> Contour Crop -> Aspect Resize (20x20) -> Center Mass (28x28)  |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        CNN INFERENCE & GRAD-CAM ENGINE                            |
|     Conv2D -> BatchNorm -> Relu -> MaxPool -> Dropout -> Dense -> Softmax (0-9)   |
|                   GradientTape -> Activation Heatmap Superimposition              |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                               WEB APPLICATION (Flask)                             |
|     REST API (/api/predict) -> Live Probabilities -> PDF Report -> History Log    |
+-----------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
MNIST-Digit-Recognition/
├── config.py                 # Central configuration: paths, hyperparameters, seeds
├── requirements.txt          # Python dependencies
├── .gitignore                # Git exclusions
├── README.md                 # Complete documentation
│
├── src/                      # Production Core ML Engine (OOP)
│   ├── __init__.py
│   ├── data_loader.py        # Dataset fetching and stratified splitting
│   ├── preprocess.py         # OpenCV image transformations & center-of-mass centering
│   ├── model.py              # Keras CNN model builder
│   ├── train.py              # Model training with EarlyStopping & Checkpoints
│   ├── evaluate.py           # Evaluation metrics & Confusion Matrix exporter
│   ├── predict.py            # Real-time inference engine
│   └── utils.py              # Grad-CAM visualizer & PDF report generator
│
├── models/                   # Saved model binaries (.keras)
│   └── cnn_model.keras
│
├── artifacts/                # Metrics summaries & generated plots
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── metrics_summary.json
│
├── notebooks/                # Exploratory Data Analysis & experiments
│   └── mnist_eda_and_experiments.ipynb
│
├── project_explanations/     # 12 In-Depth Educational Markdown Tutorials
│   ├── 01_Project_Overview.md
│   ├── 02_Project_Structure.md
│   ├── 03_MNIST_Dataset.md
│   ├── 04_Data_Loading.md
│   ├── 05_Image_Preprocessing.md
│   ├── 06_CNN_Architecture.md
│   ├── 07_Model_Training.md
│   ├── 08_Model_Evaluation.md
│   ├── 09_Model_Saving.md
│   ├── 10_Prediction.md
│   ├── 11_Web_Application.md
│   └── 12_Interview_Questions.md
│
└── app/                      # Web Application Layer (Flask)
    ├── app.py                # REST API endpoints
    ├── templates/
    │   └── index.html        # Modern Dark Dashboard UI
    └── static/
        ├── css/style.css     # CSS Design System
        └── js/main.js        # HTML5 Canvas & AJAX client
```

---

## 🚀 Quickstart & Installation

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/your-username/MNIST-Digit-Recognition.git
cd MNIST-Digit-Recognition

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Usage Commands

### Train the CNN Model
```bash
python src/train.py
```

### Evaluate Model on Test Set
```bash
python src/evaluate.py
```

### Test Inference Engine
```bash
python src/predict.py
```

### Launch Interactive Flask Web Dashboard
```bash
python app/app.py
```
Navigate to `http://127.0.0.1:5000` in your web browser.

---

## 📊 Dataset & Model Performance

### Dataset Overview
- **Dataset**: MNIST Handwritten Digits
- **Training Samples**: 54,000 (after 90/10 Stratified Train/Val split)
- **Validation Samples**: 6,000
- **Test Samples**: 10,000
- **Dimensions**: $28 \times 28 \times 1$ Grayscale

### Evaluation Metrics

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **99.2%** |
| **Test Loss** | **0.024** |
| **Precision (Weighted)** | **0.992** |
| **Recall (Weighted)** | **0.992** |
| **F1-Score (Weighted)** | **0.992** |

---

## 🎓 Project Explanations & Interview Prep

The `project_explanations/` folder contains 12 comprehensive markdown guides walking through every phase of the project from beginner concepts to advanced ML interview questions:
- [01_Project_Overview.md](project_explanations/01_Project_Overview.md)
- [06_CNN_Architecture.md](project_explanations/06_CNN_Architecture.md)
- [10_Prediction.md](project_explanations/10_Prediction.md) (Grad-CAM Math)
- [12_Interview_Questions.md](project_explanations/12_Interview_Questions.md) (Top Tech Company Q&A)

---

## 🔮 Future Improvements

1. **Model Compression & Edge Deployment**: Quantize model weights to INT8 via TensorFlow Lite for embedded microcontrollers (Qualcomm Snapdragon, ESP32-CAM).
2. **Data Augmentation**: Introduce random rotations ($\pm 15^\circ$), zoom, and elastic deformations during training to improve robustness against extreme handwriting slants.
3. **Multi-Digit OCR**: Extend bounding box detection using OpenCV `findContours` to segment and classify multi-digit strings (e.g., zip codes or math equations).
