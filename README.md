# 🫁 Pneumonia Detection from Chest X-Rays

A deep learning system for detecting pneumonia from chest X-ray images using **PyTorch**, featuring lung segmentation preprocessing, transfer learning with **ResNet-18** and **EfficientNet-B0**, **Grad-CAM** visual explanations, and an interactive **Streamlit** web application.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage](#-usage)
  - [1. Lung Segmentation Preprocessing](#1-lung-segmentation-preprocessing)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Prediction with Grad-CAM](#4-prediction-with-grad-cam)
  - [5. Web Application](#5-web-application)
- [Model Architecture](#-model-architecture)
- [Pipeline](#-pipeline)
- [Results & Outputs](#-results--outputs)
- [License](#-license)

---

## 🔬 Overview

This project implements an end-to-end pipeline for **early-stage pneumonia detection** from chest X-ray images. The system preprocesses raw X-ray images using an AI-based **lung segmentation** model, trains a classifier using **transfer learning**, and provides interpretable predictions through **Grad-CAM heatmaps** — all accessible via a user-friendly **Streamlit web interface**.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Lung Segmentation** | Automated lung region extraction using `torchxrayvision` PSPNet model |
| **Transfer Learning** | Fine-tuned ResNet-18 and EfficientNet-B0 pretrained on ImageNet |
| **Grad-CAM Visualization** | Heatmaps highlighting regions that influenced the model's decision |
| **Early Stopping** | Prevents overfitting by monitoring validation accuracy with configurable patience |
| **Learning Rate Scheduling** | Adaptive LR reduction via `ReduceLROnPlateau` |
| **Custom Threshold** | Adjustable classification threshold (default: 0.45) for sensitivity/specificity tuning |
| **Comprehensive Evaluation** | Confusion matrix, classification report, sensitivity, specificity, AUC-ROC curve |
| **Web Interface** | Interactive Streamlit app for real-time X-ray upload and diagnosis |

---

## 📂 Project Structure

```
code/
│
├── app.py                  # Streamlit web application
├── build_model.py          # Model architecture builder (ResNet / EfficientNet)
├── data.py                 # Dataset loading, transforms, and DataLoaders
├── train.py                # Main training script (orchestrates the pipeline)
├── trainer.py              # Training loop with early stopping & LR scheduling
├── evaluate.py             # Evaluation metrics, ROC curve, training curves
├── predict.py              # Prediction pipeline with segmentation + Grad-CAM
├── gradcam.py              # Grad-CAM heatmap generation
├── lung_segmentation.py    # Lung region segmentation using torchxrayvision
├── segment_dataset.py      # Batch segmentation of the entire dataset
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
│
├── dataset/                # Original chest X-ray dataset
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
│
├── dataset_segmented/      # Lung-segmented dataset (generated)
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/                 # Saved model weights
│   ├── resnet_best.pth
│   ├── efficientnet_best.pth
│   └── ...
│
└── outputs/                # Generated outputs
    ├── gradcam_result.png
    ├── roc_curve.png
    └── training_curve.png
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.x** | Programming language |
| **PyTorch** | Deep learning framework |
| **torchvision** | Pretrained models & image transforms |
| **torchxrayvision** | Lung segmentation (PSPNet) |
| **OpenCV** | Image processing & Grad-CAM overlay |
| **Streamlit** | Interactive web application |
| **scikit-learn** | Evaluation metrics (AUC, ROC, confusion matrix) |
| **matplotlib** | Plotting training curves & ROC curve |
| **Pillow (PIL)** | Image loading & conversion |
| **NumPy** | Numerical computations |
| **tqdm** | Progress bars for dataset processing |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd code
```

### 2. Create a Virtual Environment

```bash
python -m venv pneumonia_env
pneumonia_env\Scripts\activate     # Windows
# source pneumonia_env/bin/activate  # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/) **before** running the above command.

<details>
<summary>📦 Full dependency list</summary>

| Library | Purpose |
|---|---|
| `torch` | Deep learning framework |
| `torchvision` | Pretrained models & image transforms |
| `torchxrayvision` | Lung segmentation model |
| `streamlit` | Web application |
| `opencv-python` | Image processing |
| `scikit-learn` | Evaluation metrics |
| `matplotlib` | Plotting & visualization |
| `numpy` | Numerical computation |
| `Pillow` | Image I/O |
| `tqdm` | Progress bars |

</details>

---

## 📊 Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

The dataset is organized into 3 splits:

| Split | Purpose |
|---|---|
| `train/` | Training the model |
| `val/` | Validation during training (early stopping) |
| `test/` | Final model evaluation |

Each split contains two classes: **NORMAL** and **PNEUMONIA**.

Place the dataset in the `dataset/` directory with the folder structure shown in [Project Structure](#-project-structure).

---

## 🚀 Usage

### 1. Lung Segmentation Preprocessing

Before training, preprocess the raw X-rays by extracting lung regions. This step uses a pretrained PSPNet model from `torchxrayvision` to isolate the lung area, removing noise from non-lung regions.

```bash
python segment_dataset.py
```

This processes all images in `dataset/` and saves segmented versions to `dataset_segmented/`.

### 2. Training

Train the model using transfer learning. The script supports both **ResNet-18** and **EfficientNet-B0**.

```bash
python train.py
```

**Configuration** (edit `train.py`):

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"resnet"` | Choose `"resnet"` or `"efficientnet"` |
| `lr` | `0.0001` | Learning rate |
| `epochs` | `20` | Maximum training epochs |
| `patience` | `3` | Early stopping patience |
| `batch_size` | `32` | Batch size (set in `data.py`) |

**Training features:**
- 🔄 **Transfer Learning** — Only the last convolutional block + classifier are fine-tuned
- ⏹️ **Early Stopping** — Training stops if validation accuracy doesn't improve for `patience` epochs
- 📉 **LR Scheduling** — Learning rate is halved after 1 epoch of no improvement
- 💾 **Best Model Saving** — Automatically saves the best model to `models/`

### 3. Evaluation

Evaluation is automatically run after training. It generates:

- **Test Accuracy**
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)
- **Sensitivity & Specificity**
- **AUC-ROC Score & Curve** (saved to `outputs/roc_curve.png`)
- **Training Curves** (saved to `outputs/training_curve.png`)

### 4. Prediction with Grad-CAM

Run a standalone prediction on any chest X-ray image:

```bash
python predict.py
```

Or generate a Grad-CAM heatmap directly:

```bash
python gradcam.py
```

The prediction pipeline:
1. **Segments the lung** from the input X-ray
2. **Classifies** the image as NORMAL or PNEUMONIA
3. **Generates a Grad-CAM heatmap** showing which regions influenced the prediction
4. Outputs the heatmap to `outputs/gradcam_result.png`

### 5. Web Application

Launch the interactive Streamlit web application:

```bash
streamlit run app.py
```

**Features:**
- 📤 Upload a chest X-ray image (JPG, JPEG, PNG)
- 🔍 View the uploaded image
- 🏷️ Get the prediction (NORMAL / PNEUMONIA) with confidence percentage
- 🗺️ View the Grad-CAM heatmap for model explainability

---

## 🧠 Model Architecture

### ResNet-18 (Default)
- **Base:** ResNet-18 pretrained on ImageNet
- **Frozen layers:** All layers except `layer4` and the fully connected layer
- **Classifier:** `nn.Linear(512, 2)`

### EfficientNet-B0
- **Base:** EfficientNet-B0 pretrained on ImageNet
- **Frozen layers:** All layers except the last feature block and the classifier
- **Classifier:** `nn.Linear(1280, 2)`

Both models use **binary classification** (NORMAL vs. PNEUMONIA) with softmax output.

---

## 🔄 Pipeline

```
┌──────────────────┐
│  Raw Chest X-ray │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  Lung Segmentation       │
│  (torchxrayvision PSPNet)│
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Data Augmentation       │
│  (RandomCrop, Flip, etc.)│
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Transfer Learning           │
│  (ResNet-18 / EfficientNet)  │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Prediction              │
│  NORMAL / PNEUMONIA      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Grad-CAM Heatmap        │
│  (Visual Explanation)    │
└──────────────────────────┘
```

---

## 📈 Results & Outputs

After training and evaluation, the following outputs are generated in the `outputs/` directory:

| Output | File | Description |
|---|---|---|
| **Grad-CAM Heatmap** | `outputs/gradcam_result.png` | Visual explanation of model prediction |
| **ROC Curve** | `outputs/roc_curve.png` | Receiver Operating Characteristic curve with AUC score |
| **Training Curves** | `outputs/training_curve.png` | Training loss and validation accuracy over epochs |

---

## 📝 License

This project is developed as an academic final project.

---

<p align="center">
  Built with ❤️ using PyTorch & Streamlit
</p>
