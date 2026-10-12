# Deployment Guide

This document covers environment setup, training, inference, and deployment of the Fruit and Vegetable Classifier.

---

## 1. Requirements

- Python 3.10 or later
- CUDA-capable GPU recommended for training (CPU inference is supported)
- Git

---

## 2. Environment Setup

### Clone the Repository

```bash
git clone https://github.com/<your-username>/fruits-vegetable-classifier.git
cd fruits-vegetable-classifier
```

### Create a Virtual Environment

```bash
python -m venv venv

# Activate -- Windows
venv\Scripts\activate

# Activate -- macOS / Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:

| Package | Version | Purpose |
|---|---|---|
| torch | >=2.0.0 | Core deep learning framework |
| torchvision | >=0.15.0 | EfficientNet model and transforms |
| streamlit | >=1.30.0 | Web application |
| pytorch-grad-cam | >=1.5.0 | GradCAM visualization |
| scikit-learn | >=1.3.0 | Classification report and metrics |
| matplotlib / seaborn | >=3.7.0 / >=0.12.0 | Plotting |

---

## 3. Dataset Download

The dataset is downloaded from Kaggle using the `kagglehub` library. Kaggle authentication credentials must be configured before running the script.

**Configure Kaggle credentials:**

1. Log in to [kaggle.com](https://www.kaggle.com) and go to Account > API > Create New API Token.
2. Place the downloaded `kaggle.json` file at `~/.kaggle/kaggle.json` (Linux/macOS) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).

**Run the download script:**

```bash
python download_data.py
```

This script downloads the dataset and structures it into `data/train/`, `data/validation/`, and `data/test/` with one subdirectory per class.

---

## 4. Configuration

All hyperparameters are centralized in `train_config.yaml`. Edit this file before training to change any setting:

```yaml
model:
  backbone: efficientnet_b0
  num_classes: 36
  dropout: 0.3

training:
  batch_size: 32
  epochs_phase1: 5       # Feature extraction -- backbone frozen
  epochs_phase2: 10      # Fine-tuning -- last 2 blocks unfrozen
  lr_phase1: 0.001
  lr_phase2: 0.0001
  weight_decay: 0.0001
  label_smoothing: 0.1
  early_stop_patience: 5
  seed: 42

data:
  image_size: 224
  num_workers: 4
```

---

## 5. Training

```bash
python -m src.train
```

Training output is logged to the terminal (loss and accuracy per epoch for both phases). The best checkpoint is saved automatically to `results/best_model.pth` based on lowest validation loss.

**Expected training time:**

| Hardware | Approx. Duration |
|---|---|
| NVIDIA RTX 3060 or equivalent | ~8 minutes |
| Google Colab T4 GPU | ~12 minutes |
| CPU only | ~2 hours |

---

## 6. Evaluation

After training, generate the confusion matrix, classification report, GradCAM heatmaps, and misclassification samples:

```bash
python -m src.evaluate
```

Output files written to `results/`:

| File | Description |
|---|---|
| `classification_report.csv` | Per-class precision, recall, F1-score |
| `confusion_matrix.png` | 36x36 normalized confusion matrix |
| `gradcam.png` | GradCAM saliency heatmaps (sample images) |
| `misclassifications.png` | Top misclassified image pairs |
| `training_curves.png` | Loss and accuracy across epochs |

---

## 7. Single-Image Inference (CLI)

Run inference on a single image directly from the terminal:

```bash
python -m src.predict --image path/to/image.jpg --top_k 5
```

**Example output:**

```
Predictions for: data/test/banana/Image_1.jpg
----------------------------------------
  1. banana                97.6%
  2. garlic                 0.3%
  3. chilli pepper          0.2%
  4. mango                  0.1%
  5. lemon                  0.1%
```

**Options:**

| Argument | Default | Description |
|---|---|---|
| `--image` | required | Path to input image file |
| `--checkpoint` | `results/best_model.pth` | Path to model checkpoint |
| `--top_k` | `5` | Number of top predictions to display |
| `--image_size` | `224` | Input resolution (must match training) |

---

## 8. Streamlit Web Application

Launch the interactive web interface locally:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Features:
- Drag-and-drop or browse image upload (JPG, PNG, WEBP supported)
- Top-5 predictions displayed with confidence progress bars
- Sidebar shows model architecture summary

---

## 9. Deployment to Streamlit Community Cloud

The repository is ready for zero-cost cloud deployment:

1. Push the repository to GitHub (ensure `results/best_model.pth` is committed or uploaded as a GitHub Release asset).
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New app**, select your repository, set the branch to `main`, and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit will install dependencies from `requirements.txt` automatically.

> **Note on model weights:** The `.gitignore` excludes `*.pth` by default. To deploy, either commit the weights directly (if size permits), upload to GitHub Releases and add a download step in `app.py`, or host on Hugging Face Hub and load via the `huggingface_hub` library.

---

## 10. Project Reset

To retrain from scratch:

```bash
# Remove existing checkpoint and results
del results\best_model.pth
del results\*.png
del results\classification_report.csv

# Retrain
python -m src.train
python -m src.evaluate
```
