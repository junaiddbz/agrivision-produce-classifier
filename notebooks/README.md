# Notebooks

This directory contains interactive Jupyter notebooks providing exploratory data analysis, dataset profiling, end-to-end model training, evaluation benchmarks, and Grad-CAM interpretability.

---

## Available Notebooks

| Notebook | Description | Open in Colab |
|---|---|---|
| [`01_exploration.ipynb`](01_exploration.ipynb) | **Exploratory Data Analysis (EDA):** Per-class sample counts, class balance checks, 36-produce image gallery, image resolution distributions, channel statistics, and data augmentation visualizer. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/junaiddbz/agrivision-produce-classifier/blob/main/notebooks/01_exploration.ipynb) |
| [`02_training_and_evaluation.ipynb`](02_training_and_evaluation.ipynb) | **End-to-End Walkthrough:** PyTorch DataLoader setup, EfficientNet-B0 transfer learning architecture, two-phase training loop, full test evaluation metrics, confusion matrix, Grad-CAM attention maps, and inference engine. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/junaiddbz/agrivision-produce-classifier/blob/main/notebooks/02_training_and_evaluation.ipynb) |

---

## Running Locally

To run the notebooks in your local virtual environment:

```bash
# Activate your environment
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install Jupyter if not already installed
pip install jupyterlab

# Launch JupyterLab
jupyter lab notebooks/
```

---

## Running on Google Colab

Both notebooks can be run directly in Google Colab with free GPU acceleration:
1. Click the **Open in Colab** badge next to the desired notebook above.
2. In Colab, navigate to **Runtime > Change runtime type** and select **T4 GPU**.
3. Run the cells sequentially.
