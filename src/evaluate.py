"""
src/evaluate.py
---------------
Post-training evaluation utilities:
  - Full classification report (precision / recall / F1 per class)
  - 36x36 confusion matrix heatmap
  - Top-N misclassified samples grid
  - GradCAM saliency visualisation

All outputs are saved to results/ and can be committed to the repo.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ProduceDataset, get_transforms
from src.model import ProduceClassifier


# ── Inference helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: ProduceClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], List[float]]:
    """Run inference over an entire DataLoader.

    Returns:
        (all_true_labels, all_pred_labels, all_max_confidences)
    """
    model.eval()
    all_true, all_pred, all_conf = [], [], []

    for images, labels in tqdm(loader, desc="Running inference"):
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        all_true.extend(labels.tolist())
        all_pred.extend(pred.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())

    return all_true, all_pred, all_conf


# ── Classification report ────────────────────────────────────────────────────

def print_and_save_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str | Path,
) -> None:
    """Print and save a per-class precision/recall/F1 report as CSV."""
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print("\n" + report_str)

    import pandas as pd
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    df = pd.DataFrame(report_dict).T
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)
    print(f"📄 Classification report saved → {save_path}")


# ── Confusion matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str | Path,
    figsize: Tuple[int, int] = (20, 18),
) -> None:
    """Save a normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label",      fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix (Normalised)", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")


# ── Misclassification grid ────────────────────────────────────────────────────

def plot_misclassifications(
    dataset: ProduceDataset,
    y_true: List[int],
    y_pred: List[int],
    all_conf: List[float],
    idx_to_class: Dict[int, str],
    save_path: str | Path,
    n: int = 16,
) -> None:
    """Save a grid of the n most-confident misclassified images."""
    errors = [
        (i, conf)
        for i, (t, p, conf) in enumerate(zip(y_true, y_pred, all_conf))
        if t != p
    ]
    # Sort by confidence descending — most egregious mistakes
    errors.sort(key=lambda x: x[1], reverse=True)
    errors = errors[:n]

    cols = 4
    rows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).flatten()

    _inv_mean = np.array([0.485, 0.456, 0.406])
    _inv_std  = np.array([0.229, 0.224, 0.225])

    for ax, (idx, conf) in zip(axes, errors):
        img_tensor, _ = dataset[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * _inv_std + _inv_mean).clip(0, 1)
        ax.imshow(img_np)
        ax.set_title(
            f"True: {idx_to_class[y_true[idx]]}\n"
            f"Pred: {idx_to_class[y_pred[idx]]}\n"
            f"Conf: {conf:.2%}",
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes[len(errors):]:
        ax.axis("off")

    plt.suptitle("Top Misclassifications (by confidence)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🔍 Misclassification grid saved → {save_path}")


# ── GradCAM visualisation ─────────────────────────────────────────────────────

def plot_gradcam(
    model: ProduceClassifier,
    image_paths: List[str | Path],
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    device: torch.device,
    save_path: str | Path,
    image_size: int = 224,
) -> None:
    """Generate GradCAM heatmaps for a list of images.

    Targets the last convolutional block of EfficientNet-B0 (features[8]).

    Args:
        model:        Trained ProduceClassifier.
        image_paths:  List of image file paths to visualise.
        class_to_idx: Class name → index mapping.
        idx_to_class: Index → class name mapping.
        device:       Inference device.
        save_path:    Output PNG path.
        image_size:   Should match training resolution (224).
    """
    # Ensure parameters require gradients for GradCAM backprop
    for param in model.parameters():
        param.requires_grad = True

    target_layer = [model.features[-1]]  # Last conv block of EfficientNet-B0
    cam = GradCAM(model=model, target_layers=target_layer)

    transform = get_transforms("test", image_size)

    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 3.5))
    if n == 1:
        axes = [axes]

    _inv_mean = np.array([0.485, 0.456, 0.406])
    _inv_std  = np.array([0.229, 0.224, 0.225])

    model.eval()
    for i, img_path in enumerate(image_paths):
        pil_img    = Image.open(img_path).convert("RGB")
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits  = model(img_tensor)
            pred_idx = logits.argmax(dim=1).item()

        grayscale_cam = cam(
            input_tensor=img_tensor,
            targets=[ClassifierOutputTarget(pred_idx)],
        )[0]

        # Reconstruct RGB for overlay
        img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * _inv_std + _inv_mean).clip(0, 1).astype(np.float32)
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        axes[i][0].imshow(img_np)
        axes[i][0].set_title(f"Original\nPred: {idx_to_class[pred_idx]}", fontsize=9)
        axes[i][0].axis("off")

        axes[i][1].imshow(cam_image)
        axes[i][1].set_title("GradCAM Activation", fontsize=9)
        axes[i][1].axis("off")

    plt.suptitle("GradCAM — What the model looks at", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"🔥 GradCAM visualisation saved → {save_path}")


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def run_full_evaluation(
    model: ProduceClassifier,
    test_loader: DataLoader,
    test_dataset: ProduceDataset,
    device: torch.device,
    results_dir: str | Path = "results/",
    gradcam_samples: Optional[List[str]] = None,
) -> None:
    """Run the complete evaluation suite and save all outputs.

    Args:
        model:           Trained model with best weights loaded.
        test_loader:     DataLoader for the test split.
        test_dataset:    Raw ProduceDataset for the test split (for image access).
        device:          Inference device.
        results_dir:     Output directory for all saved files.
        gradcam_samples: List of image paths for GradCAM visualisation.
                         If None, samples are picked automatically.
    """
    results_dir = Path(results_dir)
    class_names  = test_dataset.classes
    idx_to_class = test_dataset.idx_to_class

    print("\n" + "="*60)
    print("🔬 FULL EVALUATION")
    print("="*60)

    y_true, y_pred, all_conf = get_predictions(model, test_loader, device)

    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"\n✅ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    print_and_save_report(
        y_true, y_pred, class_names,
        save_path=results_dir / "classification_report.csv",
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=results_dir / "confusion_matrix.png",
    )
    plot_misclassifications(
        test_dataset, y_true, y_pred, all_conf, idx_to_class,
        save_path=results_dir / "misclassifications.png",
    )

    # GradCAM: pick one image per class if no explicit list given
    if gradcam_samples is None:
        seen = set()
        gradcam_samples = []
        for path, label in test_dataset.samples:
            cls = idx_to_class[label]
            if cls not in seen:
                seen.add(cls)
                gradcam_samples.append(str(path))
                if len(gradcam_samples) >= 8:
                    break

    plot_gradcam(
        model, gradcam_samples,
        class_to_idx=test_dataset.class_to_idx,
        idx_to_class=idx_to_class,
        device=device,
        save_path=results_dir / "gradcam.png",
    )

    print(f"\n✅ All evaluation outputs saved to: {results_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    from src.dataset import build_dataloaders
    from src.model import build_model

    cfg = yaml.safe_load(open("train_config.yaml"))

    _, _, test_loader, class_to_idx = build_dataloaders(
        train_dir  = cfg["data"]["train_dir"],
        val_dir    = cfg["data"]["val_dir"],
        test_dir   = cfg["data"]["test_dir"],
        image_size = cfg["data"]["image_size"],
        batch_size = cfg["training"]["batch_size"],
        num_workers= cfg["data"]["num_workers"],
    )

    test_dataset = ProduceDataset(
        root_dir    = cfg["data"]["test_dir"],
        transform   = get_transforms("test", cfg["data"]["image_size"]),
        class_to_idx= class_to_idx,
    )

    model, device = build_model(
        num_classes=cfg["model"]["num_classes"],
        dropout    =cfg["model"]["dropout"],
    )
    ckpt = torch.load(cfg["output"]["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    run_full_evaluation(
        model, test_loader, test_dataset, device,
        results_dir=cfg["output"]["results_dir"],
    )
