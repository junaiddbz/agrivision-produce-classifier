"""
src/train.py
------------
Two-phase training loop for the Fruit & Vegetable Classifier.

  Phase 1  (epochs_phase1)  — backbone frozen, head only, lr = lr_phase1
  Phase 2  (epochs_phase2)  — top 2 blocks unfrozen,   lr = lr_phase2

Features:
  - Early stopping on validation loss
  - Best-model checkpointing (saves best_model.pth)
  - Training curves saved to results/training_curves.png
  - Per-epoch console logging with tqdm progress bars
"""

import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.model import build_model, ProduceClassifier


# ── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Core training helpers ────────────────────────────────────────────────────

def train_one_epoch(
    model: ProduceClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Run one full training epoch.

    Returns:
        Tuple of (avg_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: ProduceClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> Tuple[float, float]:
    """Evaluate model on a data split.

    Returns:
        Tuple of (avg_loss, accuracy).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in tqdm(loader, desc=f"  [{split}]", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds    = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return running_loss / total, correct / total


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str | Path,
) -> None:
    """Save a 2-panel loss/accuracy plot to disk.

    Args:
        history:   Dict with keys 'train_loss', 'val_loss',
                   'train_acc', 'val_acc'.
        save_path: Where to save the PNG.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss panel
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val loss",   linewidth=2)
    ax1.set_title("Loss per Epoch", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy panel
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   "r-o", label="Val acc",   linewidth=2)
    ax2.set_title("Accuracy per Epoch", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training curves saved → {save_path}")


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience:  Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_epoch = epoch
            self.counter    = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ── Main training loop ────────────────────────────────────────────────────────

def train(cfg: Dict) -> None:
    """Full two-phase training run.

    Args:
        cfg: Parsed YAML config dict (from train_config.yaml).
    """
    set_seed(cfg["training"]["seed"])

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg["output"]["checkpoint_path"])

    # ── Data ─────────────────────────────────────────────────────────────────
    print("📂 Loading datasets...")
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        train_dir  = cfg["data"]["train_dir"],
        val_dir    = cfg["data"]["val_dir"],
        test_dir   = cfg["data"]["test_dir"],
        image_size = cfg["data"]["image_size"],
        batch_size = cfg["training"]["batch_size"],
        num_workers= cfg["data"]["num_workers"],
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")
    print(f"  Classes       : {len(class_to_idx)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, device = build_model(
        num_classes=cfg["model"]["num_classes"],
        dropout    =cfg["model"]["dropout"],
    )
    print(f"\n🧠 Model loaded on {device}")
    summary = model.param_summary()
    print(f"  Total params    : {summary['total']:,}")
    print(f"  Trainable (P1)  : {summary['trainable']:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }
    best_val_loss = float("inf")
    early_stop    = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])
    start_time    = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1 — Feature extraction (backbone frozen)
    # ─────────────────────────────────────────────────────────────────────────
    p1_epochs = cfg["training"]["epochs_phase1"]
    lr_p1     = cfg["training"]["lr_phase1"]
    print(f"\n{'='*60}")
    print(f"🔵 PHASE 1 — Head-only training ({p1_epochs} epochs, lr={lr_p1})")
    print(f"{'='*60}")

    model.freeze_backbone()
    optimizer = AdamW(model.trainable_params(), lr=lr_p1, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=p1_epochs, eta_min=lr_p1 * 0.1)

    for epoch in range(1, p1_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, p1_epochs
        )
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device, "val")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch {epoch:02d}/{p1_epochs} | "
            f"Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
            f"Val loss: {vl_loss:.4f} acc: {vl_acc:.3f}"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "class_to_idx": class_to_idx, "val_acc": vl_acc},
                ckpt_path,
            )
            print(f"  ✅ Best model saved (val_loss={vl_loss:.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 — Fine-tuning (top 2 blocks unfrozen)
    # ─────────────────────────────────────────────────────────────────────────
    p2_epochs = cfg["training"]["epochs_phase2"]
    lr_p2     = cfg["training"]["lr_phase2"]
    print(f"\n{'='*60}")
    print(f"🟢 PHASE 2 — Fine-tuning top blocks ({p2_epochs} epochs, lr={lr_p2})")
    print(f"{'='*60}")

    model.unfreeze_top_blocks(n_blocks=2)
    summary2 = model.param_summary()
    print(f"  Trainable params now: {summary2['trainable']:,}")

    optimizer = AdamW(model.trainable_params(), lr=lr_p2, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=p2_epochs, eta_min=lr_p2 * 0.1)
    early_stop = EarlyStopping(patience=cfg["training"]["early_stopping_patience"])

    for epoch in range(1, p2_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, p2_epochs
        )
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device, "val")
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch {epoch:02d}/{p2_epochs} | "
            f"Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
            f"Val loss: {vl_loss:.4f} acc: {vl_acc:.3f}"
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "class_to_idx": class_to_idx, "val_acc": vl_acc},
                ckpt_path,
            )
            print(f"  ✅ Best model saved (val_loss={vl_loss:.4f})")

        if early_stop(vl_loss, epoch):
            print(f"  ⏹️  Early stopping triggered (no improvement for {early_stop.patience} epochs)")
            break

    # ── Wrap-up ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total training time: {elapsed/60:.1f} min")
    print(f"📦 Best checkpoint   : {ckpt_path}  (val_loss={best_val_loss:.4f})")

    plot_training_curves(history, results_dir / "training_curves.png")

    # Final test set evaluation
    print("\n🧪 Evaluating on test set...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, "test")
    print(f"  Test loss : {test_loss:.4f}")
    print(f"  Test acc  : {test_acc:.4f}  ({test_acc*100:.2f}%)")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = yaml.safe_load(open("train_config.yaml"))
    train(cfg)
