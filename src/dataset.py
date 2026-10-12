"""
src/dataset.py
--------------
Custom PyTorch Dataset and DataLoader factory for the
Fruit & Vegetable Classifier (36-class image classification).

Dataset layout expected (mirrors the Kaggle download):
    <root>/
        train/
            apple/  *.jpg
            banana/ *.jpg
            ...
        validation/
            apple/  *.jpg
            ...
        test/
            apple/  *.jpg
            ...
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# ── ImageNet normalisation stats ────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_transforms(split: str, image_size: int = 224) -> transforms.Compose:
    """Return the appropriate torchvision transform pipeline for a given split.

    Args:
        split:      One of ``'train'``, ``'val'``, or ``'test'``.
        image_size: Target square crop size (default 224 for EfficientNet).

    Returns:
        A ``transforms.Compose`` pipeline ready to be passed to a Dataset.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
    else:
        # Consistent deterministic pipeline for val / test
        return transforms.Compose([
            transforms.Resize(image_size + 32),          # 256 for size=224
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])


class ProduceDataset(Dataset):
    """ImageFolder-style dataset for the Fruits & Vegetables classification task.

    Supports lazy loading — images are read from disk only when ``__getitem__``
    is called, keeping RAM usage flat regardless of dataset size.

    Args:
        root_dir:   Path to the split directory (e.g. ``data/train``).
        transform:  A torchvision transform pipeline.  If ``None``, only a
                    ``ToTensor`` conversion is applied.
        class_to_idx: Optional pre-built ``{class_name: int}`` mapping.  When
                    provided all splits share the same label space (recommended).
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.ToTensor()

        # Discover classes from subdirectory names, sorted for reproducibility
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            classes = sorted(
                [d.name for d in self.root_dir.iterdir() if d.is_dir()]
            )
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        self.idx_to_class: Dict[int, str] = {
            v: k for k, v in self.class_to_idx.items()
        }
        self.classes: List[str] = list(self.class_to_idx.keys())

        # Build flat list of (image_path, label_index) pairs
        self.samples: List[Tuple[Path, int]] = []
        _image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for class_name, label in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in _image_exts:
                    self.samples.append((img_path, label))

    # ── Core Dataset interface ───────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")  # Ensure 3-channel
        if self.transform:
            image = self.transform(image)
        return image, label

    # ── Convenience helpers ──────────────────────────────────────────────────

    def get_class_counts(self) -> Dict[str, int]:
        """Return a dict mapping class name -> number of samples."""
        counts: Dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[self.idx_to_class[label]] += 1
        return counts

    def __repr__(self) -> str:
        return (
            f"ProduceDataset("
            f"root='{self.root_dir}', "
            f"num_classes={len(self.classes)}, "
            f"num_samples={len(self.samples)})"
        )


# ── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Create DataLoaders for all three splits.

    All splits share the same ``class_to_idx`` mapping derived from the
    training directory, so label integers are consistent across splits.

    Args:
        train_dir:   Path to training split root.
        val_dir:     Path to validation split root.
        test_dir:    Path to test split root.
        image_size:  Square crop size fed to the network.
        batch_size:  Number of samples per batch.
        num_workers: Subprocess workers for parallel data loading.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader, class_to_idx)``.
    """
    # Build train dataset first to establish the shared label mapping
    train_dataset = ProduceDataset(
        root_dir=train_dir,
        transform=get_transforms("train", image_size),
    )
    shared_class_to_idx = train_dataset.class_to_idx

    val_dataset = ProduceDataset(
        root_dir=val_dir,
        transform=get_transforms("val", image_size),
        class_to_idx=shared_class_to_idx,
    )
    test_dataset = ProduceDataset(
        root_dir=test_dir,
        transform=get_transforms("test", image_size),
        class_to_idx=shared_class_to_idx,
    )

    _common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, **_common
    )
    val_loader   = DataLoader(val_dataset,  shuffle=False, **_common)
    test_loader  = DataLoader(test_dataset, shuffle=False, **_common)

    return train_loader, val_loader, test_loader, shared_class_to_idx


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load(open("train_config.yaml"))

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        train_dir=cfg["data"]["train_dir"],
        val_dir=cfg["data"]["val_dir"],
        test_dir=cfg["data"]["test_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    imgs, labels = next(iter(train_loader))
    print(f"Batch shape : {imgs.shape}")         # [32, 3, 224, 224]
    print(f"Label range : {labels.min()}–{labels.max()}")
    print(f"Num classes : {len(class_to_idx)}")  # 36
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")
