"""
src/model.py
------------
EfficientNet-B0 backbone with a custom classification head for
36-class fruit & vegetable recognition.

Two-phase training strategy:
  Phase 1 — backbone frozen, only the head trains (fast convergence)
  Phase 2 — last 2 EfficientNet blocks + head fine-tuned at lower LR
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


class ProduceClassifier(nn.Module):
    """EfficientNet-B0 transfer-learning model for produce classification.

    Args:
        num_classes: Number of output categories (default 36).
        dropout:     Dropout rate applied before the final linear layer.
        pretrained:  Load ImageNet weights when True (always True in practice).
    """

    def __init__(
        self,
        num_classes: int = 36,
        dropout: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # ── Feature extractor (all layers except the original classifier) ──
        self.features = backbone.features
        self.avgpool  = backbone.avgpool       # AdaptiveAvgPool2d → [B, 1280, 1, 1]

        # ── Custom classification head ──────────────────────────────────────
        in_features = backbone.classifier[1].in_features  # 1280 for B0
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    # ── Phase control helpers ────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """Phase 1: freeze all feature layers, train head only."""
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True

    def unfreeze_top_blocks(self, n_blocks: int = 2) -> None:
        """Phase 2: unfreeze the last *n_blocks* EfficientNet blocks + BN.

        EfficientNet-B0 has 9 feature blocks (indices 0–8).
        Unfreezing the last 2 (indices 7 & 8) gives a good trade-off between
        adaptation and overfitting risk.
        """
        # First ensure everything is frozen
        self.freeze_backbone()

        # Unfreeze the final n_blocks feature blocks
        total_blocks = len(self.features)
        for i in range(total_blocks - n_blocks, total_blocks):
            for param in self.features[i].parameters():
                param.requires_grad = True

        # Always keep classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    def trainable_params(self) -> List[nn.Parameter]:
        """Return only the parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def param_summary(self) -> Dict[str, int]:
        """Return total vs trainable parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ── Factory function ─────────────────────────────────────────────────────────

def build_model(
    num_classes: int = 36,
    dropout: float = 0.3,
    device: str | torch.device | None = None,
) -> Tuple["ProduceClassifier", torch.device]:
    """Instantiate model, move to device, freeze backbone for Phase 1.

    Args:
        num_classes: Number of output classes.
        dropout:     Dropout probability in the classifier head.
        device:      Target device. Auto-detects CUDA/MPS/CPU if None.

    Returns:
        Tuple of (model, device).
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    model = ProduceClassifier(num_classes=num_classes, dropout=dropout)
    model.freeze_backbone()   # Start in Phase 1 (head-only training)
    model = model.to(device)

    return model, device


# ── Quick sanity check ───────────────────────────────────────────────────────
if __name__ == "__main__":
    model, device = build_model(num_classes=36)

    # Print parameter summary
    summary = model.param_summary()
    print(f"Device     : {device}")
    print(f"Total params    : {summary['total']:,}")
    print(f"Trainable params: {summary['trainable']:,}  (Phase 1 — head only)")
    print(f"Frozen params   : {summary['frozen']:,}")

    # Forward pass smoke test
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")  # [4, 36]
    assert out.shape == (4, 36), "Shape mismatch!"
    print("[OK] Model sanity check passed.")

    # Phase 2 check
    model.unfreeze_top_blocks(n_blocks=2)
    summary2 = model.param_summary()
    print(f"\nAfter unfreeze_top_blocks(2):")
    print(f"Trainable params: {summary2['trainable']:,}")
