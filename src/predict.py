"""
src/predict.py
--------------
Single-image inference utility.

Used by app.py (Streamlit) and can also be run from the command line:

    python -m src.predict --image path/to/image.jpg --top_k 3
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.dataset import get_transforms
from src.model import ProduceClassifier


# ── Model loader (cached for Streamlit) ──────────────────────────────────────

_LOADED_MODEL: Dict[str, object] = {}   # Simple module-level cache


def load_model(
    checkpoint_path: str | Path,
    num_classes: int = 36,
    dropout: float = 0.3,
    device: str | torch.device | None = None,
) -> Tuple[ProduceClassifier, Dict[str, int], Dict[int, str], torch.device]:
    """Load a trained model from a checkpoint file.

    Results are cached in-process so Streamlit doesn't reload on every run.

    Args:
        checkpoint_path: Path to the .pth file saved by train.py.
        num_classes:     Must match the trained model (default 36).
        dropout:         Must match the trained model.
        device:          Target device; auto-detects if None.

    Returns:
        Tuple of (model, class_to_idx, idx_to_class, device).
    """
    key = str(checkpoint_path)
    if key in _LOADED_MODEL:
        return _LOADED_MODEL[key]  # type: ignore[return-value]

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Build model skeleton
    model = ProduceClassifier(num_classes=num_classes, dropout=dropout)

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    _LOADED_MODEL[key] = (model, class_to_idx, idx_to_class, device)
    return model, class_to_idx, idx_to_class, device


# ── Core inference function ───────────────────────────────────────────────────

@torch.no_grad()
def predict(
    image: Image.Image,
    model: ProduceClassifier,
    idx_to_class: Dict[int, str],
    device: torch.device,
    top_k: int = 3,
    image_size: int = 224,
) -> List[Tuple[str, float]]:
    """Run inference on a single PIL image.

    Args:
        image:        Input PIL image (any mode; will be converted to RGB).
        model:        Loaded ProduceClassifier in eval mode.
        idx_to_class: Index → class name mapping from the checkpoint.
        device:       Inference device.
        top_k:        Number of top predictions to return.
        image_size:   Must match the size used during training (224).

    Returns:
        List of (class_name, probability) tuples, sorted by probability desc.
    """
    transform = get_transforms("test", image_size)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    logits = model(tensor)                         # [1, num_classes]
    probs  = F.softmax(logits, dim=1).squeeze(0)  # [num_classes]

    top_k  = min(top_k, probs.size(0))
    values, indices = torch.topk(probs, top_k)

    return [
        (idx_to_class[idx.item()], val.item())
        for val, idx in zip(values, indices)
    ]


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fruit/vegetable classification on a single image."
    )
    parser.add_argument("--image",      required=True,  help="Path to image file")
    parser.add_argument("--checkpoint", default="results/best_model.pth",
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--top_k",      type=int, default=3,
                        help="Number of top predictions to display")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    model, class_to_idx, idx_to_class, device = load_model(
        checkpoint_path=args.checkpoint
    )

    image = Image.open(args.image)
    results = predict(image, model, idx_to_class, device, top_k=args.top_k)

    print(f"\n🍎 Predictions for: {args.image}")
    print("-" * 40)
    for rank, (label, prob) in enumerate(results, 1):
        bar = "█" * int(prob * 30)
        print(f"  {rank}. {label:<20} {prob*100:5.1f}%  {bar}")
