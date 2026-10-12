"""
src/gradcam.py
--------------
Native PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Extracts class activation heatmaps directly from the convolutional backbone
without external dependencies, using robust module hooks with guaranteed cleanup.
"""

from typing import Tuple
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from src.dataset import get_transforms
from src.model import ProduceClassifier


def generate_gradcam(
    image: Image.Image,
    model: ProduceClassifier,
    target_class_idx: int,
    device: torch.device,
    image_size: int = 224,
    colormap_name: str = "JET",
    alpha: float = 0.5,
) -> Tuple[Image.Image, Image.Image, Image.Image, np.ndarray]:
    """Generate Grad-CAM heatmaps for an input PIL image.

    Args:
        image: Input PIL image.
        model: Trained ProduceClassifier model.
        target_class_idx: Class index to compute activations for.
        device: Device to run computation on (CPU or CUDA).
        image_size: Square dimension expected by the network (default 224).
        colormap_name: OpenCV colormap name (JET, VIRIDIS, TURBO, PLASMA).
        alpha: Blending weight between original image and heatmap overlay (0.0 - 1.0).

    Returns:
        Tuple containing:
            - original_processed: PIL Image of the cropped/preprocessed input
            - overlay_image: PIL Image with heatmap blended over the input
            - filter_only_image: PIL Image of the raw colormapped heatmap
            - cam_normalized: 2D numpy array [H, W] with values in [0.0, 1.0]
    """
    model.eval()

    target_layer = model.features[-1]

    # Clear any stale hooks
    target_layer._forward_hooks.clear()
    target_layer._backward_hooks.clear()

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h_forward = target_layer.register_forward_hook(forward_hook)
    h_backward = target_layer.register_full_backward_hook(backward_hook)

    try:
        with torch.enable_grad():
            for param in model.parameters():
                param.requires_grad = True

            transform = get_transforms("test", image_size)
            rgb_img = image.convert("RGB")
            input_tensor = transform(rgb_img).unsqueeze(0).to(device)

            model.zero_grad()
            logits = model(input_tensor)

            target_score = logits[0, target_class_idx]
            target_score.backward()

    finally:
        # Guarantee hook removal under all circumstances
        h_forward.remove()
        h_backward.remove()
        model.zero_grad()
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM failed to capture activations or gradients.")

    act = activations[0]  # shape: [1, 1280, 7, 7]
    grad = gradients[0]   # shape: [1, 1280, 7, 7]

    # Global Average Pooling over spatial dimensions (H, W) to obtain channel weights
    weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, 1280, 1, 1]

    # Weighted linear combination of activation maps
    cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, 7, 7]
    cam = F.relu(cam)  # ReLU: only positive contributions matter

    # Bilinear upsample to input resolution
    cam = F.interpolate(cam, size=(image_size, image_size), mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()

    # Min-max normalization
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    else:
        cam_norm = np.zeros_like(cam)

    # Reconstruct input image as RGB float array in [0, 1] matching model's crop
    inv_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    inv_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_np = input_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * inv_std + inv_mean).clip(0.0, 1.0)
    img_uint8 = np.uint8(255 * img_np)

    # Select colormap
    cv2_colormaps = {
        "JET": cv2.COLORMAP_JET,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
        "TURBO": cv2.COLORMAP_TURBO,
        "PLASMA": cv2.COLORMAP_PLASMA,
        "INFERNO": cv2.COLORMAP_INFERNO,
    }
    cmap_code = cv2_colormaps.get(colormap_name.upper(), cv2.COLORMAP_JET)

    # Generate colored heatmap (Filter only)
    cam_uint8 = np.uint8(255 * cam_norm)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cmap_code)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Generate blended overlay: (1 - alpha) * original + alpha * heatmap
    overlay_rgb = np.uint8((1.0 - alpha) * img_uint8 + alpha * heatmap_rgb)

    original_processed = Image.fromarray(img_uint8)
    overlay_image = Image.fromarray(overlay_rgb)
    filter_only_image = Image.fromarray(heatmap_rgb)

    return original_processed, overlay_image, filter_only_image, cam_norm
