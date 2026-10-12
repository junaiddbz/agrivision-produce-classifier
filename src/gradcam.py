"""
src/gradcam.py
--------------
Native PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Extracts class activation heatmaps directly from the convolutional backbone
without external dependencies or garbage-collection warnings.
"""

from typing import Optional, Tuple
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

    # Enable gradients across model parameters for backward hook
    for param in model.parameters():
        param.requires_grad = True

    # Hooks to capture activations and gradients on the last convolutional block
    activations = []
    gradients = []

    def forward_hook(module, input_tensor, output_tensor):
        activations.append(output_tensor)
        output_tensor.register_hook(lambda grad: gradients.append(grad))

    # Target the last feature block of EfficientNet-B0 (features[-1])
    hook_handle = model.features[-1].register_forward_hook(forward_hook)

    # Transform image for model input
    transform = get_transforms("test", image_size)
    rgb_img = image.convert("RGB")
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)

    # Clear gradients and run forward pass
    model.zero_grad()
    logits = model(input_tensor)

    # Compute target score and backpropagate
    target_score = logits[0, target_class_idx]
    target_score.backward()

    # Detach and remove hook immediately
    hook_handle.remove()

    # Extract feature activations and gradients
    act = activations[0]  # shape: [1, 1280, 7, 7]
    grad = gradients[0]   # shape: [1, 1280, 7, 7]

    # Global Average Pooling over spatial dimensions (H, W) to obtain channel weights
    weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, 1280, 1, 1]

    # Weighted linear combination of activation maps
    cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, 7, 7]
    cam = F.relu(cam)  # ReLU: only positive contributions matter

    # Bilinear upsample to input resolution
    cam = F.interpolate(cam, size=(image_size, image_size), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()

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
