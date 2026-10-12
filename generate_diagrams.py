"""
generate_diagrams.py
--------------------
Generates high-resolution, publication-quality architecture image diagrams
for docs/architecture.md and README.md.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. End-to-End Model Architecture Diagram ──────────────────────────────────
def render_model_architecture():
    fig, ax = plt.subplots(figsize=(18, 9), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Title & Subtitle
    ax.text(9, 8.4, "Produce Classifier: EfficientNet-B0 End-to-End Architecture", 
            ha="center", va="center", fontsize=18, fontweight="bold", color="#1E293B", fontfamily="sans-serif")
    ax.text(9, 8.0, "36-Class Fruit & Vegetable Image Recognition System | 4.05M Parameters | Input: 224 x 224 x 3", 
            ha="center", va="center", fontsize=11, color="#64748B", fontfamily="sans-serif")

    # Stages definition: (name, subtext, tensor_shape, color, x_pos, y_pos, width, height)
    boxes = [
        # Input
        ("Input Image", "Raw Produce Photo\nRGB Normalization", "3 x 224 x 224", "#F1F5F9", "#475569", 0.6, 4.0, 1.6, 2.2),
        # Stem
        ("Stem Conv", "Conv2d 3x3 (s=2)\nBN + SiLU", "32 x 112 x 112", "#E0E7FF", "#4338CA", 2.6, 4.0, 1.5, 2.2),
        # Stage 1
        ("Stage 1", "1x MBConv1\nk=3x3, s=1", "16 x 112 x 112", "#DBEAFE", "#1D4ED8", 4.4, 4.0, 1.4, 2.2),
        # Stage 2
        ("Stage 2", "2x MBConv6\nk=3x3, s=2", "24 x 56 x 56", "#DBEAFE", "#1D4ED8", 6.0, 4.0, 1.4, 2.2),
        # Stage 3
        ("Stage 3", "2x MBConv6\nk=5x5, s=2", "40 x 28 x 28", "#DBEAFE", "#1D4ED8", 7.6, 4.0, 1.4, 2.2),
        # Stage 4
        ("Stage 4", "3x MBConv6\nk=3x3, s=2", "80 x 14 x 14", "#CCFBF1", "#0F766E", 9.2, 4.0, 1.4, 2.2),
        # Stage 5
        ("Stage 5", "3x MBConv6\nk=5x5, s=1", "112 x 14 x 14", "#CCFBF1", "#0F766E", 10.8, 4.0, 1.4, 2.2),
        # Stage 6
        ("Stage 6", "4x MBConv6\nk=5x5, s=2", "192 x 7 x 7", "#FEF3C7", "#B45309", 12.4, 4.0, 1.4, 2.2),
        # Stage 7
        ("Stage 7", "1x MBConv6\nk=3x3, s=1", "320 x 7 x 7", "#FEF3C7", "#B45309", 14.0, 4.0, 1.4, 2.2),
        # Head Conv
        ("Stage 8", "Conv1x1\nBN + SiLU", "1280 x 7 x 7", "#FCE7F3", "#BE185D", 15.6, 4.0, 1.4, 2.2),
    ]

    # Draw feature extractor background card
    backbone_box = FancyBboxPatch((2.4, 3.4), 14.8, 3.4, boxstyle="round,pad=0.2,rounding_size=0.15",
                                  facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5, linestyle="--")
    ax.add_patch(backbone_box)
    ax.text(9.8, 6.9, "EfficientNet-B0 Backbone (4,007,548 Parameters)", 
            ha="center", va="center", fontsize=11, fontweight="bold", color="#334155")

    # Render boxes and arrows
    for i, (title, desc, shape, bg, border, x, y, w, h) in enumerate(boxes):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.1",
                               facecolor=bg, edgecolor=border, linewidth=1.8)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", fontsize=10, fontweight="bold", color=border)
        ax.text(x + w/2, y + h/2, desc, ha="center", va="center", fontsize=8.2, color="#334155")
        
        # Shape badge at bottom
        shape_patch = FancyBboxPatch((x + 0.05, y + 0.1), w - 0.1, 0.45, boxstyle="round,pad=0.04,rounding_size=0.06",
                                    facecolor="#FFFFFF", edgecolor=border, linewidth=0.8)
        ax.add_patch(shape_patch)
        ax.text(x + w/2, y + 0.32, shape, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1E293B")

        # Arrow to next
        if i < len(boxes) - 1:
            next_x = boxes[i+1][5]
            arr = FancyArrowPatch((x + w, y + h/2), (next_x, y + h/2),
                                  arrowstyle="->", mutation_scale=12, linewidth=1.6, color="#94A3B8")
            ax.add_patch(arr)

    # Classification Head Blocks below
    # Down arrow from Stage 8 to Classifier Head
    arr_down = FancyArrowPatch((16.3, 4.0), (16.3, 2.6),
                               arrowstyle="->", mutation_scale=14, linewidth=2.0, color="#BE185D")
    ax.add_patch(arr_down)

    head_cards = [
        ("AdaptiveAvgPool2d", "Global Spatial Pooling\n(7x7 -> 1x1)", "1280", "#F1F5F9", "#475569", 14.8, 1.2, 2.6, 1.2),
        ("Dropout (p=0.3)", "Regularization to\nprevent overfitting", "1280", "#FEF2F2", "#DC2626", 11.4, 1.2, 2.6, 1.2),
        ("Linear Classifier", "Fully Connected Layer\nWeights: 1280 x 36", "36 Logits", "#ECFDF5", "#059669", 8.0, 1.2, 2.6, 1.2),
        ("Softmax Output", "Class Probabilities &\nTop-5 Predictions", "36 Classes", "#EFF6FF", "#2563EB", 4.6, 1.2, 2.6, 1.2),
    ]

    # Draw classifier head card
    head_bg = FancyBboxPatch((4.2, 0.6), 13.5, 2.0, boxstyle="round,pad=0.2,rounding_size=0.15",
                             facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=1.5, linestyle="--")
    ax.add_patch(head_bg)
    ax.text(10.95, 0.45, "Custom Classification Head (46,116 Parameters)", 
            ha="center", va="center", fontsize=11, fontweight="bold", color="#166534")

    for i, (title, desc, shape, bg, border, x, y, w, h) in enumerate(head_cards):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
                               facecolor=bg, edgecolor=border, linewidth=1.6)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h - 0.28, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=border)
        ax.text(x + w/2, y + 0.55, desc, ha="center", va="center", fontsize=7.8, color="#334155")
        ax.text(x + w/2, y + 0.2, f"Shape: {shape}", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0F172A")

        if i < len(head_cards) - 1:
            prev_x = head_cards[i+1][5] + head_cards[i+1][6]
            arr_left = FancyArrowPatch((x, y + h/2), (prev_x, y + h/2),
                                       arrowstyle="->", mutation_scale=12, linewidth=1.6, color="#059669")
            ax.add_patch(arr_left)

    # Legend / Annotation Key
    ax.text(1.0, 2.4, "Color Key:", fontsize=9.5, fontweight="bold", color="#334155")
    legend_items = [
        ("Stem & Lower Blocks (Frozen)", "#DBEAFE", "#1D4ED8"),
        ("Mid Features (General Shapes)", "#CCFBF1", "#0F766E"),
        ("Upper Blocks (Unfrozen Phase 2)", "#FEF3C7", "#B45309"),
        ("Classifier Head (Trainable)", "#ECFDF5", "#059669"),
    ]
    for idx, (lbl, lbg, lbrd) in enumerate(legend_items):
        ly = 2.0 - idx * 0.4
        lp = FancyBboxPatch((1.0, ly - 0.08), 0.35, 0.24, boxstyle="square,pad=0.02",
                            facecolor=lbg, edgecolor=lbrd, linewidth=1)
        ax.add_patch(lp)
        ax.text(1.5, ly + 0.04, lbl, fontsize=8.5, va="center", color="#334155")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "model_architecture.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# ── 2. MBConv Inverted Residual Block Diagram ──────────────────────────────────
def render_mbconv_block():
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(7.5, 7.5, "MBConv Inverted Residual Block with Squeeze-and-Excitation (SE)", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1E293B")
    ax.text(7.5, 7.1, "Core Computational Unit of EfficientNet-B0 | Expansion t=6 | Depthwise Conv | Channel Attention", 
            ha="center", va="center", fontsize=10.5, color="#64748B")

    # Main stream blocks
    blocks = [
        ("Input Tensor", "Channels: C_in\nSpatial: H x W", 0.8, 3.2, 1.8, 1.6, "#F1F5F9", "#475569"),
        ("1x1 Expansion", "Pointwise Conv2d\nC_in -> t * C_in (exp=6)\nBatchNorm + SiLU", 3.2, 3.2, 2.2, 1.6, "#E0E7FF", "#4338CA"),
        ("Depthwise Conv", "Depthwise 3x3 / 5x5\nStride = 1 or 2\nBatchNorm + SiLU", 6.0, 3.2, 2.2, 1.6, "#DBEAFE", "#1D4ED8"),
        ("1x1 Projection", "Linear Pointwise\nt * C_in -> C_out\nBatchNorm (No Act)", 9.2, 3.2, 2.2, 1.6, "#FCE7F3", "#BE185D"),
        ("Output Tensor", "Channels: C_out\nSpatial: H' x W'", 12.4, 3.2, 1.8, 1.6, "#F1F5F9", "#475569"),
    ]

    for title, desc, x, y, w, h, bg, border in blocks:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.1",
                               facecolor=bg, edgecolor=border, linewidth=1.6)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h - 0.3, title, ha="center", va="center", fontsize=10, fontweight="bold", color=border)
        ax.text(x + w/2, y + (h - 0.3)/2, desc, ha="center", va="center", fontsize=8.2, color="#334155")

    # Connecting arrows for main stream
    ax.add_patch(FancyArrowPatch((2.6, 4.0), (3.2, 4.0), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#64748B"))
    ax.add_patch(FancyArrowPatch((5.4, 4.0), (6.0, 4.0), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#64748B"))
    ax.add_patch(FancyArrowPatch((8.2, 4.0), (9.2, 4.0), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#64748B"))
    ax.add_patch(FancyArrowPatch((11.4, 4.0), (12.4, 4.0), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#64748B"))

    # Squeeze-and-Excitation Loop above Depthwise Conv
    se_bg = FancyBboxPatch((5.7, 5.0), 5.4, 1.7, boxstyle="round,pad=0.15,rounding_size=0.12",
                           facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.4, linestyle="--")
    ax.add_patch(se_bg)
    ax.text(8.4, 6.45, "Squeeze-and-Excitation (SE) Channel Attention Loop", ha="center", fontsize=9.5, fontweight="bold", color="#92400E")

    se_steps = [
        ("Global Pooling", "Squeeze: 1x1xC", 6.0, 5.2, 1.4, 0.95),
        ("FC Reduction", "C -> C / 4 + SiLU", 7.6, 5.2, 1.5, 0.95),
        ("FC Expand & Sigmoid", "C/4 -> C (Weights s)", 9.3, 5.2, 1.6, 0.95),
    ]
    for stitle, sdesc, sx, sy, sw, sh in se_steps:
        sp = FancyBboxPatch((sx, sy), sw, sh, boxstyle="round,pad=0.06,rounding_size=0.06",
                            facecolor="#FFFBEB", edgecolor="#B45309", linewidth=1.2)
        ax.add_patch(sp)
        ax.text(sx + sw/2, sy + sh - 0.25, stitle, ha="center", va="center", fontsize=8.2, fontweight="bold", color="#78350F")
        ax.text(sx + sw/2, sy + 0.3, sdesc, ha="center", va="center", fontsize=7.2, color="#451A03")

    ax.add_patch(FancyArrowPatch((7.4, 5.65), (7.6, 5.65), arrowstyle="->", mutation_scale=10, linewidth=1.2, color="#B45309"))
    ax.add_patch(FancyArrowPatch((9.1, 5.65), (9.3, 5.65), arrowstyle="->", mutation_scale=10, linewidth=1.2, color="#B45309"))

    # Branch up to SE from Depthwise and back down
    ax.add_patch(FancyArrowPatch((7.1, 4.8), (6.7, 5.2), arrowstyle="->", mutation_scale=10, linewidth=1.4, color="#D97706"))
    ax.add_patch(FancyArrowPatch((10.1, 5.2), (9.7, 4.8), arrowstyle="->", mutation_scale=10, linewidth=1.4, color="#D97706"))
    ax.text(10.6, 4.9, "Scale: X * s", fontsize=8, fontweight="bold", color="#92400E")

    # Residual Skip Connection below
    ax.plot([1.7, 1.7, 11.9, 11.9], [3.2, 1.8, 1.8, 3.2], color="#059669", linewidth=2.0, linestyle="-")
    arr_add = FancyArrowPatch((11.9, 1.8), (11.9, 3.2), arrowstyle="->", mutation_scale=12, linewidth=2.0, color="#059669")
    ax.add_patch(arr_add)

    # Circle add icon
    add_circle = plt.Circle((11.9, 4.0), 0.28, facecolor="#DCFCE7", edgecolor="#059669", linewidth=1.8, zorder=5)
    ax.add_patch(add_circle)
    ax.text(11.9, 4.0, "+", ha="center", va="center", fontsize=14, fontweight="bold", color="#059669", zorder=6)

    ax.text(6.8, 1.5, "Residual Skip Connection (Active when Stride == 1 and C_in == C_out)", 
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#059669")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "mbconv_block.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# ── 3. Two-Phase Transfer Learning Diagram ────────────────────────────────────
def render_two_phase_training():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)
    fig.suptitle("Two-Phase Transfer Learning Strategy", fontsize=17, fontweight="bold", color="#1E293B", y=0.98)

    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # Phase 1: Feature Extraction
    ax1.set_title("Phase 1: Feature Extraction (Epochs 1 - 5)", fontsize=13, fontweight="bold", color="#1D4ED8", pad=15)
    ax1_bg = FancyBboxPatch((0.5, 0.5), 9.0, 8.8, boxstyle="round,pad=0.2,rounding_size=0.15",
                            facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5)
    ax1.add_patch(ax1_bg)

    # Frozen backbone
    frozen_box = FancyBboxPatch((1.2, 4.8), 7.6, 3.8, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor="#E2E8F0", edgecolor="#64748B", linewidth=2.0)
    ax1.add_patch(frozen_box)
    ax1.text(5.0, 7.9, "EfficientNet-B0 Backbone (Stages 0 - 8)", ha="center", fontsize=11, fontweight="bold", color="#334155")
    ax1.text(5.0, 7.4, "4,007,548 Parameters", ha="center", fontsize=9.5, color="#64748B")
    ax1.text(5.0, 6.3, "STATUS: FROZEN\nGradients Disabled (requires_grad = False)", 
             ha="center", va="center", fontsize=11, fontweight="bold", color="#DC2626")
    ax1.text(5.0, 5.3, "Preserves general low/mid-level ImageNet features", ha="center", fontsize=9, color="#475569")

    # Connecting arrow
    ax1.add_patch(FancyArrowPatch((5.0, 4.8), (5.0, 3.8), arrowstyle="->", mutation_scale=14, linewidth=2.0, color="#2563EB"))

    # Trainable head
    head_box = FancyBboxPatch((1.2, 1.4), 7.6, 2.4, boxstyle="round,pad=0.15,rounding_size=0.1",
                              facecolor="#DCFCE7", edgecolor="#16A34A", linewidth=2.0)
    ax1.add_patch(head_box)
    ax1.text(5.0, 3.2, "Classification Head (Dropout + Linear 1280 -> 36)", ha="center", fontsize=11, fontweight="bold", color="#15803D")
    ax1.text(5.0, 2.7, "46,116 Trainable Parameters (1.14% of model)", ha="center", fontsize=9.5, color="#166534")
    ax1.text(5.0, 2.0, "Optimizer: AdamW | Learning Rate: 1e-3\nRapid convergence on task classification manifold", 
             ha="center", fontsize=9, color="#14532D")

    # Phase 2: Fine-Tuning
    ax2.set_title("Phase 2: Fine-Tuning (Epochs 6 - 15)", fontsize=13, fontweight="bold", color="#0F766E", pad=15)
    ax2_bg = FancyBboxPatch((0.5, 0.5), 9.0, 8.8, boxstyle="round,pad=0.2,rounding_size=0.15",
                            facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.5)
    ax2.add_patch(ax2_bg)

    # Stages 0-6 Frozen
    st06_box = FancyBboxPatch((1.2, 6.4), 7.6, 2.2, boxstyle="round,pad=0.15,rounding_size=0.1",
                              facecolor="#E2E8F0", edgecolor="#64748B", linewidth=1.8)
    ax2.add_patch(st06_box)
    ax2.text(5.0, 7.9, "Stages 0 - 6 (Lower Feature Extractor)", ha="center", fontsize=10.5, fontweight="bold", color="#334155")
    ax2.text(5.0, 7.2, "2,302,268 Parameters | STATUS: FROZEN", ha="center", fontsize=10, fontweight="bold", color="#DC2626")
    ax2.text(5.0, 6.7, "Prevents distortion of core edge and texture primitives", ha="center", fontsize=8.5, color="#475569")

    # Stages 7-8 Unfrozen
    ax2.add_patch(FancyArrowPatch((5.0, 6.4), (5.0, 5.6), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#0F766E"))

    st78_box = FancyBboxPatch((1.2, 3.6), 7.6, 2.0, boxstyle="round,pad=0.15,rounding_size=0.1",
                              facecolor="#FEF3C7", edgecolor="#D97706", linewidth=1.8)
    ax2.add_patch(st78_box)
    ax2.text(5.0, 4.95, "Stages 7 & 8 (Top 2 Feature Blocks)", ha="center", fontsize=10.5, fontweight="bold", color="#92400E")
    ax2.text(5.0, 4.35, "1,705,280 Parameters | STATUS: UNFROZEN", ha="center", fontsize=10, fontweight="bold", color="#15803D")
    ax2.text(5.0, 3.85, "Adapts high-level produce representations (stems, seeds, rind)", ha="center", fontsize=8.5, color="#78350F")

    # Head in Phase 2
    ax2.add_patch(FancyArrowPatch((5.0, 3.6), (5.0, 2.8), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#0F766E"))

    h2_box = FancyBboxPatch((1.2, 1.0), 7.6, 1.8, boxstyle="round,pad=0.15,rounding_size=0.1",
                            facecolor="#DCFCE7", edgecolor="#16A34A", linewidth=1.8)
    ax2.add_patch(h2_box)
    ax2.text(5.0, 2.25, "Classification Head (46,116 Parameters)", ha="center", fontsize=10.5, fontweight="bold", color="#15803D")
    ax2.text(5.0, 1.7, "STATUS: UNFROZEN | lr = 1e-4 with Cosine Annealing Decay", ha="center", fontsize=9.5, fontweight="bold", color="#166534")
    ax2.text(5.0, 1.25, "Total Trainable in Phase 2: 1,751,396 Parameters (43.2%)", ha="center", fontsize=8.5, color="#14532D")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "transfer_learning_phases.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


# ── 4. Grad-CAM Workflow Diagram ──────────────────────────────────────────────
def render_gradcam_workflow():
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(8.0, 6.5, "Grad-CAM (Gradient-Weighted Class Activation Mapping) Dataflow", 
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1E293B")
    ax.text(8.0, 6.1, "Explainability Pipeline: Isolates regions driving class prediction by propagating class gradients", 
            ha="center", va="center", fontsize=10.5, color="#64748B")

    steps = [
        ("1. Input Image", "Preprocessed 224x224 crop\nCenter crop + ImageNet norm", 0.6, 2.4, 2.0, 2.4, "#F1F5F9", "#475569"),
        ("2. Forward Pass", "EfficientNet-B0 Forward Pass\nFeature Maps A^k [1280, 7, 7]\nat Stage 8 Conv", 3.2, 2.4, 2.4, 2.4, "#E0E7FF", "#4338CA"),
        ("3. Class Target Score", "Extract scalar logit y^c for\ntarget produce class c", 6.2, 2.4, 2.2, 2.4, "#FEF3C7", "#D97706"),
        ("4. Backpropagation", "Compute gradients:\nd(y^c) / d(A^k)\nGlobal Average Pool -> alpha_k", 9.0, 2.4, 2.5, 2.4, "#FCE7F3", "#BE185D"),
        ("5. Saliency Heatmap", "Linear combo: sum(alpha * A)\nReLU rectification\nBilinear upsample to 224x224", 12.1, 2.4, 3.2, 2.4, "#DCFCE7", "#16A34A"),
    ]

    for title, desc, x, y, w, h, bg, border in steps:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.1",
                               facecolor=bg, edgecolor=border, linewidth=1.6)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h - 0.4, title, ha="center", va="center", fontsize=10, fontweight="bold", color=border)
        ax.text(x + w/2, y + (h - 0.4)/2, desc, ha="center", va="center", fontsize=8.2, color="#334155")

    # Arrows between steps
    ax.add_patch(FancyArrowPatch((2.6, 3.6), (3.2, 3.6), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#64748B"))
    ax.add_patch(FancyArrowPatch((5.6, 3.6), (6.2, 3.6), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#64748B"))
    ax.add_patch(FancyArrowPatch((8.4, 3.6), (9.0, 3.6), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#64748B"))
    ax.add_patch(FancyArrowPatch((11.5, 3.6), (12.1, 3.6), arrowstyle="->", mutation_scale=12, linewidth=1.8, color="#64748B"))

    # Output Modes Box below
    modes_box = FancyBboxPatch((3.0, 0.4), 10.0, 1.5, boxstyle="round,pad=0.15,rounding_size=0.12",
                               facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.4, linestyle="--")
    ax.add_patch(modes_box)
    ax.text(8.0, 1.5, "Streamlit Inspection Outputs", ha="center", fontsize=10, fontweight="bold", color="#1E293B")
    
    outputs = [
        ("Original Crop", "224x224 model view", 3.4, 0.6, 2.6, 0.7),
        ("Filter Only (Heatmap)", "Colormapped activations", 6.7, 0.6, 2.6, 0.7),
        ("Grad-CAM Overlay", "Blended produce + attention", 10.0, 0.6, 2.6, 0.7),
    ]
    for otitle, odesc, ox, oy, ow, oh in outputs:
        op = FancyBboxPatch((ox, oy), ow, oh, boxstyle="round,pad=0.06,rounding_size=0.06",
                            facecolor="#FFFFFF", edgecolor="#0284C7", linewidth=1.2)
        ax.add_patch(op)
        ax.text(ox + ow/2, oy + oh - 0.22, otitle, ha="center", fontsize=8.5, fontweight="bold", color="#0369A1")
        ax.text(ox + ow/2, oy + 0.18, odesc, ha="center", fontsize=7.5, color="#475569")

    # Down arrow from Step 5 to inspection outputs
    ax.add_patch(FancyArrowPatch((13.7, 2.4), (11.5, 1.3), arrowstyle="->", mutation_scale=12, linewidth=1.6, color="#16A34A"))

    plt.tight_layout()
    save_path = OUTPUT_DIR / "gradcam_pipeline.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Generating architectural visual diagrams...")
    render_model_architecture()
    render_mbconv_block()
    render_two_phase_training()
    render_gradcam_workflow()
    print("All diagrams generated successfully in docs/images/!")
