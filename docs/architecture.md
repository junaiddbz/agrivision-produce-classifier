# Model Architecture

This document provides a technical and visual specification of the deep learning model implemented for the 36-class Fruit and Vegetable Classification system.

---

## 1. End-to-End Network Architecture

The network utilizes **EfficientNet-B0** pre-trained on ImageNet-1K, adapted with a task-specific classification head for 36 fine-grained produce categories.

![EfficientNet-B0 End-to-End Architecture](images/model_architecture.png)

### Key Specifications

| Specification | Value | Description |
|---|---|---|
| **Input Shape** | `[Batch, 3, 224, 224]` | Center-cropped RGB images normalized with ImageNet statistics |
| **Backbone** | EfficientNet-B0 | 8 stages of inverted bottleneck residual blocks (MBConv) |
| **Total Parameters** | 4,053,664 | 4.01M feature extractor + 46.1K classification head |
| **Feature Dimension** | 1,280 | Extracted via global spatial pooling prior to classification |
| **Output** | 36 Logits / Probabilities | Softmax distribution across all 36 produce categories |

---

## 2. Stage-by-Stage Layer Breakdown

The table below outlines the dimensional flow, kernel sizes, and parameter allocation from input to final logits.

| Stage | Block Type | Input Dimension | Operator Details | Output Dimension | Parameters |
|---|---|---|---|---|---|
| **0** | Stem Conv | `(3, 224, 224)` | Conv2d 3x3, Stride 2, BatchNorm, SiLU | `(32, 112, 112)` | 864 |
| **1** | MBConv1 | `(32, 112, 112)` | 1x Inverted Residual (exp=1, k=3x3, s=1), SE | `(16, 112, 112)` | 3,138 |
| **2** | MBConv6 | `(16, 112, 112)` | 2x Inverted Residual (exp=6, k=3x3, s=2), SE | `(24, 56, 56)` | 24,124 |
| **3** | MBConv6 | `(24, 56, 56)` | 2x Inverted Residual (exp=6, k=5x5, s=2), SE | `(40, 28, 28)` | 72,994 |
| **4** | MBConv6 | `(40, 28, 28)` | 3x Inverted Residual (exp=6, k=3x3, s=2), SE | `(80, 14, 14)` | 243,308 |
| **5** | MBConv6 | `(80, 14, 14)` | 3x Inverted Residual (exp=6, k=5x5, s=1), SE | `(112, 14, 14)` | 473,088 |
| **6** | MBConv6 | `(112, 14, 14)` | 4x Inverted Residual (exp=6, k=5x5, s=2), SE | `(192, 7, 7)` | 1,489,456 |
| **7** | MBConv6 | `(192, 7, 7)` | 1x Inverted Residual (exp=6, k=3x3, s=1), SE | `(320, 7, 7)` | 1,294,400 |
| **8** | Head Conv | `(320, 7, 7)` | Conv2d 1x1, Stride 1, BatchNorm, SiLU | `(1280, 7, 7)` | 410,880 |
| **Pooling** | Adaptive Pool | `(1280, 7, 7)` | AdaptiveAvgPool2d (Global Average Pooling) | `(1280, 1, 1)` | 0 |
| **Head** | Classifier | `(1280)` | Dropout (p=0.3) + Linear (1280 -> 36) | `(36)` | 46,116 |

---

## 3. MBConv Inverted Residual Block

EfficientNet-B0's foundational module is the Mobile Inverted Bottleneck Convolution (MBConv) with integrated Squeeze-and-Excitation (SE) channel attention.

![MBConv Block Architecture](images/mbconv_block.png)

### Structural Mechanism

1. **Expansion (1x1 Conv):** Channels expand by factor $t=6$ to project into higher-dimensional feature space.
2. **Depthwise Filtering (3x3 or 5x5):** Spatial filtering is applied per channel independently, separating spatial feature extraction from channel mixing.
3. **Squeeze-and-Excitation Attention:**
   - **Squeeze:** Spatial pooling reduces feature maps to $1 \times 1 \times C$.
   - **Excitation:** Two fully connected layers ($C \rightarrow C/4 \rightarrow C$) with a Sigmoid activation assign an importance weight to each channel.
   - **Rescale:** Channels are multiplied by their learned importance weights.
4. **Pointwise Projection (1x1 Conv):** Compresses channels back to target dimension without non-linear activation.
5. **Residual Skip Connection:** Adds input directly to output when stride is 1 and channel dimensions match.

---

## 4. Two-Phase Transfer Learning Strategy

To adapt ImageNet feature extractors without catastrophic forgetting or gradient instability, training is divided into two sequential phases.

![Two-Phase Transfer Learning](images/transfer_learning_phases.png)

### Phase Comparison

| Dimension | Phase 1 (Feature Extraction) | Phase 2 (Fine-Tuning) |
|---|---|---|
| **Epochs** | 1 to 5 | 6 to 15 |
| **Backbone Status** | Entire backbone frozen (requires_grad = False) | Stages 0-6 frozen; Stages 7 & 8 unfrozen |
| **Head Status** | Trainable | Trainable |
| **Trainable Params** | 46,116 (1.14%) | 1,751,396 (43.20%) |
| **Optimizer** | AdamW (weight_decay = 1e-4) | AdamW (weight_decay = 1e-4) |
| **Learning Rate** | 1e-3 (constant) | 1e-4 with Cosine Annealing decay |
| **Objective** | Fit linear head on frozen representations | Adapt upper convolutional blocks to produce textures |

---

## 5. Grad-CAM Explainability Pipeline

Gradient-weighted Class Activation Mapping (Grad-CAM) explains individual model predictions by computing gradients of the predicted class score with respect to feature maps in the final convolutional stage (`features[8]`).

![Grad-CAM Explainability Pipeline](images/gradcam_pipeline.png)

### Mathematical Formulation

1. **Gradient Pooling (Channel Weights):**
   $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$
   where $Z = 7 \times 7 = 49$ spatial dimensions, $y^c$ is the logit for class $c$, and $A^k$ is feature map $k$.

2. **Weighted Linear Combination with ReLU:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{1280} \alpha_k^c A^k \right)$$

3. **Bilinear Upsampling and Overlay:**
   The $7 \times 7$ saliency map is upsampled to $224 \times 224$, min-max normalized, and blended with the input image at $\alpha = 0.5$.

---

## 6. Computational Profiling

| Metric | EfficientNet-B0 | ResNet-50 Reference | Advantage |
|---|---|---|---|
| **Total Parameters** | 4.05 M | 25.56 M | **6.3x fewer parameters** |
| **FLOPs (Inference)** | ~0.39 GFLOPs | ~4.12 GFLOPs | **10.5x fewer FLOPs** |
| **Checkpoint Size** | 16.5 MB | 102.5 MB | **86 MB storage reduction** |
| **GPU Latency (RTX 3060)** | ~6.8 ms | ~14.2 ms | **2.1x lower latency** |
| **Test Accuracy** | 94.15% | ~93.80% | **Equal or superior accuracy** |
