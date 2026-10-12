# Methodology

This document describes the model architecture, data pipeline, training strategy, and key design decisions for the Fruit and Vegetable Classifier.

---

## 1. Dataset

**Source:** Kritik Seth -- [Fruit and Vegetable Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) (Kaggle)

| Split | Images | Classes |
|---|---|---|
| Train | 3,115 | 36 |
| Validation | 351 | 36 |
| Test | 359 | 36 |

The dataset covers 36 produce categories: apple, banana, beetroot, bell pepper, cabbage, capsicum, carrot, cauliflower, chilli pepper, corn, cucumber, eggplant, garlic, ginger, grapes, jalapeno, kiwi, lemon, lettuce, mango, onion, orange, paprika, pear, peas, pineapple, pomegranate, potato, radish, soy beans, spinach, sweetcorn, sweet potato, tomato, turnip, and watermelon.

Each class has approximately 10 test images, providing a balanced evaluation benchmark.

---

## 2. Model Architecture

### Backbone: EfficientNet-B0

EfficientNet-B0 was selected as the convolutional backbone for the following reasons:

- **Parameter efficiency:** 4.05M total parameters -- significantly fewer than ResNet-50 (25.6M) while achieving comparable or superior accuracy on ImageNet.
- **Compound scaling:** Width, depth, and resolution are scaled jointly using a principled compound coefficient, producing strong representations at minimal computational cost.
- **Transfer learning quality:** Pretrained ImageNet-1K weights provide rich low-level and mid-level feature extractors (edges, textures, shapes) that generalize well to produce classification.

### Classification Head

The original ImageNet head (1,000-class linear layer) is replaced with a task-specific head:

```
EfficientNet-B0 Backbone
    --> Global Average Pooling  (1280-dimensional feature vector)
    --> Dropout (p = 0.3)
    --> Linear (1280 --> 36)
    --> Log-Softmax (inference)
```

The dropout layer provides regularization against overfitting given the relatively small dataset size (~3,000 training images).

For the full layer-by-layer specification, MBConv block diagrams, Squeeze-and-Excitation attention details, and computational complexity analysis, refer to [`docs/architecture.md`](architecture.md).

---

## 3. Data Pipeline

### Augmentation Strategy (Training)

Training augmentations introduce controlled variance to improve generalization:

| Transform | Parameters | Rationale |
|---|---|---|
| `RandomResizedCrop` | size=224, scale=(0.7, 1.0) | Simulates distance variation and partial occlusion |
| `RandomHorizontalFlip` | p=0.5 | Natural mirror symmetry for most produce |
| `ColorJitter` | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05 | Robustness to lighting and color variation |
| `RandomRotation` | degrees=15 | Orientation invariance |
| `Normalize` | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] | ImageNet channel statistics |

### Validation and Test Transforms (Deterministic)

```
Resize(256) --> CenterCrop(224) --> ToTensor() --> Normalize(ImageNet stats)
```

No random transforms are applied at evaluation time to ensure reproducibility of reported metrics.

---

## 4. Training Strategy

### Two-Phase Transfer Learning

Training is split into two sequential phases to maximize the benefit of pretrained weights while avoiding destructive fine-tuning.

**Phase 1: Feature Extraction (Epochs 1-5)**

- Backbone weights are frozen (gradients disabled).
- Only the classification head is trained.
- Higher learning rate (`lr = 1e-3`) is used since randomly initialized head weights need rapid adaptation.
- This phase establishes a stable, well-calibrated classifier on top of fixed ImageNet features.

**Phase 2: Fine-Tuning (Epochs 6-15)**

- The top two convolutional blocks of EfficientNet-B0 are unfrozen.
- Lower learning rate (`lr = 1e-4`) prevents large gradient updates from corrupting pretrained representations.
- The classifier head retains its Phase 1 weights as a warm start.
- Fine-tuning adapts upper-layer features to produce-specific visual patterns (color gradients, surface textures, stems).

### Optimization Configuration

| Component | Configuration |
|---|---|
| Optimizer | AdamW (weight_decay = 1e-4) |
| Loss Function | CrossEntropyLoss with label smoothing = 0.1 |
| LR Scheduler | CosineAnnealingLR |
| Early Stopping | Patience = 5 epochs (monitors validation loss) |
| Batch Size | 32 |
| Random Seed | 42 (fixed for reproducibility) |

### Label Smoothing

Label smoothing with epsilon = 0.1 replaces hard one-hot targets with soft targets, reducing overconfidence and improving calibration on ambiguous classes (e.g., capsicum vs. bell pepper).

---

## 5. Model Explainability: GradCAM

**Gradient-weighted Class Activation Mapping (GradCAM)** is applied post-training to produce visual saliency maps that highlight which spatial regions of an input image most influenced the model's prediction.

GradCAM computes the gradient of the class score with respect to the final convolutional feature maps, then produces a weighted combination of those maps:

```
L_GradCAM = ReLU( sum_k( alpha_k * A_k ) )
```

where `alpha_k` is the global-average-pooled gradient for feature map `k`, and `A_k` is feature map `k`.

The resulting heatmap is overlaid on the input image, revealing that the model attends to semantically meaningful regions: the outer skin of citrus fruits, the distinctive shapes of leafy vegetables, and the surface textures of root vegetables.

---

## 6. Reproducibility

All hyperparameters are centralized in `train_config.yaml`. The random seed is fixed at the start of training via:

```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
numpy.random.seed(42)
```

The `best_model.pth` checkpoint stores both the model state dictionary and the `class_to_idx` mapping, ensuring that the same label ordering is used at inference time without re-scanning the dataset.
