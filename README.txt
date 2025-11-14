## Overview

This notebook trains a CNN classifier on **8,631 face identities** and uses the trained model as an **embedding extractor** for pairwise face verification.

---

## Config (Defaults)
Key defaults in `config`:

| batch_size | lr      | epochs | num_classes | seed |
|------------|---------|--------|-------------|------|
| 512        | 0.0075  | 40     | 8631        | 5330 |

---

## Transforms & Augmentations

Image preprocessing:

- `create_transforms(image_size=112, augment=True)`:
  - Resize to `112 × 112`
  - Convert to float tensor
  - Optional augmentations: random flip, rotation, resized crop, color jitter, blur
  - Normalize with mean/std = `0.5`

Batch-level augmentations:

- CutMix / MixUp via a custom `collate_fn`.

---

## Model

Provided architectures:

- `Resnet34` – custom model returning `{"feats", "out"}`
- `ResNet34Scratch` – torchvision-based ResNet-34 (no pretrained weights)

Both support classification and embedding extraction for verification.

---

## Training & Evaluation

- **Embedding training**  
  - Loss: `TripletMarginLoss`  
  - Dataset: `TripletImageDataset`  
  - Loop: `train_epoch_triplet`

- **Classification training / validation**  
  - Loss: `CrossEntropy` (for validation)  
  - Loop: `train_epoch`  
    - Supports MixUp / CutMix  
    - Uses mixed precision

- **Verification**  
  - L2-normalize embeddings  
  - Compute cosine similarity  
  - Metrics: **EER**, **AUC**, **ACC**, **TPR@FPR**

