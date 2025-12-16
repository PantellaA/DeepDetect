# DeepDetect — Repository Structure Overview

This repository contains a modular pipeline for **binary face image classification** (AI-generated vs real) using:
* A **custom CNN** ($256 \times 256$ input).
* A **fine-tuned ResNet50** pre-trained on ImageNet.

The codebase is organized to clearly separate **data handling, models, training, evaluation, and utilities**.

---

##  Entry Points (root directory)

* `main_cnn256.py`
    Runs the full pipeline for the custom CNN:
    data preparation → training → validation → testing.

* `main_finetuned.py`
    Runs the fine-tuning pipeline for ResNet50 in two stages:
    1.  Training only the classifier head.
    2.  Fine-tuning the last convolutional block + head.

---


## Core Modules (`src/`)

### `src/data/`
Handles **dataset preparation**:
* `download.py`: dataset download
* `build.py`: construction of train/validation/test splits
* `compute_stats.py`: computation of mean and std of train set for normalization (only CNN256)
* `loaders.py`: creation of PyTorch dataloaders.

--- 

### `src/models/`
Contains **model definitions**:
* `cnn256.py`: custom CNN architecture for 256×256 images
* `resnet50.py`: ResNet50 builder with a custom binary classification head.

---

### `src/train_engine/`
Implements **training loops**:
* CNN-specific training
* ResNet50 fine-tuning with early stopping and checkpointing.

---

### `src/evaluate/`
Provides **evaluation utilities**:
* loss and accuracy
* confusion matrix
* classification metrics (precision, recall, F1).

---

### `src/utils/`
General utility functions:
* random seed control (reproducibility)
* class-weight computation
* freezing/unfreezing model layers during fine-tuning.

---

`src/ResNet/`
Supporting components for ResNet-based models:
* normalization utilities aligned with ImageNet preprocessing.
