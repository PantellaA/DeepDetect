# DeepDetect — Repository Structure Overview

This repository contains a modular pipeline for **binary face image classification** (AI-generated vs real) using:
* A **custom CNN** ($256 \times 256$ input).
* A **fine-tuned ResNet50** pre-trained on ImageNet.

The codebase is organized to clearly separate **data handling, models, training, evaluation, and utilities**.

---

## Dataset

This project uses the **DeepDetect-2025** dataset for real vs AI-generated image classification.

- Dataset: DeepDetect-2025  
- Source: [https://www.kaggle.com/datasets/datasetcreator/deepdetect-2025](https://www.kaggle.com/datasets/ayushmandatta1/deepdetect-2025)  
- License: Apache License 2.0  

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

## Scripts (`scripts/`)

This directory contains **standalone evaluation scripts** designed to reproduce results
**without retraining the models**.

The scripts automatically **download the best model checkpoints from the Hugging Face Hub**, rebuild the corresponding architectures, and run
**validation and test evaluation** using the same preprocessing pipeline adopted during training.

- `evaluate_cnn256.py`  
  Downloads the **best CNN256 checkpoint** from Hugging Face and evaluates the model on the
  validation and test sets.  
  
- `evaluate_resnet50.py`  
  Downloads the **best fine-tuned ResNet50 checkpoint** from Hugging Face and evaluates the model
  on the validation and test sets.

**Hugging Face model repository:**  
`PantellaA/DeepDetect-Models`

>  Note: these scripts do **not** perform training; they only load and evaluate the **best model checkpoints obtained during training**.

---

## Core Modules (`src/`)

### `src/data/`
Handles **dataset preparation**:
* `download.py`: dataset download.
* `build.py`: construction of train/validation/test splits.
* `compute_stats.py`: computation of mean and std of train set for normalization (only for CNN256).
* `loaders.py`: creation of PyTorch dataloaders.

--- 

### `src/models/`
Contains **model definitions**:
* `cnn256.py`: custom CNN architecture for 256×256 images.
* `resnet50.py`: ResNet50 builder with a custom binary classification head.

---

### `src/train_engine/`
Implements **training loops**:
* `train_cnn256.py`: training loop used for the custom CNN, with early stopping.
* `train_resnet50.py`: training loop used for ResNet50 fine-tuning, with early stopping.

---

### `src/evaluate/`
`eval.py` Provides **evaluation utilities**:
* loss and accuracy
* confusion matrix
* classification metrics (precision, recall, F1).

---

### `src/utils/`
General utility functions:
* `seed.py` Random seed control (reproducibility).
* `class_weights.py` Class-weight computation.
* `freeze.py` Freezing/unfreezing model layers during fine-tuning.

---

### `src/ResNet/`
`normalization.py` Supporting components for ResNet-based models:
* normalization utilities aligned with ImageNet preprocessing.
