import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(train_dataset, device):
    """
    Compute balanced class weights for the training dataset
    """
    targets = train_dataset.targets
    classes = np.unique(targets)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=targets
    )

    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"[INFO] Computed class weights: {weights.tolist()}")
    return weights
