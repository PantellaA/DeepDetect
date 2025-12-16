from pathlib import Path
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(
    work_dir: str,
    batch_size: int = 32,
    img_size: int = 256,
    mean=None,
    std=None,
    num_workers: int = 2,
):
    """
    Creates PyTorch ImageFolder datasets and dataloaders for the training, validation,
    and test splits, applying data augmentation to the training set and normalization
    using the provided mean and standard deviation.
    """

    work_path = Path(work_dir)

    # ----- 1. TRANSFORMS -----
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ----- 2. DATASETS -----
    image_datasets = {
        "train": datasets.ImageFolder(
            root=os.path.join(work_path, "train"),
            transform=transform_train,
        ),
        "val": datasets.ImageFolder(
            root=os.path.join(work_path, "val"),
            transform=transform_eval,
        ),
        "test": datasets.ImageFolder(
            root=os.path.join(work_path, "test"),
            transform=transform_eval,
        ),
    }

    # ----- 3. DATALOADERS -----
    dataloaders = {
        "train": DataLoader(
            im
