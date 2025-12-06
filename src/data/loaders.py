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
            image_datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            image_datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }

    dataset_sizes = {split: len(ds) for split, ds in image_datasets.items()}
    class_names = image_datasets["train"].classes  

    print("\n Dataloaders created:")
    for split in ["train", "val", "test"]:
        print(f"{split.upper():5s} → {dataset_sizes[split]:6d} immages")

    print(f"Class: {class_names}")

    return dataloaders, dataset_sizes, class_names
