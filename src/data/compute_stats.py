import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
from tqdm import tqdm


def compute_mean_std(train_dir: str, img_size: int = 256, batch_size: int = 64, num_workers: int = 2):
   
    temp_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=temp_transform)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Computing mean and std of train dataset...")

    mean = 0.0
    std = 0.0
    nb_samples = 0

    for data, _ in tqdm(loader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)

        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("\nMean computed:", mean)
    print("Std computed:", std)

    return mean.tolist(), std.tolist()
