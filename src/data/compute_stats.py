import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


def compute_mean_std(train_dir: str, img_size: int = 256, batch_size: int = 64, num_workers: int = 2):
    """
    Calcola mean e std GLOBALI del train set usando la formula corretta:
    std = sqrt(E[X^2] - (E[X])^2)
    """

    temp_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=temp_transform)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Computing GLOBAL mean and std of train dataset...")

    # accumulatori
    channels_sum = torch.zeros(3)
    channels_sq_sum = torch.zeros(3)
    total_pixels = 0

    for data, _ in tqdm(loader):
        # data shape: [B, C, H, W]
        
        channels_sum += data.sum(dim=[0, 2, 3])          # somma dei pixel per canale
        channels_sq_sum += (data ** 2).sum(dim=[0, 2, 3])  # somma dei quadrati
        total_pixels += data.size(0) * data.size(2) * data.size(3)

    # mean globale
    mean = channels_sum / total_pixels

    # std globale corretta
    std = torch.sqrt(channels_sq_sum / total_pixels - mean ** 2)

    print("\nMean computed:", mean)
    print("Std computed:", std)

    return mean.tolist(), std.tolist()
