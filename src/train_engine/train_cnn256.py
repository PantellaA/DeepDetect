import torch
import torch.nn as nn
import torch.optim as optim

from src.data.loaders import get_dataloaders
from src.data.compute_stats import compute_mean_std
from src.models.cnn256 import DeepFakeCNN256


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=10):
    pass  


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    mean, std = compute_mean_std("working_data/train", img_size=256)

    dataloaders, dataset_sizes, class_names = get_dataloaders(
        work_dir="working_data",
        batch_size=32,
        img_size=256,
        mean=mean,
        std=std,
        num_workers=2,
    )
  
    model = DeepFakeCNN256().to(device)
    print("Modello creato.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, num_epochs=30)

if __name__ == "__main__":
    main()
