import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_mean_std(train_dir: str, img_size: int = 256, batch_size: int = 64, num_workers: int = 2):
    """
    Calcola mean e std con lo stesso metodo del tuo secondo snippet:
    - mean = media delle medie per immagine
    - std  = media delle std per immagine
    """

    temp_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=temp_transform)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("🧮 Calcolo della media e deviazione standard del train_dataset...")

    mean = torch.zeros(3)
    std = torch.zeros(3)
    nb_samples = 0.0

    for data, _ in tqdm(loader):
        batch_samples = data.size(0)  # B
        data = data.view(batch_samples, data.size(1), -1)  # [B, C, H*W]

        mean += data.mean(2).sum(0)   # somma delle mean per immagine, per canale
        std  += data.std(2).sum(0)    # somma delle std  per immagine, per canale
        nb_samples += batch_samples

    mean /= nb_samples
    std  /= nb_samples

    print("\nMean computed:", mean)
    print("Std computed: ", std)

    return mean.tolist(), std.tolist()

    return mean.tolist(), std.tolist()
