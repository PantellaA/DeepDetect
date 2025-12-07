from pathlib import Path
import torch
from src.utils.seed import set_global_seed
from src.data.download import download_raw_dataset
from src.data.build import build_working_data
from src.data.compute_stats import compute_mean_std
from src.data.loaders import get_dataloaders
from src.utils.class_weights import compute_class_weights


def main():

    set_global_seed(42)

    print("\n======================")
    print(" 1) DOWNLOAD DATASET")
    print("======================")

    ddata: Path = download_raw_dataset()     # Restituisce .../ddata
    print(f"Dataset path: {ddata}")


    print("\n==========================")
    print(" 2) CREAZIONE WORKING_DATA")
    print("==========================")

    work_dir = "working_data"
    work_path = build_working_data(ddata, work_dir=work_dir)
    print(f"Working data creato in: {work_path}")


    print("\n===============================")
    print(" 3) CALCOLO MEAN / STD (TRAIN)")
    print("===============================")

    train_dir = f"{work_dir}/train"
    mean, std = compute_mean_std(train_dir)

    print("\nMean:", mean)
    print("Std :", std)


    print("\n===============================")
    print(" 4) CREAZIONE DATALOADERS")
    print("===============================")

    dataloaders, dataset_sizes, class_names = get_dataloaders(
        work_dir=work_dir,
        batch_size=32,
        img_size=256,
        mean=mean,
        std=std,
        num_workers=2,
    )

    print("\nDataloaders pronti!")
    print("Classi:", class_names)
    print("Sizes:", dataset_sizes)

    print("\n===============================")
    print(" 5) CLASS WEIGHTS")
    print("===============================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = dataloaders["train"].dataset
    class_weights = compute_class_weights(train_dataset, device)

    print("Class weights:", class_weights)

    print("\n Pipeline completata! Pronto per il training.")


if __name__ == "__main__":
    main()
