# scripts/evaluate_resnet50.py

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from huggingface_hub import hf_hub_download

from src.data.download import download_raw_dataset
from src.data.build import build_working_data
from src.data.loaders import get_dataloaders
from src.RESNET.normalization import IMAGENET_MEAN, IMAGENET_STD

from src.evaluate.eval import evaluate_model, evaluate_on_test


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ========= PATHS =========
    work_dir = Path("working_data")

    # ========= DATA PREP (ONLY IF NEEDED) =========
    if not work_dir.exists():
        print("[INFO] working_data non trovato. Avvio download + build...")
        raw_path = download_raw_dataset()             # deve restituire path cartella raw
        build_working_data(raw_path, str(work_dir))   # se accetta stringa
        print("[INFO] working_data creato.")
    else:
        print("[INFO] working_data già presente. Skip download/build.")

    # ========= DATALOADERS =========
    dataloaders, dataset_sizes, class_names = get_dataloaders(
        work_dir=str(work_dir),
        batch_size=32,
        img_size=224,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    criterion = torch.nn.CrossEntropyLoss()

    # ========= MODEL =========
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    # ========= WEIGHTS (HF) =========
    print("[INFO] Scarico checkpoint ResNet50 da Hugging Face...")
    weights_path = hf_hub_download(
        repo_id="PantellaA/DeepDetect-Models",
        filename="resnet50/resnet50_best_checkpoint.pth",
    )
    print(f"[INFO] Checkpoint scaricato: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Modello caricato e messo in eval().")

    # ========= EVALUATION =========
    print("\n[INFO] Valutazione su VALIDATION set...")
    _ = evaluate_model(
        model=model,
        dataloader=dataloaders["val"],
        dataset_size=dataset_sizes["val"],
        criterion=criterion,
        device=device,
        class_names=class_names,
        phase_name="val",
    )

    print("\n[INFO] Valutazione su TEST set...")
    _ = evaluate_on_test(
        model=model,
        dataloader=dataloaders["test"],
        dataset_size=dataset_sizes["test"],
        criterion=criterion,
        device=device,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()


