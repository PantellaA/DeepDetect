from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from src.utils.seed import set_global_seed
from src.data.download import download_raw_dataset
from src.data.build import build_working_data
from src.ResNet.normalization import IMAGENET_MEAN, IMAGENET_STD
from src.data.loaders import get_dataloaders
from src.utils.class_weights import compute_class_weights

from src.models.resnet50 import build_resnet50
from src.evaluate.eval import evaluate_model, evaluate_on_test


def main():
    set_global_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ========= WORKING_DATA =========
    work_dir = Path("working_data")

    if not work_dir.exists():
        print("[INFO] working_data not found. Starting download + build...")
        ddata = download_raw_dataset()
        _ = build_working_data(ddata, work_dir=str(work_dir))
        print("[INFO] working_data created.")
    else:
        print("[INFO] working_data already exists. Skipping download/build.")

    # ========= DATALOADERS =========
    BATCH_SIZE = 32
    IMG_SIZE = 224
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    dataloaders, dataset_sizes, class_names = get_dataloaders(
        work_dir=str(work_dir),
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        mean=mean,
        std=std,
        num_workers=2,
    )

    print("\nDataloaders created:")
    print(f"TRAIN → {dataset_sizes['train']:6d} images")
    print(f"VAL   → {dataset_sizes['val']:6d} images")
    print(f"TEST  → {dataset_sizes['test']:6d} images")
    print("Classes:", class_names)

    # ========= LOSS  =========
    train_dataset = dataloaders["train"].dataset
    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ========= MODEL =========
    model = build_resnet50(num_classes=2, pretrained=True).to(device)
    print("[INFO] Model built:", model.__class__.__name__)
    print("[INFO] Classifier head:", model.fc)

    # ========= LOAD BEST CHECKPOINT FROM HF =========
    print("[INFO] Downloading ResNet50 checkpoint from Hugging Face...")
    ckpt_path = hf_hub_download(
        repo_id="PantellaA/DeepDetect-Models",
        filename="resnet50/resnet50_best_checkpoint.pth",
    )
    print(f"[INFO] Checkpoint downloaded: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("[INFO] Model loaded and set to eval mode.")

    # ========= EVALUATION =========
    print("\n[INFO] Evaluating on VALIDATION set...")
    _ = evaluate_model(
        model=model,
        dataloader=dataloaders["val"],
        dataset_size=dataset_sizes["val"],
        criterion=criterion,
        device=device,
        class_names=class_names,
        phase_name="val",
    )

    print("\n[INFO] Evaluating on TEST set...")
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
