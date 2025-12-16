from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.utils.seed import set_global_seed
from src.data.download import download_raw_dataset
from src.data.build import build_working_data
from src.ResNet.normalization import IMAGENET_MEAN, IMAGENET_STD
from src.data.loaders import get_dataloaders
from src.utils.class_weights import compute_class_weights

from src.models.resnet50 import build_resnet50
from src.utils.freeze import (
    set_trainable_head_only,
    set_trainable_layer4_and_head,
)

from src.train_engine.train_resnet50 import train_model
from src.evaluate.eval import evaluate_model, evaluate_on_test


def main():
    # =========================
    # 0) SEED + DEVICE
    # =========================
    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # =========================
    # 1) DOWNLOAD DATASET
    # =========================
    print("\n======================")
    print(" 1) DOWNLOAD DATASET")
    print("======================")
    ddata: Path = download_raw_dataset()
    print(f"Dataset path: {ddata}")

    # =========================
    # 2) CREAZIONE WORKING_DATA
    # =========================
    print("\n==========================")
    print(" 2) CREAZIONE WORKING_DATA")
    print("==========================")
    work_dir = "working_data"
    work_path = build_working_data(ddata, work_dir=work_dir)
    print(f"Working data creato in: {work_path}")

    # =========================
    # 3) MEAN / STD (IMAGENET)
    # =========================
    print("\n===============================")
    print(" 3) MEAN AND STD (PREDEFINED)")
    print("===============================")
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    print("Mean:", mean)
    print("Std :", std)

    # =========================
    # 4) DATALOADERS
    # =========================
    print("\n===============================")
    print(" 4) CREAZIONE DATALOADERS")
    print("===============================")
    BATCH_SIZE = 32
    IMG_SIZE = 224

    dataloaders, dataset_sizes, class_names = get_dataloaders(
        work_dir=work_dir,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        mean=mean,
        std=std,
        num_workers=2,
    )

    print("Classi:", class_names)
    print("Sizes:", dataset_sizes)

    # =========================
    # 5) CLASS WEIGHTS + LOSS
    # =========================
    print("\n===============================")
    print(" 5) CLASS WEIGHTS + LOSS")
    print("===============================")

    train_dataset = dataloaders["train"].dataset
    class_weights = compute_class_weights(train_dataset, device)
    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # =========================
    # 6) MODEL (RESNET50)
    # =========================
    print("\n===============================")
    print(" 6) MODEL (RESNET50)")
    print("===============================")

    model = build_resnet50(num_classes=2, pretrained=True).to(device)
    print("Classifier head:", model.fc)

    # =========================
    # WANDB INIT
    # =========================
    wandb.init(
        project="deepdetect-resnet50",
        name="resnet50_finetune",
        config={
            "model": "ResNet50",
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "optimizer_phase1": "Adam",
            "lr_phase1": 1e-4,
            "optimizer_phase2": "Adam",
            "lr_phase2": 1e-5,
            "epochs_phase1": 10,
            "epochs_phase2": 10,
            "class_weights": "balanced",
            "normalization": "ImageNet",
        },
    )
    
    wandb.watch(model, log="all", log_freq=100)
    
    def log_fn(metrics: dict):
        wandb.log(metrics)

    
    # =========================
    # 7) PHASE 1: FC ONLY
    # =========================
    print("\n===============================")
    print(" 7) PHASE 1: TRAIN FC ONLY")
    print("===============================")

    set_trainable_head_only(model)
    optimizer_phase1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )

    model, info1 = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer_phase1,
        device=device,
        num_epochs=10,
        patience=5,
        min_delta=1e-4,
        best_model_path="checkpoints/resnet50_best_phase1.pth",
        log_fn=log_fn,
    )

    # =========================
    # 8) PHASE 2: LAYER4 + FC
    # =========================
    print("\n===============================")
    print(" 8) PHASE 2: TRAIN LAYER4 + FC")
    print("===============================")

    set_trainable_layer4_and_head(model)
    optimizer_phase2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,
    )

    model, info2 = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer_phase2,
        device=device,
        num_epochs=10,
        patience=7,
        min_delta=1e-4,
        best_model_path="checkpoints/resnet50_best_phase2.pth",
        log_fn=log_fn,
    )

    # =========================
    # 9) EVALUATION (BEST MODEL)
    # =========================
    print("\n===============================")
    print(" 9) EVALUATION")
    print("===============================")

    # Ricarico esplicitamente il best finale (scelta robusta)
    best_path = "checkpoints/resnet50_best_phase2.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    _ = evaluate_model(
        model=model,
        dataloader=dataloaders["val"],
        dataset_size=dataset_sizes["val"],
        criterion=criterion,
        device=device,
        class_names=class_names,
        phase_name="val",
    )

    _ = evaluate_on_test(
        model=model,
        dataloader=dataloaders["test"],
        dataset_size=dataset_sizes["test"],
        criterion=criterion,
        device=device,
        class_names=class_names,
    )

wandb.finish()

if __name__ == "__main__":
    main()
