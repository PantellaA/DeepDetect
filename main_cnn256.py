from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.utils.seed import set_global_seed
from src.data.download import download_raw_dataset
from src.data.build import build_working_data
from src.data.compute_stats import compute_mean_std
from src.data.loaders import get_dataloaders
from src.utils.class_weights import compute_class_weights

from src.models.cnn256 import DeepFakeCNN256
from src.train_engine.train_cnn256 import train_model

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
    # 2) CREATING WORKING_DATA
    # =========================
    print("\n==========================")
    print(" 2) CREATING WORKING_DATA")
    print("==========================")
    work_dir = "working_data"
    work_path = build_working_data(ddata, work_dir=work_dir)
    print(f"Working data created in: {work_path}")

    # =========================
    # 3) COMPUTING MEAN / STD
    # =========================
    print("\n===============================")
    print(" 3) COMPUTING MEAN / STD (TRAIN)")
    print("===============================")
    train_dir = f"{work_dir}/train"
    mean, std = compute_mean_std(train_dir)
    print("\nMean:", mean)
    print("Std :", std)

    # =========================
    # 4) DATALOADERS
    # =========================
    print("\n===============================")
    print(" 4) CREATING DATALOADERS")
    print("===============================")
    BATCH_SIZE = 32
    IMG_SIZE = 256

    dataloaders, dataset_sizes, class_names = get_dataloaders(
        work_dir=work_dir,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        mean=mean,
        std=std,
        num_workers=2,
    )

    print("\nDataloaders ready")
    print("Classes:", class_names)
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
    # 6) MODEL + OPTIMIZER
    # =========================
    print("\n===============================")
    print(" 6) MODEL + OPTIMIZER")
    print("===============================")

    model = DeepFakeCNN256().to(device)
    LR = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # =========================
    # 7) WANDB INIT 
    # =========================
    print("\n===============================")
    print(" 7) WANDB INIT")
    print("===============================")

    wandb.init(
        project="deepfake-cnn-256",
        name="run_CNN_256new",
        config={
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": 30,              
            "optimizer": "Adam",
            "lr": LR,
            "model": "DeepFakeCNN256",
            "patience": 5,
            "min_delta": 0.0,
        }
    )
    wandb.watch(model, log="all", log_freq=100)

    def log_fn(metrics: dict):
        wandb.log(metrics)

    # =========================
    # 8) TRAINING (EARLY STOP)
    # =========================
    print("\n===============================")
    print(" 8) TRAINING")
    print("===============================")

    BEST_MODEL_PATH = "checkpoints/best_model.pth"

    model_trained, info = train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=30,
        patience=5,
        min_delta=0.0,
        best_model_path=BEST_MODEL_PATH,
        log_fn=log_fn,
    )

    print("\n===============================")
    print(" TRAINING ENDED")
    print("===============================")
    print("Best val loss:", info["best_val_loss"])
    print("Best acc:", info["best_acc"])
    print("Best model saved in:", BEST_MODEL_PATH)


    # =========================
    # 9) EVALUATION (BEST MODEL)
    # =========================
    print("\n===============================")
    print(" 9) EVALUATION")
    print("===============================")

    # Validation (best model)
    _ = evaluate_model(
        model=model_trained,
        dataloader=dataloaders["val"],
        dataset_size=dataset_sizes["val"],
        criterion=criterion,
        device=device,
        class_names=class_names,
        phase_name="val",
    )

    # Test (best model)
    _ = evaluate_on_test(
        model=model_trained,
        dataloader=dataloaders["test"],
        dataset_size=dataset_sizes["test"],
        criterion=criterion,
        device=device,
        class_names=class_names,
    )


    wandb.finish()

if __name__ == "__main__":
    main()
