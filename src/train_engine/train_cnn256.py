# src/train/trainer.py
import copy
import time
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple

import torch


def train_model(
    model: torch.nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    dataset_sizes: Dict[str, int],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 30,
    patience: int = 5,
    min_delta: float = 0.0,
    best_model_path: Optional[str | Path] = None,
    start_epoch: int = 0,
    log_fn: Optional[Callable[[dict], None]] = None,
) -> Tuple[torch.nn.Module, dict]:
    """
    Training loop con early stopping su val_loss e salvataggio del best model.
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_acc = 0.0

    epochs_no_improve = 0
    stop_early = False

    if best_model_path is not None:
        best_model_path = Path(best_model_path)
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)          # logits
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if log_fn is not None:
                log_fn({
                    "epoch": epoch + 1,
                    f"{phase}_loss": epoch_loss,
                    f"{phase}_acc": epoch_acc,
                })

            # ---- EARLY STOP + BEST MODEL SOLO SU VALIDATION ----
            if phase == "val":
                improved = epoch_loss < (best_val_loss - min_delta)

                if improved:
                    best_val_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0

                    if best_model_path is not None:
                        torch.save(best_model_wts, best_model_path)
                        print("💾 Best model salvato!")
                else:
                    epochs_no_improve += 1
                    print(f"📉 Nessun miglioramento della val_loss per {epochs_no_improve} epoche.")

                    if epochs_no_improve >= patience:
                        print(f"🛑 Early Stopping attivato all'epoch {epoch+1}")
                        stop_early = True

        if stop_early:
            break

    time_elapsed = time.time() - since
    print(f"\n⏳ Training completato in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Migliore Val Accuracy (al best val_loss): {best_acc:.4f}")
    print(f"📉 Migliore Val Loss: {best_val_loss:.4f}")

    model.load_state_dict(best_model_wts)
    return model, {
        "best_val_loss": best_val_loss,
        "best_acc": best_acc,
        "history": history,
        "stopped_early": stop_early,
    }
