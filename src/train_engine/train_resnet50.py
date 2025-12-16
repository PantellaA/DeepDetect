import copy
import time
import torch


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    criterion,
    optimizer,
    device,
    num_epochs,
    phase_name="train",
    patience=5,
    min_delta=1e-4,
    checkpoint_path=None,
    best_model_path=None,
    log_fn=None,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase not in dataloaders:
                continue

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # =========================
            # LOGGING (wandb o altro)
            # =========================
            if log_fn is not None:
                log_fn({
                    f"{phase_name}/{phase}_loss": epoch_loss,
                    f"{phase_name}/{phase}_acc": epoch_acc.item(),
                    "epoch": epoch + 1,
                })

            # =========================
            # VALIDATION CHECK
            # =========================
            if phase == "val":
                if epoch_loss < best_val_loss - min_delta:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_acc = epoch_acc.item()
                    epochs_no_improve = 0

                    if best_model_path is not None:
                        torch.save(best_model_wts, best_model_path)
                else:
                    epochs_no_improve += 1

        # =========================
        # CHECKPOINT
        # =========================
        if checkpoint_path is not None:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )

        # =========================
        # EARLY STOPPING
        # =========================
        if epochs_no_improve >= patience:
            break

    time_elapsed = time.time() - since

    model.load_state_dict(best_model_wts)

    info = {
        "best_val_loss": best_val_loss,
        "best_acc": best_acc,
        "time_minutes": time_elapsed / 60,
    }

    return model, info

