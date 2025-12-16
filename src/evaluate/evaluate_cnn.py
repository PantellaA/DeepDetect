import torch
import torch.nn.functional as F
from typing import Dict, Tuple, List

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    dataset_size: int,
    criterion: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    phase_name: str = "val",
):
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        probs = F.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    val_loss = running_loss / dataset_size
    val_acc = running_corrects / dataset_size

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    print(f"\n================ {phase_name.upper()} SET =================")
    print(f"{phase_name.capitalize()} Loss: {val_loss:.4f}")
    print(f"{phase_name.capitalize()} Accuracy: {val_acc:.4f}")

    print("\nConfusion matrix (righe = veri, colonne = predetti):")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        print("ROC-AUC non calcolabile (una sola classe presente).")

    print("===========================================")

    return val_loss, val_acc


@torch.no_grad()
def evaluate_on_test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    dataset_size: int,
    criterion: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_corrects = 0

    all_labels = []
    all_preds = []
    all_probs = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        probs = F.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (preds == labels).sum().item()

        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())

    test_loss = running_loss / dataset_size
    test_acc = running_corrects / dataset_size

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    print("\n================ TEST SET =================")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    print("\nConfusion matrix (righe = veri, colonne = predetti):")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        print("ROC-AUC non calcolabile (una sola classe presente).")

    print("===========================================")

    return test_loss, test_acc
