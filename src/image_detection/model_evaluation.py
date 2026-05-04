from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _balanced_accuracy(
    correct_per_class: torch.Tensor,
    count_per_class: torch.Tensor,
) -> tuple[float, list[float]]:
    valid = count_per_class > 0
    if not torch.any(valid):
        return 0.0, []
    recalls = correct_per_class.float() / count_per_class.clamp_min(1).float()
    balanced = float(recalls[valid].mean().item())
    return balanced, [float(item) for item in recalls.tolist()]


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> dict[str, float | list[float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(0, dtype=torch.long)
    class_count = torch.zeros(0, dtype=torch.long)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += images.size(0)
            num_classes = logits.size(1)
            if class_correct.numel() != num_classes:
                class_correct = torch.zeros(num_classes, dtype=torch.long)
                class_count = torch.zeros(num_classes, dtype=torch.long)
            for class_idx in range(num_classes):
                class_mask = labels == class_idx
                class_count[class_idx] += class_mask.sum().cpu()
                class_correct[class_idx] += ((predictions == labels) & class_mask).sum().cpu()

    balanced_acc, per_class_recall = _balanced_accuracy(class_correct, class_count)

    return {
        "test_loss": total_loss / max(total_samples, 1),
        "test_accuracy": total_correct / max(total_samples, 1),
        "test_balanced_accuracy": balanced_acc,
        "per_class_recall": per_class_recall,
    }
