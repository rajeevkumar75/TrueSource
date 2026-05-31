from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from src.text_detection.data_transformation import CLASS_NAMES
from src.text_detection.inference_utils import collect_probabilities, tune_ai_threshold


def evaluate_text_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    ai_threshold: float | None = None,
) -> dict[str, float | list[float]]:
    device = next(model.parameters()).device
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(2, dtype=torch.long)
    class_count = torch.zeros(2, dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            for class_idx in range(2):
                mask = labels == class_idx
                class_count[class_idx] += mask.sum().cpu()
                class_correct[class_idx] += ((predictions == labels) & mask).sum().cpu()

    recalls = class_correct.float() / class_count.clamp_min(1).float()
    valid = class_count > 0
    balanced_acc = float(recalls[valid].mean().item()) if torch.any(valid) else 0.0

    metrics: dict[str, float | list[float]] = {
        "test_loss": total_loss / max(total_samples, 1),
        "test_accuracy": total_correct / max(total_samples, 1),
        "test_balanced_accuracy": balanced_acc,
        "per_class_recall": [float(item) for item in recalls.tolist()],
    }

    if ai_threshold is not None:
        probs_ai, labels = collect_probabilities(model, test_loader)
        ai_index = CLASS_NAMES.index("ai")
        threshold_correct = 0
        human_correct = 0
        human_total = 0
        ai_correct = 0
        ai_total = 0

        for prob, label in zip(probs_ai, labels, strict=True):
            predicted_ai = prob >= ai_threshold
            if label == ai_index:
                ai_total += 1
                ai_correct += int(predicted_ai)
                threshold_correct += int(predicted_ai)
            else:
                human_total += 1
                human_correct += int(not predicted_ai)
                threshold_correct += int(not predicted_ai)

        metrics["threshold_accuracy"] = threshold_correct / max(len(labels), 1)
        if human_total and ai_total:
            metrics["threshold_balanced_accuracy"] = (
                (human_correct / human_total) + (ai_correct / ai_total)
            ) / 2.0

    return metrics


MIN_SAMPLES_FOR_THRESHOLD_TUNING = 8


def tune_and_save_threshold(model: torch.nn.Module, val_loader: DataLoader, model_path: str) -> float:
    from src.text_detection.inference_utils import DEFAULT_AI_THRESHOLD, save_inference_config

    probs_ai, labels = collect_probabilities(model, val_loader)
    if len(labels) < MIN_SAMPLES_FOR_THRESHOLD_TUNING:
        threshold = DEFAULT_AI_THRESHOLD
        save_inference_config(
            model_path,
            ai_threshold=threshold,
            metrics={
                "note": (
                    f"Validation set has only {len(labels)} samples; "
                    f"using default threshold {threshold}"
                ),
            },
        )
        return threshold

    threshold, balanced_acc = tune_ai_threshold(probs_ai, labels)
    save_inference_config(
        model_path,
        ai_threshold=threshold,
        metrics={"val_balanced_accuracy_at_threshold": round(balanced_acc, 4)},
    )
    return threshold
