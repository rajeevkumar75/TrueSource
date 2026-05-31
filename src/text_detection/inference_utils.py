from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.text_detection.data_transformation import CLASS_NAMES

DEFAULT_AI_THRESHOLD = 0.5
LOW_CONFIDENCE_CUTOFF = 0.52


@lru_cache(maxsize=2)
def _load_model_and_tokenizer(model_path: str) -> tuple[torch.nn.Module, object, torch.device]:
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def load_ai_threshold(model_path: str) -> float:
    config_file = Path(model_path) / "inference.json"
    if not config_file.exists():
        return DEFAULT_AI_THRESHOLD
    try:
        payload = json.loads(config_file.read_text(encoding="utf-8"))
        return float(payload.get("ai_threshold", DEFAULT_AI_THRESHOLD))
    except (json.JSONDecodeError, TypeError, ValueError):
        return DEFAULT_AI_THRESHOLD


def save_inference_config(model_path: str, ai_threshold: float, metrics: dict | None = None) -> None:
    payload: dict[str, object] = {"ai_threshold": round(ai_threshold, 4)}
    if metrics:
        payload["validation_metrics"] = metrics
    config_file = Path(model_path) / "inference.json"
    config_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _labels_from_loader(loader: DataLoader) -> list[int]:
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        base = dataset.dataset
        return [int(base[index]["labels"]) for index in dataset.indices]
    return [int(dataset[index]["labels"]) for index in range(len(dataset))]


def compute_class_weights(loader: DataLoader, num_classes: int = 2) -> torch.Tensor:
    labels = _labels_from_loader(loader)
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels:
        counts[label] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


@torch.no_grad()
def collect_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
) -> tuple[list[float], list[int]]:
    device = next(model.parameters()).device
    model.eval()
    ai_index = CLASS_NAMES.index("ai")
    probs_ai: list[float] = []
    labels: list[int] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        batch_labels = batch.pop("labels")
        logits = model(**batch).logits
        batch_probs = torch.softmax(logits, dim=1)[:, ai_index].cpu().tolist()
        probs_ai.extend(float(value) for value in batch_probs)
        labels.extend(int(value) for value in batch_labels.cpu().tolist())

    return probs_ai, labels


def tune_ai_threshold(probs_ai: list[float], labels: list[int]) -> tuple[float, float]:
    if not probs_ai:
        return DEFAULT_AI_THRESHOLD, 0.0

    best_threshold = DEFAULT_AI_THRESHOLD
    best_balanced_acc = -1.0
    for step in range(26):
        threshold = 0.35 + step * 0.01
        human_correct = 0
        human_total = 0
        ai_correct = 0
        ai_total = 0

        for prob, label in zip(probs_ai, labels, strict=True):
            predicted_ai = prob >= threshold
            if label == CLASS_NAMES.index("ai"):
                ai_total += 1
                ai_correct += int(predicted_ai)
            else:
                human_total += 1
                human_correct += int(not predicted_ai)

        recalls: list[float] = []
        if human_total > 0:
            recalls.append(human_correct / human_total)
        if ai_total > 0:
            recalls.append(ai_correct / ai_total)
        balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_threshold = threshold

    return best_threshold, best_balanced_acc


def decision_from_probabilities(
    class_probabilities: dict[str, float],
    ai_threshold: float,
    low_confidence_below: float = LOW_CONFIDENCE_CUTOFF,
) -> tuple[str, str]:
    p_ai = class_probabilities["ai"]
    p_human = class_probabilities["human"]
    confidence = max(p_ai, p_human)

    if p_ai >= ai_threshold:
        label = "ai"
        decision = "ai_generated"
    else:
        label = "human"
        decision = "human_written"

    if confidence < low_confidence_below:
        decision = f"{decision} (low confidence)"
    return label, decision
