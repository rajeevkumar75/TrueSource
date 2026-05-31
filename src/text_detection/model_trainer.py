from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.text_detection.data_transformation import CLASS_NAMES
from src.text_detection.inference_utils import compute_class_weights


@dataclass(frozen=True)
class TextTrainConfig:
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    model_name: str = "distilbert-base-uncased"
    model_path: str = "models/text_detection_best"
    device: str = "auto"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_class_weights: bool = True
    classifier_dropout: float = 0.2


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    loss_fn: nn.Module,
    max_grad_norm: float,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(2, dtype=torch.long)
    class_count = torch.zeros(2, dtype=torch.long)

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch.pop("labels")

        with torch.set_grad_enabled(is_train):
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
    accuracy = total_correct / max(total_samples, 1)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, accuracy, balanced_acc


def train_text_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_labels: int,
    config: TextTrainConfig,
    tokenizer: object | None = None,
) -> tuple[torch.nn.Module, dict[str, float]]:
    device = _resolve_device(config.device)
    id2label = {idx: name for idx, name in enumerate(CLASS_NAMES)}
    label2id = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    hf_config = AutoConfig.from_pretrained(config.model_name)
    hf_config.num_labels = num_labels
    hf_config.id2label = id2label
    hf_config.label2id = label2id
    if hasattr(hf_config, "dropout"):
        hf_config.dropout = config.classifier_dropout
    if hasattr(hf_config, "seq_classif_dropout"):
        hf_config.seq_classif_dropout = config.classifier_dropout

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        config=hf_config,
    )
    model.to(device)

    class_weights = None
    if config.use_class_weights:
        class_weights = compute_class_weights(train_loader, num_classes=num_labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, len(train_loader) * config.epochs)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_balanced_acc = -1.0
    patience_counter = 0
    last_train_metrics = {"train_loss": 0.0, "train_accuracy": 0.0, "train_balanced_accuracy": 0.0}
    last_val_metrics = {"val_loss": 0.0, "val_accuracy": 0.0, "val_balanced_accuracy": 0.0}

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, train_bal_acc = _run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scheduler,
            loss_fn,
            config.max_grad_norm,
        )
        val_loss, val_acc, val_bal_acc = _run_epoch(
            model,
            val_loader,
            device,
            None,
            None,
            loss_fn,
            config.max_grad_norm,
        )

        last_train_metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_balanced_accuracy": train_bal_acc,
        }
        last_val_metrics = {
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_balanced_accuracy": val_bal_acc,
        }

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_bal_acc={train_bal_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_bal_acc={val_bal_acc:.4f}"
        )

        if val_bal_acc > best_balanced_acc + config.early_stopping_min_delta:
            best_balanced_acc = val_bal_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_dir = Path(config.model_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(save_dir)

    metrics = {**last_train_metrics, **last_val_metrics, "best_val_balanced_accuracy": best_balanced_acc}
    return model, metrics
