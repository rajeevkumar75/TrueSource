from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models


class FocalLoss(nn.Module):
    """Down-weights easy examples; helps reduce \"everything looks real\" collapse on hard fakes."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        object.__setattr__(self, "_class_weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self._class_weight, reduction="none")
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce
        return focal.mean()


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    model_path: str = "models/image_detection_best.pth"
    freeze_backbone: bool = True
    use_class_weights: bool = False
    focal_gamma: float | None = None
    device: str = "auto"
    use_amp: bool = True
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 1e-4


def _extract_targets(dataset: object) -> torch.Tensor | None:
    if hasattr(dataset, "targets"):
        return torch.as_tensor(getattr(dataset, "targets"), dtype=torch.long)
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = getattr(dataset, "dataset")
        indices = torch.as_tensor(getattr(dataset, "indices"), dtype=torch.long)
        if hasattr(base, "targets"):
            base_targets = torch.as_tensor(getattr(base, "targets"), dtype=torch.long)
            return base_targets[indices]
    return None


def _balanced_accuracy(
    correct_per_class: torch.Tensor,
    count_per_class: torch.Tensor,
) -> float:
    valid = count_per_class > 0
    if not torch.any(valid):
        return 0.0
    recalls = correct_per_class[valid] / count_per_class[valid].clamp_min(1)
    return float(recalls.mean().item())


def create_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        # Fallback for offline environments where pretrained weights cannot be downloaded.
        model = models.resnet18(weights=None)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> tuple[float, float, float]:
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = torch.zeros(0, dtype=torch.long)
    class_count = torch.zeros(0, dtype=torch.long)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if optimizer is not None:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

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

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    balanced_acc = _balanced_accuracy(class_correct, class_count)
    return avg_loss, accuracy, balanced_acc


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    config: TrainConfig,
) -> tuple[nn.Module, dict[str, float]]:
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. Install CUDA-enabled PyTorch or use --device cpu."
            )
        device = torch.device("cuda")
    elif config.device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device must be one of: auto, cuda, cpu")

    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(
        f"Training on device={device} | "
        f"cuda_available={torch.cuda.is_available()} | amp={use_amp}"
    )

    model = create_model(num_classes=num_classes, freeze_backbone=config.freeze_backbone)
    model.to(device)

    class_weights = None
    if config.use_class_weights:
        targets = _extract_targets(train_loader.dataset)
    else:
        targets = None
    if targets is not None:
        class_counts = torch.bincount(targets, minlength=num_classes).float()
        class_weights = (class_counts.sum() / class_counts.clamp_min(1.0)).to(device)
        class_weights = class_weights / class_weights.mean()

    if config.focal_gamma is not None:
        focal_weight = class_weights.clone() if class_weights is not None else None
        criterion = FocalLoss(gamma=config.focal_gamma, weight=focal_weight).to(device)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing,
        )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_balanced_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        train_loss, train_acc, train_bal_acc = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc, val_bal_acc = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            scaler=None,
            use_amp=use_amp,
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_bal_acc={train_bal_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_bal_acc={val_bal_acc:.4f}"
        )
        scheduler.step(val_loss)

        if val_bal_acc > (best_val_balanced_acc + config.early_stopping_min_delta):
            best_val_acc = val_acc
            best_val_balanced_acc = val_bal_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no val_bal_acc improvement for {config.early_stopping_patience} epochs)"
                )
                break

    model.load_state_dict(best_state)

    model_output = Path(config.model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output)

    metrics = {
        "best_val_accuracy": best_val_acc,
        "best_val_balanced_accuracy": best_val_balanced_acc,
        "device": str(device),
        "label_smoothing": config.label_smoothing,
        "weight_decay": config.weight_decay,
        "class_weighting": float(class_weights is not None),
        "focal_gamma": float(config.focal_gamma) if config.focal_gamma is not None else -1.0,
        "amp": float(use_amp),
        "epochs_ran": float(epoch + 1),
    }
    return model, metrics
