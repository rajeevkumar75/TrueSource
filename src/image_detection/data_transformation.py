from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from src.image_detection.data_ingestion import DatasetPaths


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]
    split_summary: dict[str, dict[str, int]]


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


def _group_key_from_path(path: str) -> str:
    path_obj = Path(path)
    stem = path_obj.stem.lower()
    stem = re.sub(r"[_-]?frame[_-]?\d+$", "", stem)
    stem = re.sub(r"[_-]?\d+$", "", stem)
    parent = path_obj.parent.name.lower()
    return f"{parent}:{stem}"


def _split_class_indices(
    class_indices: torch.Tensor,
    paths: list[str],
    val_split: float,
    test_split: float,
    generator: torch.Generator,
    group_aware_split: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = class_indices.numel()
    target_val = max(1, int(total * val_split))
    target_test = max(1, int(total * test_split))
    target_train = max(1, total - target_val - target_test)
    if target_train <= 0:
        target_train = max(1, total // 2)
        remaining = total - target_train
        target_val = max(1, remaining // 2)
        target_test = max(0, remaining - target_val)

    permuted = class_indices[torch.randperm(total, generator=generator)]
    if not group_aware_split:
        val_idx = permuted[:target_val]
        test_idx = permuted[target_val : target_val + target_test]
        train_idx = permuted[target_val + target_test :]
        if train_idx.numel() == 0:
            train_idx = permuted[-1:]
            test_idx = test_idx[:-1] if test_idx.numel() > 0 else test_idx
        return train_idx, val_idx, test_idx

    groups: dict[str, list[int]] = {}
    for idx in permuted.tolist():
        key = _group_key_from_path(paths[idx])
        groups.setdefault(key, []).append(idx)

    group_items = list(groups.items())
    order = torch.randperm(len(group_items), generator=generator).tolist()
    train_ids: list[int] = []
    val_ids: list[int] = []
    test_ids: list[int] = []
    val_count = 0
    test_count = 0

    for order_idx in order:
        _, members = group_items[order_idx]
        if val_count < target_val:
            val_ids.extend(members)
            val_count += len(members)
        elif test_count < target_test:
            test_ids.extend(members)
            test_count += len(members)
        else:
            train_ids.extend(members)

    if len(train_ids) == 0:
        fallback = val_ids[-1:] if len(val_ids) > 1 else test_ids[-1:]
        train_ids.extend(fallback)
        if len(val_ids) > 1:
            val_ids = val_ids[:-1]
        elif len(test_ids) > 1:
            test_ids = test_ids[:-1]

    return (
        torch.as_tensor(train_ids, dtype=torch.long),
        torch.as_tensor(val_ids, dtype=torch.long),
        torch.as_tensor(test_ids, dtype=torch.long),
    )


def _train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.75, 1.0),
                ratio=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    dataset_paths: DatasetPaths,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    val_split: float = 0.1,
    test_split: float = 0.1,
    split_seed: int = 42,
    use_weighted_sampler: bool = True,
    group_aware_split: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoaders:
    pin_memory = torch.cuda.is_available()

    if dataset_paths.pre_split:
        train_dataset = datasets.ImageFolder(
            root=str(dataset_paths.train_dir), transform=_train_transform(image_size)
        )
        val_dataset = datasets.ImageFolder(
            root=str(dataset_paths.val_dir), transform=_eval_transform(image_size)
        )
        test_dataset = datasets.ImageFolder(
            root=str(dataset_paths.test_dir), transform=_eval_transform(image_size)
        )
        class_names = train_dataset.classes
    else:
        if dataset_paths.root_dir is None:
            raise ValueError("root_dir is required when pre_split is False")
        if val_split <= 0 or test_split <= 0 or (val_split + test_split) >= 1:
            raise ValueError("val_split and test_split must be >0 and sum to <1")

        train_base = datasets.ImageFolder(
            root=str(dataset_paths.root_dir), transform=_train_transform(image_size)
        )
        eval_base = datasets.ImageFolder(
            root=str(dataset_paths.root_dir), transform=_eval_transform(image_size)
        )
        class_names = train_base.classes
        sample_paths = [sample[0] for sample in train_base.samples]

        targets = torch.as_tensor(train_base.targets, dtype=torch.long)
        generator = torch.Generator().manual_seed(split_seed)
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for class_id in range(len(class_names)):
            class_indices = torch.where(targets == class_id)[0]
            train_idx, val_idx, test_idx = _split_class_indices(
                class_indices=class_indices,
                paths=sample_paths,
                val_split=val_split,
                test_split=test_split,
                generator=generator,
                group_aware_split=group_aware_split,
            )

            train_indices.extend(train_idx.tolist())
            val_indices.extend(val_idx.tolist())
            test_indices.extend(test_idx.tolist())

        train_dataset = torch.utils.data.Subset(train_base, train_indices)
        val_dataset = torch.utils.data.Subset(eval_base, val_indices)
        test_dataset = torch.utils.data.Subset(eval_base, test_indices)

    train_sampler = None
    if use_weighted_sampler:
        train_targets = _extract_targets(train_dataset)
        if train_targets is not None:
            class_counts = torch.bincount(train_targets, minlength=len(class_names)).float()
            class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
            sample_weights = class_weights[train_targets]
            train_sampler = WeightedRandomSampler(
                weights=sample_weights.double(),
                num_samples=len(sample_weights),
                replacement=True,
            )

    effective_prefetch = prefetch_factor if num_workers > 0 else None
    effective_persistent = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
        persistent_workers=effective_persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
        persistent_workers=effective_persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch,
        persistent_workers=effective_persistent,
    )

    def _class_counts(ds: object) -> dict[str, int]:
        targets = _extract_targets(ds)
        if targets is None:
            return {name: 0 for name in class_names}
        counts = torch.bincount(targets, minlength=len(class_names)).tolist()
        return {class_names[i]: int(counts[i]) for i in range(len(class_names))}

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=class_names,
        split_summary={
            "train": _class_counts(train_dataset),
            "val": _class_counts(val_dataset),
            "test": _class_counts(test_dataset),
        },
    )
