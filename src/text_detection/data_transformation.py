from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.text_detection.data_ingestion import TextRecord, load_text_dataset

CLASS_NAMES = ("human", "ai")


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        records: list[TextRecord],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self.tokenizer(
            record.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.label_to_id[record.label], dtype=torch.long)
        return item


@dataclass(frozen=True)
class TextDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: tuple[str, str]
    split_summary: dict[str, dict[str, int]]
    tokenizer: PreTrainedTokenizerBase


def _stratified_split_indices(
    labels: list[int],
    val_split: float,
    test_split: float,
    seed: int,
    small_dataset: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices_by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        indices_by_class.setdefault(label, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for class_indices in indices_by_class.values():
        count = len(class_indices)
        order = torch.randperm(count, generator=generator).tolist()
        shuffled = [class_indices[i] for i in order]

        if small_dataset:
            n_test = 1 if count >= 3 else 0
            n_val = 1 if count >= 2 else 0
        else:
            n_test = max(1, int(count * test_split)) if count >= 3 else 0
            n_val = max(1, int(count * val_split)) if count >= 2 else 0
        if count - n_val - n_test < 1:
            n_val = min(n_val, max(0, count - 1))
            n_test = min(n_test, max(0, count - n_val - 1))

        test_idx.extend(shuffled[:n_test])
        val_idx.extend(shuffled[n_test : n_test + n_val])
        train_idx.extend(shuffled[n_test + n_val :])

    if not train_idx:
        all_indices = list(range(len(labels)))
        perm = torch.randperm(len(all_indices), generator=generator).tolist()
        ordered = [all_indices[i] for i in perm]
        train_idx = ordered[:-2] if len(ordered) > 2 else ordered[:1]
        val_idx = ordered[-2:-1] if len(ordered) > 1 else []
        test_idx = ordered[-1:] if len(ordered) > 1 else []

    return train_idx, val_idx, test_idx


def build_text_dataloaders(
    data_path: str,
    model_name: str,
    max_length: int = 256,
    batch_size: int = 16,
    val_split: float = 0.1,
    test_split: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 0,
    small_dataset: bool = False,
) -> TextDataLoaders:
    records = load_text_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    full_dataset = TextClassificationDataset(records, tokenizer=tokenizer, max_length=max_length)
    labels = [CLASS_NAMES.index(record.label) for record in records]

    train_indices, val_indices, test_indices = _stratified_split_indices(
        labels=labels,
        val_split=val_split,
        test_split=test_split,
        seed=split_seed,
        small_dataset=small_dataset,
    )

    def _count_labels(indices: list[int]) -> dict[str, int]:
        counts = {name: 0 for name in CLASS_NAMES}
        for idx in indices:
            counts[CLASS_NAMES[labels[idx]]] += 1
        return counts

    split_summary = {
        "train": _count_labels(train_indices),
        "val": _count_labels(val_indices),
        "test": _count_labels(test_indices),
    }

    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(full_dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return TextDataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=CLASS_NAMES,
        split_summary=split_summary,
        tokenizer=tokenizer,
    )
