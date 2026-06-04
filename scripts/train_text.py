from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.text_detection import (
    TextTrainConfig,
    build_text_dataloaders,
    evaluate_text_model,
    load_text_dataset,
    train_text_model,
    tune_and_save_threshold,
)

MIN_RECOMMENDED_SAMPLES = 200


def _load_yaml_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def _pick(cli_value: object, yaml_value: object, default: object) -> object:
    """CLI wins when explicitly set; otherwise use config.yaml, then default."""
    if cli_value is not None:
        return cli_value
    if yaml_value is not None:
        return yaml_value
    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI vs human text detector")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="YAML config (optional; CLI flags override)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="JSONL file or directory with JSONL / human/ai folders",
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--val_split", type=float, default=None)
    parser.add_argument("--test_split", type=float, default=None)
    parser.add_argument("--split_seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_cfg = _load_yaml_config(args.config)

    dataset_cfg = yaml_cfg.get("dataset", {}) if isinstance(yaml_cfg.get("dataset"), dict) else {}
    training_cfg = yaml_cfg.get("training", {}) if isinstance(yaml_cfg.get("training"), dict) else {}

    data_path = _pick(
        args.data_path,
        dataset_cfg.get("data_path"),
        "data/text data/all.jsonl",
    )
    model_name = _pick(args.model_name, training_cfg.get("model_name"), "distilbert-base-uncased")

    records = load_text_dataset(str(data_path))
    if len(records) < MIN_RECOMMENDED_SAMPLES:
        print(
            f"Warning: only {len(records)} samples found. "
            f"For reliable detection, use at least {MIN_RECOMMENDED_SAMPLES} "
            "balanced human and AI examples."
        )

    val_split = float(_pick(args.val_split, dataset_cfg.get("val_split"), 0.1))
    test_split = float(_pick(args.test_split, dataset_cfg.get("test_split"), 0.1))
    if len(records) < MIN_RECOMMENDED_SAMPLES:
        # Keep more data for training when the dataset is tiny.
        val_split = min(val_split, 0.08)
        test_split = min(test_split, 0.08)
        print(
            f"Small dataset: using val_split={val_split}, test_split={test_split} "
            "(1 sample per class max for val/test)."
        )

    dataloaders = build_text_dataloaders(
        data_path=str(data_path),
        model_name=str(model_name),
        max_length=int(_pick(args.max_length, dataset_cfg.get("max_length"), 256)),
        batch_size=int(_pick(args.batch_size, training_cfg.get("batch_size"), 8)),
        val_split=val_split,
        test_split=test_split,
        split_seed=int(_pick(args.split_seed, dataset_cfg.get("split_seed"), 42)),
        num_workers=args.num_workers,
        small_dataset=len(records) < MIN_RECOMMENDED_SAMPLES,
    )

    config = TextTrainConfig(
        epochs=int(_pick(args.epochs, training_cfg.get("epochs"), 5)),
        learning_rate=float(_pick(args.learning_rate, training_cfg.get("learning_rate"), 2e-5)),
        weight_decay=float(_pick(args.weight_decay, training_cfg.get("weight_decay"), 0.01)),
        model_name=str(model_name),
        model_path=str(
            _pick(args.model_path, training_cfg.get("model_path"), "models/text_detection_best")
        ),
        device=str(_pick(args.device, training_cfg.get("device"), "auto")),
        early_stopping_patience=int(
            _pick(args.early_stopping_patience, training_cfg.get("early_stopping_patience"), 3)
        ),
        use_class_weights=bool(training_cfg.get("use_class_weights", True)),
        classifier_dropout=float(training_cfg.get("classifier_dropout", 0.2)),
    )

    print(f"Training for {config.epochs} epochs (config: {args.config})")

    model, train_metrics = train_text_model(
        train_loader=dataloaders.train,
        val_loader=dataloaders.val,
        num_labels=len(dataloaders.class_names),
        config=config,
        tokenizer=dataloaders.tokenizer,
    )
    ai_threshold = tune_and_save_threshold(model, dataloaders.val, config.model_path)
    test_metrics = evaluate_text_model(model, dataloaders.test, ai_threshold=ai_threshold)

    class_file = Path(config.model_path) / "class_labels.txt"
    class_file.write_text("\n".join(dataloaders.class_names), encoding="utf-8")

    print("\nClass names:", dataloaders.class_names)
    print("Split summary:", dataloaders.split_summary)
    print("Training metrics:", train_metrics)
    print("Tuned AI threshold:", ai_threshold)
    print("Test metrics:", test_metrics)

    test_bal_acc = float(test_metrics.get("test_balanced_accuracy", 0.0))
    if test_bal_acc < 0.85:
        print(
            "\nNote: test accuracy is weak on only "
            f"{sum(dataloaders.split_summary['test'].values())} held-out samples. "
            "This is common with ~40 examples. Add more real data, then re-run "
            "`python train_text.py` and `python scripts/eval_text_model.py`."
        )
    print(f"\nSaved model to: {config.model_path}")


if __name__ == "__main__":
    main()
