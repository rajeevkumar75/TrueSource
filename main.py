from __future__ import annotations

import argparse
from pathlib import Path

from src.image_detection import (
    TrainConfig,
    build_dataloaders,
    evaluate_model,
    get_dataset_paths,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deepfake image detector")
    default_data_dir = "data/deepfake_images"
    if not Path(default_data_dir).exists():
        default_data_dir = "data/img dataset"
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data_dir,
        help="Path containing train/val/test folders",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Batches prefetched per worker (num_workers > 0)",
    )
    parser.add_argument(
        "--no_persistent_workers",
        action="store_true",
        help="Disable persistent dataloader workers between epochs",
    )
    parser.add_argument(
        "--no_weighted_sampler",
        action="store_true",
        help="Disable weighted random sampler for class-balanced training batches",
    )
    parser.add_argument(
        "--no_group_aware_split",
        action="store_true",
        help="Disable group-aware splitting for class-only datasets",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device: auto selects CUDA if available",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable mixed precision training (CUDA only)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio when dataset has only class folders",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Test split ratio when dataset has only class folders",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for class-wise split generation",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/image_detection_best.pth",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze feature extractor and train only classification head",
    )
    parser.add_argument(
        "--no_freeze_backbone",
        action="store_true",
        help="Deprecated alias to fine-tune full model",   
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable inverse-frequency class weights in the loss (weighted sampler is used instead)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=None,
        help="If set (e.g. 2.0), train with focal loss to emphasize hard fakes",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=4,
        help="Stop training if val_bal_acc does not improve",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum val_bal_acc improvement to reset patience",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_class_weights = not args.no_class_weights
    paths = get_dataset_paths(args.data_dir)
    dataloaders = build_dataloaders(
        dataset_paths=paths,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
        split_seed=args.split_seed,
        use_weighted_sampler=(not args.no_weighted_sampler) and (not use_class_weights),
        group_aware_split=not args.no_group_aware_split,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
    )

    freeze_backbone = args.freeze_backbone and not args.no_freeze_backbone
    config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        model_path=args.model_path,
        freeze_backbone=freeze_backbone,
        use_class_weights=use_class_weights,
        focal_gamma=args.focal_gamma,
        device=args.device,
        use_amp=not args.no_amp,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )

    model, train_metrics = train_model(
        train_loader=dataloaders.train,
        val_loader=dataloaders.val,
        num_classes=len(dataloaders.class_names),
        config=config,
    )
    test_metrics = evaluate_model(model, dataloaders.test)

    class_file = Path(args.model_path).with_suffix(".classes.txt")
    class_file.parent.mkdir(parents=True, exist_ok=True)
    class_file.write_text("\n".join(dataloaders.class_names), encoding="utf-8")

    print("\nClass names:", dataloaders.class_names)
    print("Split summary:", dataloaders.split_summary)
    print("Training metrics:", train_metrics)
    print("Test metrics:", test_metrics)
    print(f"Saved best model to: {args.model_path}")
    print(f"Saved class names to: {class_file}")


if __name__ == "__main__":
    main()
