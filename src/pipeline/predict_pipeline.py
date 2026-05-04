from __future__ import annotations

import argparse
from pathlib import Path

from src.video_detection import predict_video


def _discover_class_names(data_root: str) -> list[str]:
    train_dir = Path(data_root) / "train"
    if not train_dir.exists():
        train_dir = Path(data_root) / "training"
    if not train_dir.exists():
        return ["fake", "real"]
    class_dirs = sorted([path.name for path in train_dir.iterdir() if path.is_dir()])
    return class_dirs or ["fake", "real"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict whether a video is real or deepfake/AI-generated"
    )
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/video_detection_frame_model.pth",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/deepfake_images",
        help="Used to infer class names from train/training folder",
    )
    parser.add_argument(
        "--class_names",
        type=str,
        default="",
        help="Comma-separated class names. If empty, auto-detected from data_dir/train",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument(
        "--fake_threshold",
        type=float,
        default=0.42,
        help="Binary decision: predict fake when aggregated P(fake) ≥ this (video) or P(fake) ≥ this (2-class image)",
    )
    parser.add_argument(
        "--frame_aggregation",
        type=str,
        default="mean_max",
        choices=["mean", "max", "mean_max"],
        help="How to combine per-frame probabilities for video",
    )
    parser.add_argument(
        "--mean_max_weight",
        type=float,
        default=0.45,
        help="In mean_max mode, blend weight on mean vs max (0=max only, 1=mean only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.class_names.strip():
        class_names = [item.strip() for item in args.class_names.split(",") if item.strip()]
    else:
        classes_file = Path(args.model_path).with_suffix(".classes.txt")
        if classes_file.exists():
            class_names = [
                line.strip()
                for line in classes_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            class_names = _discover_class_names(args.data_dir)

    prediction = predict_video(
        video_path=args.video_path,
        model_path=args.model_path,
        class_names=class_names,
        image_size=args.image_size,
        max_frames=args.max_frames,
        fake_probability_threshold=args.fake_threshold,
        frame_aggregation=args.frame_aggregation,
        mean_max_weight=args.mean_max_weight,
    )

    print("\nVideo classification result")
    print(f"- Predicted class: {prediction.predicted_label}")
    print(f"- Confidence: {prediction.confidence:.2%}")
    print(f"- Final decision: {prediction.decision}")
    print(f"- Frames used: {prediction.total_frames_used}")
    print("- Class probabilities:")
    for label, score in prediction.class_probabilities.items():
        print(f"  - {label}: {score:.2%}")


if __name__ == "__main__":
    main()
