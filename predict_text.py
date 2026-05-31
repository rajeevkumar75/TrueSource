from __future__ import annotations

import argparse
import json

from src.text_detection.predict import predict_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether text is AI- or human-generated")
    parser.add_argument("--text", type=str, default=None, help="Text to classify")
    parser.add_argument("--file", type=str, default=None, help="Path to a .txt file to classify")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/text_detection_best",
    )
    parser.add_argument("--ai_threshold", type=float, default=0.5)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            text = handle.read()
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Provide --text or --file")

    result = predict_text(
        text=text,
        model_path=args.model_path,
        max_length=args.max_length,
        ai_threshold=args.ai_threshold,
    )
    print(
        json.dumps(
            {
                "predicted_label": result.predicted_label,
                "confidence": round(result.confidence, 4),
                "decision": result.decision,
                "probabilities": {
                    key: round(value, 4) for key, value in result.class_probabilities.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
