"""Quick evaluation: in-distribution test split + holdout samples."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForSequenceClassification

from src.text_detection import (
    build_text_dataloaders,
    evaluate_text_model,
    load_ai_threshold,
    load_text_dataset,
    predict_text,
)

HOLDOUT = [
    ("human", "Yesterday I spilled coffee on my laptop keyboard and now half the keys stick."),
    ("human", "We argued about whose turn it is to take out the trash. It was mine."),
    ("human", "The dentist said I need to floss more. Same lecture every visit."),
    ("human", "Caught the 6am train because I could not sleep anyway."),
    ("human", "My sister borrowed my hoodie three weeks ago and still has it."),
    ("ai", "This document provides a structured overview of key enterprise deployment considerations."),
    ("ai", "The subsequent analysis synthesizes empirical findings with theoretical frameworks."),
    ("ai", "Users should consult official documentation for authoritative configuration guidance."),
    ("ai", "In summary, integrating these components yields a robust scalable architecture."),
    ("ai", "The following sections delineate best practices for operational excellence."),
]


def main() -> None:
    data_path = "data/text data/all.jsonl"
    model_path = "models/text_detection_best"

    records = load_text_dataset(data_path)
    print(f"Dataset: {len(records)} samples")

    if Path(model_path).exists():
        loaders = build_text_dataloaders(
            data_path, "distilbert-base-uncased", batch_size=8, num_workers=0
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        threshold = load_ai_threshold(model_path)
        print("Tuned AI threshold:", threshold)
        print(
            "Test split:",
            evaluate_text_model(model, loaders.test, ai_threshold=threshold),
        )
        print("Split summary:", loaders.split_summary)
    else:
        print("No saved model at", model_path)

    correct = 0
    print("\nHoldout (not in training file):")
    for expected, text in HOLDOUT:
        result = predict_text(text, model_path=model_path)
        ok = result.predicted_label == expected
        correct += int(ok)
        p_ai = result.class_probabilities["ai"]
        flag = "low conf" if "low confidence" in result.decision else ""
        print(
            f"  {expected:5} -> {result.predicted_label:5} "
            f"P(ai)={p_ai:.3f} {flag} {'OK' if ok else 'MISS'}"
        )
    print(f"\nHoldout accuracy: {correct}/{len(HOLDOUT)} ({100 * correct / len(HOLDOUT):.0f}%)")


if __name__ == "__main__":
    main()
