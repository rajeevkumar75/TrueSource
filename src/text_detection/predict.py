from __future__ import annotations

from dataclasses import dataclass

import torch

from src.text_detection.data_transformation import CLASS_NAMES
from src.text_detection.inference_utils import (
    _load_model_and_tokenizer,
    decision_from_probabilities,
    load_ai_threshold,
)


@dataclass(frozen=True)
class TextPredictionResult:
    predicted_label: str
    confidence: float
    class_probabilities: dict[str, float]
    decision: str


def predict_text(
    text: str,
    model_path: str,
    max_length: int = 256,
    ai_threshold: float | None = None,
) -> TextPredictionResult:
    if not text.strip():
        raise ValueError("Text is empty.")

    threshold = ai_threshold if ai_threshold is not None else load_ai_threshold(model_path)
    model, tokenizer, device = _load_model_and_tokenizer(model_path)

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    probs = {label: float(probabilities[idx].item()) for idx, label in enumerate(CLASS_NAMES)}
    predicted_label, decision = decision_from_probabilities(probs, threshold)
    confidence = max(probs.values())

    return TextPredictionResult(
        predicted_label=predicted_label,
        confidence=confidence,
        class_probabilities=probs,
        decision=decision,
    )
