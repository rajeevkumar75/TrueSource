from __future__ import annotations

import re

import torch


_REAL_ALIASES = frozenset({"real", "authentic", "genuine", "original"})


def _normalize_label(name: str) -> str:
    return name.lower().strip()


def resolve_fake_class_index(class_names: list[str]) -> int:
    """Return the index of the fake / synthetic class for thresholding and aggregation."""
    if len(class_names) < 2:
        raise ValueError("At least two class names are required.")

    lowered = [_normalize_label(n) for n in class_names]

    for i, n in enumerate(lowered):
        if n in _REAL_ALIASES:
            continue
        if "fake" in n or "deepfake" in n or "forgery" in n or "synthetic" in n:
            return i
        if re.search(r"\bai\b", n) or "generated" in n or "spoof" in n:
            return i

    for i, n in enumerate(lowered):
        if n in _REAL_ALIASES:
            if len(class_names) == 2:
                return 1 - i
            continue

    return 0


def is_real_label(label: str) -> bool:
    n = _normalize_label(label)
    return n in _REAL_ALIASES or n == "real"


def aggregate_fake_probability(
    frame_probs_fake: torch.Tensor,
    mode: str,
    mean_max_weight: float,
) -> float:
    """
    Combine per-frame P(fake) into a single video-level score.

    mean_max_weight in [0, 1]: blend of mean(frame_probs) and max(frame_probs).
    mode: "mean" | "max" | "mean_max"
    """
    if frame_probs_fake.numel() == 0:
        return 0.0
    mean_max_weight = float(min(1.0, max(0.0, mean_max_weight)))
    m = mode.lower().strip()
    if m == "mean":
        return float(frame_probs_fake.mean().item())
    if m == "max":
        return float(frame_probs_fake.max().item())
    if m == "mean_max":
        mean_p = frame_probs_fake.mean()
        max_p = frame_probs_fake.max()
        blended = mean_max_weight * mean_p + (1.0 - mean_max_weight) * max_p
        return float(blended.item())
    raise ValueError(f"Unknown aggregation mode: {mode!r}")


def binary_decision_from_two_class_probs(
    probabilities: torch.Tensor,
    class_names: list[str],
    fake_probability_threshold: float,
) -> tuple[str, str, float]:
    """
    For binary classification, decide label using a threshold on P(fake).

    Using argmax alone biases toward the majority / \"safer\" class; a threshold below 0.5
    surfaces more fake predictions when P(fake) is uncertain.
    """
    if len(class_names) != 2:
        confidence, idx = torch.max(probabilities, dim=0)
        predicted = class_names[int(idx.item())]
        decision = "real" if is_real_label(predicted) else "deepfake_or_ai_generated"
        fake_idx = resolve_fake_class_index(class_names)
        p_fake = float(probabilities[fake_idx].item())
        return predicted, decision, p_fake

    fake_idx = resolve_fake_class_index(class_names)
    real_idx = 1 - fake_idx
    p_fake = float(probabilities[fake_idx].item())

    if p_fake >= fake_probability_threshold:
        predicted = class_names[fake_idx]
        decision = "deepfake_or_ai_generated"
    else:
        predicted = class_names[real_idx]
        decision = "real"

    return predicted, decision, p_fake
