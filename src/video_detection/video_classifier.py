from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision import transforms

from src.image_detection.data_transformation import IMAGENET_MEAN, IMAGENET_STD
from src.image_detection.inference_utils import (
    aggregate_fake_probability,
    binary_decision_from_two_class_probs,
    resolve_fake_class_index,
)
from src.image_detection.model_trainer import create_model
from src.video_detection.frame_extractor import extract_video_frames


@dataclass(frozen=True)
class VideoPrediction:
    predicted_label: str
    confidence: float
    class_probabilities: dict[str, float]
    decision: str
    total_frames_used: int


def _to_binary_decision(predicted_label: str) -> str:
    normalized = predicted_label.lower().strip()
    if normalized in {"real", "authentic", "genuine"}:
        return "real"
    return "deepfake_or_ai_generated"


def predict_video(
    video_path: str,
    model_path: str,
    class_names: list[str],
    image_size: int = 224,
    max_frames: int = 32,
    *,
    fake_probability_threshold: float = 0.42,
    frame_aggregation: str = "mean_max",
    mean_max_weight: float = 0.45,
) -> VideoPrediction:
    if len(class_names) < 2:
        raise ValueError("At least two class names are required for prediction.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(torch.load(Path(model_path), map_location=device))
    model.to(device)
    model.eval()

    frame_images = extract_video_frames(video_path=video_path, max_frames=max_frames)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    frame_tensors = torch.stack([transform(image) for image in frame_images]).to(device)

    fake_idx = resolve_fake_class_index(class_names)

    with torch.no_grad():
        logits = model(frame_tensors)
        probabilities = torch.softmax(logits, dim=1)
        frame_fake = probabilities[:, fake_idx].cpu()
        aggregated_fake = aggregate_fake_probability(
            frame_fake,
            mode=frame_aggregation,
            mean_max_weight=mean_max_weight,
        )

    if len(class_names) == 2:
        real_idx = 1 - fake_idx
        synthetic_probs = torch.zeros(2, dtype=torch.float32)
        synthetic_probs[fake_idx] = float(aggregated_fake)
        synthetic_probs[real_idx] = max(0.0, 1.0 - aggregated_fake)
        predicted_label, decision, _p_fake = binary_decision_from_two_class_probs(
            synthetic_probs,
            class_names,
            fake_probability_threshold=fake_probability_threshold,
        )
        confidence = float(
            aggregated_fake
            if predicted_label == class_names[fake_idx]
            else max(0.0, 1.0 - aggregated_fake)
        )
        class_probability_map = {
            class_names[i]: float(synthetic_probs[i].item()) for i in range(2)
        }
    else:
        video_probabilities = probabilities.mean(dim=0).cpu()
        confidence, idx = torch.max(video_probabilities, dim=0)
        predicted_label = class_names[idx.item()]
        class_probability_map = {
            class_name: float(video_probabilities[class_index].item())
            for class_index, class_name in enumerate(class_names)
        }
        decision = _to_binary_decision(predicted_label)

    return VideoPrediction(
        predicted_label=predicted_label,
        confidence=confidence,
        class_probabilities=class_probability_map,
        decision=decision,
        total_frames_used=len(frame_images),
    )
