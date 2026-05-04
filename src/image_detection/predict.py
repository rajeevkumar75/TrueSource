from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.image_detection.data_transformation import IMAGENET_MEAN, IMAGENET_STD
from src.image_detection.inference_utils import binary_decision_from_two_class_probs
from src.image_detection.model_trainer import create_model


def predict_image(
    image_path: str,
    model_path: str,
    class_names: list[str],
    image_size: int = 224,
    fake_probability_threshold: float = 0.42,
) -> tuple[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(torch.load(Path(model_path), map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_label, _decision, _p_fake = binary_decision_from_two_class_probs(
        probabilities,
        class_names,
        fake_probability_threshold=fake_probability_threshold,
    )
    confidence = float(probabilities[class_names.index(predicted_label)].item())
    return predicted_label, confidence
