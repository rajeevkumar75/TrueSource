from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download

from src.image_detection.data_transformation import IMAGENET_MEAN, IMAGENET_STD
from src.image_detection.inference_utils import (
    aggregate_fake_probability,
    binary_decision_from_two_class_probs,
    resolve_fake_class_index,
)
from src.image_detection.model_trainer import create_model
from src.text_detection.inference_utils import _load_model_and_tokenizer, load_ai_threshold
from src.text_detection.predict import predict_text
from src.video_detection.frame_extractor import extract_video_frames

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_REPO = "rajeevkumar75/truesource-image-detector"
DEFAULT_IMAGE_FILENAME = "image_detection_best.pth"
DEFAULT_IMAGE_CLASSES_FILENAME = "image_detection_best.classes.txt"
DEFAULT_TEXT_REPO = "rajeevkumar75/truesource-text-model"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "deepfake_images"
IMAGE_SIZE = 224
_MODELS_LOADED = {"image": False, "text": False}

_IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def discover_class_names(dataset_root: Path | None = None) -> list[str]:
    try:
        token = os.getenv("HF_TOKEN")
        cached_file = hf_hub_download(
            repo_id=DEFAULT_IMAGE_REPO, 
            filename=DEFAULT_IMAGE_CLASSES_FILENAME, 
            token=token
        )
        names = [
            line.strip()
            for line in Path(cached_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) >= 2:
            return names
    except Exception:
        pass

    root = dataset_root or DEFAULT_DATASET_ROOT
    train_dir = root / "train"
    if not train_dir.exists():
        train_dir = root / "training"
    if train_dir.exists():
        class_dirs = [path.name for path in train_dir.iterdir() if path.is_dir()]
        if class_dirs:
            return sorted(class_dirs)

    if root.exists():
        class_dirs = [path.name for path in root.iterdir() if path.is_dir()]
        if class_dirs:
            return sorted(class_dirs)
    return ["fake", "real"]


@lru_cache(maxsize=1)
def _get_image_model(model_path: str, class_count: int) -> tuple[torch.nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=class_count, freeze_backbone=False)
    state_dict = torch.load(Path(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def _ensure_text_model(model_path: str) -> None:
    _load_model_and_tokenizer(model_path)


def warmup_models() -> dict[str, bool]:
    """Load models into memory so first user request is fast."""
    names = discover_class_names()
    loaded = {"image": False, "text": False}
    token = os.getenv("HF_TOKEN")

    try:
        path = hf_hub_download(repo_id=DEFAULT_IMAGE_REPO, filename=DEFAULT_IMAGE_FILENAME, token=token)
        _get_image_model(path, len(names))
        loaded["image"] = True
        _MODELS_LOADED["image"] = True
    except Exception as e:
        print("Image model warmup failed:", e)

    try:
        _ensure_text_model(DEFAULT_TEXT_REPO)
        loaded["text"] = True
        _MODELS_LOADED["text"] = True
    except Exception as e:
        print("Text model warmup failed:", e)

    return loaded


def get_models_loaded() -> dict[str, bool]:
    return dict(_MODELS_LOADED)


def get_app_status() -> dict:
    text_threshold = 0.5
    try:
        text_threshold = load_ai_threshold(DEFAULT_TEXT_REPO)
    except Exception:  # noqa: BLE001
        pass

    return {
        "image_model": True,
        "text_model": True,
        "video_model": True,
        "class_names": discover_class_names(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": {
            "image": {
                "path": DEFAULT_IMAGE_REPO,
                "ready": _MODELS_LOADED["image"],
                "type": "ResNet18",
            },
            "video": {
                "path": DEFAULT_IMAGE_REPO,
                "ready": _MODELS_LOADED["image"],
                "type": "Frame-level ResNet18",
            },
            "text": {
                "path": DEFAULT_TEXT_REPO,
                "ready": _MODELS_LOADED["text"],
                "type": "DistilBERT",
                "ai_threshold": text_threshold,
            },
        },
    }


def predict_image_upload(
    image: Image.Image,
    class_names: list[str] | None = None,
    fake_threshold: float = 0.42,
    model_path: str | None = None,
) -> dict:
    started = time.perf_counter()
    names = class_names or discover_class_names()
    path = model_path
    if not path:
        path = hf_hub_download(repo_id=DEFAULT_IMAGE_REPO, filename=DEFAULT_IMAGE_FILENAME, token=os.getenv("HF_TOKEN"))
    if not Path(path).exists():
        raise FileNotFoundError(f"Image model not found: {path}")

    model, device = _get_image_model(path, len(names))
    tensor = _IMAGE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_label, decision, p_fake = binary_decision_from_two_class_probs(
        probabilities,
        names,
        fake_probability_threshold=fake_threshold,
    )
    confidence = float(probabilities[names.index(predicted_label)].item())
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "decision": decision,
        "p_fake": float(p_fake),
        "class_probabilities": {
            label: float(probabilities[idx].item()) for idx, label in enumerate(names)
        },
        "model": "ResNet18",
        "inference_ms": elapsed_ms,
    }


def predict_video_upload(
    video_bytes: bytes,
    suffix: str,
    class_names: list[str] | None = None,
    fake_threshold: float = 0.42,
    max_frames: int = 32,
    frame_aggregation: str = "mean_max",
    mean_max_weight: float = 0.45,
    model_path: str | None = None,
) -> dict:
    started = time.perf_counter()
    names = class_names or discover_class_names()
    path = model_path
    if not path:
        path = hf_hub_download(repo_id=DEFAULT_IMAGE_REPO, filename=DEFAULT_IMAGE_FILENAME, token=os.getenv("HF_TOKEN"))
    if not Path(path).exists():
        raise FileNotFoundError(f"Video model not found: {path}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        model, device = _get_image_model(path, len(names))
        frame_images = extract_video_frames(video_path=tmp_path, max_frames=max_frames)
        if not frame_images:
            raise ValueError("Could not extract frames from video.")

        frame_tensors = torch.stack(
            [_IMAGE_TRANSFORM(image) for image in frame_images]
        ).to(device)
        fake_idx = resolve_fake_class_index(names)

        with torch.no_grad():
            logits = model(frame_tensors)
            probabilities = torch.softmax(logits, dim=1)
            frame_fake = probabilities[:, fake_idx].cpu()
            aggregated_fake = aggregate_fake_probability(
                frame_fake,
                mode=frame_aggregation,
                mean_max_weight=mean_max_weight,
            )

        real_idx = 1 - fake_idx
        synthetic_probs = torch.zeros(2, dtype=torch.float32)
        synthetic_probs[fake_idx] = float(aggregated_fake)
        synthetic_probs[real_idx] = max(0.0, 1.0 - aggregated_fake)
        predicted_label, decision, p_fake = binary_decision_from_two_class_probs(
            synthetic_probs,
            names,
            fake_probability_threshold=fake_threshold,
        )
        confidence = float(
            aggregated_fake
            if predicted_label == names[fake_idx]
            else max(0.0, 1.0 - aggregated_fake)
        )
        class_probability_map = {
            names[i]: float(synthetic_probs[i].item()) for i in range(2)
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "decision": decision,
        "p_fake": float(p_fake),
        "class_probabilities": class_probability_map,
        "total_frames_used": len(frame_images),
        "frame_aggregation": frame_aggregation,
        "model": "Frame-level ResNet18",
        "inference_ms": elapsed_ms,
    }


def predict_text_input(
    text: str,
    ai_threshold: float | None = None,
    model_path: str | None = None,
) -> dict:
    started = time.perf_counter()
    path = model_path or DEFAULT_TEXT_REPO

    threshold = ai_threshold
    if threshold is None:
        threshold = load_ai_threshold(path)

    _ensure_text_model(path)
    result = predict_text(text=text, model_path=path, ai_threshold=threshold)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    return {
        "predicted_label": result.predicted_label,
        "confidence": result.confidence,
        "decision": result.decision,
        "class_probabilities": result.class_probabilities,
        "p_ai": result.class_probabilities.get("ai", 0.0),
        "word_count": len(text.split()),
        "model": "DistilBERT",
        "inference_ms": elapsed_ms,
        "ai_threshold": threshold,
    }
