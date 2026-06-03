from __future__ import annotations

import io
import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_REPO = "rajeevkumar75/truesource-image-detector"
DEFAULT_IMAGE_FILENAME = "image_detection_best.pth"
DEFAULT_IMAGE_CLASSES_FILENAME = "image_detection_best.classes.txt"
DEFAULT_TEXT_REPO = "rajeevkumar75/truesource-text-model"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "deepfake_images"
IMAGE_SIZE = 224
_MODELS_LOADED = {"image": False, "text": False}

USE_HF_API = os.getenv("USE_HF_API", "1") == "1"

def _hf_api_request(model_repo: str, data: bytes | dict) -> dict | list:
    token = os.getenv("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"https://api-inference.huggingface.co/models/{model_repo}"
    
    for _ in range(3):
        if isinstance(data, dict):
            response = requests.post(url, headers=headers, json=data)
        else:
            response = requests.post(url, headers=headers, data=data)
        
        if response.status_code == 503:  # Model is loading
            time.sleep(3)
            continue
            
        response.raise_for_status()
        return response.json()
        
    raise RuntimeError(f"HF API failed after retries for {model_repo}: {response.text}")


def discover_class_names(dataset_root: Path | None = None) -> list[str]:
    try:
        from huggingface_hub import hf_hub_download
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


def _load_ai_threshold(model_path: str) -> float:
    DEFAULT_AI_THRESHOLD = 0.5
    try:
        if Path(model_path).exists():
            config_file = Path(model_path) / "inference.json"
            if not config_file.exists():
                return DEFAULT_AI_THRESHOLD
            payload = json.loads(config_file.read_text(encoding="utf-8"))
        else:
            from huggingface_hub import hf_hub_download
            token = os.getenv("HF_TOKEN")
            cached = hf_hub_download(repo_id=model_path, filename="inference.json", token=token)
            payload = json.loads(Path(cached).read_text(encoding="utf-8"))
            
        return float(payload.get("ai_threshold", DEFAULT_AI_THRESHOLD))
    except Exception:
        return DEFAULT_AI_THRESHOLD


@lru_cache(maxsize=1)
def _get_image_model(model_path: str, class_count: int):
    import torch
    from src.image_detection.model_trainer import create_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=class_count, freeze_backbone=False)
    state_dict = torch.load(Path(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def _ensure_text_model(model_path: str) -> None:
    from src.text_detection.inference_utils import _load_model_and_tokenizer
    _load_model_and_tokenizer(model_path)


def warmup_models() -> dict[str, bool]:
    """Load models into memory so first user request is fast."""
    names = discover_class_names()
    loaded = {"image": False, "text": False}
    token = os.getenv("HF_TOKEN")

    if USE_HF_API:
        print("Using Hugging Face Inference API. Skipping local model load.")
        loaded["image"] = True
        _MODELS_LOADED["image"] = True
        loaded["text"] = True
        _MODELS_LOADED["text"] = True
        return loaded

    try:
        from huggingface_hub import hf_hub_download
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
    text_threshold = _load_ai_threshold(DEFAULT_TEXT_REPO)
    
    device = "cpu"
    if not USE_HF_API:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pass

    return {
        "image_model": True,
        "text_model": True,
        "video_model": True,
        "class_names": discover_class_names(),
        "device": device,
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
    
    if USE_HF_API:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'JPEG')
        img_bytes = img_byte_arr.getvalue()
        try:
            api_result = _hf_api_request(DEFAULT_IMAGE_REPO, img_bytes)
            probabilities = {}
            for item in api_result:
                label = item.get("label", "").lower()
                probabilities[label] = float(item.get("score", 0.0))
            
            for n in names:
                if n not in probabilities:
                    probabilities[n] = 0.0
            
            p_fake = probabilities.get("fake", 0.0)
            if p_fake >= fake_threshold:
                predicted_label = "fake"
                decision = "Likely Fake"
            else:
                predicted_label = "real"
                decision = "Likely Real"
                
            confidence = probabilities.get(predicted_label, 0.0)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            return {
                "predicted_label": predicted_label,
                "confidence": confidence,
                "decision": decision,
                "p_fake": float(p_fake),
                "class_probabilities": probabilities,
                "model": "ResNet18 (HF API)",
                "inference_ms": elapsed_ms,
            }
        except Exception as e:
            print(f"HF API image error: {e}")
            raise e

    path = model_path
    from huggingface_hub import hf_hub_download
    if not path:
        path = hf_hub_download(repo_id=DEFAULT_IMAGE_REPO, filename=DEFAULT_IMAGE_FILENAME, token=os.getenv("HF_TOKEN"))
    if not Path(path).exists():
        raise FileNotFoundError(f"Image model not found: {path}")

    import torch
    from torchvision import transforms
    from src.image_detection.data_transformation import IMAGENET_MEAN, IMAGENET_STD
    from src.image_detection.inference_utils import binary_decision_from_two_class_probs
    
    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    model, device = _get_image_model(path, len(names))
    tensor = image_transform(image.convert("RGB")).unsqueeze(0).to(device)
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
    from src.video_detection.frame_extractor import extract_video_frames

    if USE_HF_API:
        max_frames = min(max_frames, 8)  # Avoid hitting API limits

    if not path and not USE_HF_API:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=DEFAULT_IMAGE_REPO, filename=DEFAULT_IMAGE_FILENAME, token=os.getenv("HF_TOKEN"))
    if not USE_HF_API and not Path(path).exists():
        raise FileNotFoundError(f"Video model not found: {path}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        frame_images = extract_video_frames(video_path=tmp_path, max_frames=max_frames)
        if not frame_images:
            raise ValueError("Could not extract frames from video.")

        if USE_HF_API:
            frame_fake_probs = []
            for img in frame_images:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()
                api_result = _hf_api_request(DEFAULT_IMAGE_REPO, img_bytes)
                
                p_fake = 0.0
                for item in api_result:
                    if item.get("label", "").lower() == "fake":
                        p_fake = float(item.get("score", 0.0))
                frame_fake_probs.append(p_fake)
            
            if frame_aggregation == "mean_max":
                mean_prob = sum(frame_fake_probs) / len(frame_fake_probs)
                max_prob = max(frame_fake_probs)
                aggregated_fake = (mean_max_weight * max_prob) + ((1.0 - mean_max_weight) * mean_prob)
            elif frame_aggregation == "max":
                aggregated_fake = max(frame_fake_probs)
            else:
                aggregated_fake = sum(frame_fake_probs) / len(frame_fake_probs)
            
            fake_idx = 0 if len(names) > 0 and names[0].lower() == 'fake' else 1
            if 'fake' in names:
                fake_idx = names.index('fake')
                
            real_idx = 1 - fake_idx
            
            synthetic_probs = [0.0, 0.0]
            synthetic_probs[fake_idx] = float(aggregated_fake)
            synthetic_probs[real_idx] = max(0.0, 1.0 - aggregated_fake)
            
            if synthetic_probs[fake_idx] >= fake_threshold:
                predicted_label = names[fake_idx]
                decision = "Likely Fake"
            else:
                predicted_label = names[real_idx]
                decision = "Likely Real"
                
            p_fake_val = synthetic_probs[fake_idx]
            
            confidence = float(
                aggregated_fake
                if predicted_label == names[fake_idx]
                else max(0.0, 1.0 - aggregated_fake)
            )
            class_probability_map = {
                names[i]: float(synthetic_probs[i]) for i in range(2)
            }
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            return {
                "predicted_label": predicted_label,
                "confidence": confidence,
                "decision": decision,
                "p_fake": float(p_fake_val),
                "class_probabilities": class_probability_map,
                "total_frames_used": len(frame_images),
                "frame_aggregation": frame_aggregation,
                "model": "Frame-level ResNet18 (HF API)",
                "inference_ms": elapsed_ms,
            }

        import torch
        from torchvision import transforms
        from src.image_detection.data_transformation import IMAGENET_MEAN, IMAGENET_STD
        from src.image_detection.inference_utils import aggregate_fake_probability, binary_decision_from_two_class_probs, resolve_fake_class_index

        image_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        model, device = _get_image_model(path, len(names))
        frame_tensors = torch.stack(
            [image_transform(image) for image in frame_images]
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
        threshold = _load_ai_threshold(path)

    if USE_HF_API:
        try:
            api_result = _hf_api_request(DEFAULT_TEXT_REPO, {"inputs": text})
            if isinstance(api_result, list) and len(api_result) > 0 and isinstance(api_result[0], list):
                api_result = api_result[0]
            
            probabilities = {}
            for item in api_result:
                label = item.get("label", "").lower()
                probabilities[label] = float(item.get("score", 0.0))
                
            p_ai = probabilities.get("ai", 0.0)
            if p_ai >= threshold:
                predicted_label = "ai"
                decision = "Likely AI-generated"
            else:
                predicted_label = "human"
                decision = "Likely Human-written"
                
            confidence = probabilities.get(predicted_label, 0.0)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            return {
                "predicted_label": predicted_label,
                "confidence": confidence,
                "decision": decision,
                "class_probabilities": probabilities,
                "p_ai": float(p_ai),
                "word_count": len(text.split()),
                "model": "DistilBERT (HF API)",
                "inference_ms": elapsed_ms,
                "ai_threshold": threshold,
            }
        except Exception as e:
            print(f"HF API text error: {e}")
            raise e

    from src.text_detection.predict import predict_text
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
