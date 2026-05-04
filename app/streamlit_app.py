from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

# Ensure project root is importable when launching via `streamlit run app/streamlit_app.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_detection.data_transformation import IMAGENET_MEAN, IMAGENET_STD
from src.image_detection.inference_utils import binary_decision_from_two_class_probs
from src.image_detection.model_trainer import create_model
from src.video_detection.video_classifier import predict_video


DEFAULT_MODEL_PATH = "models\image_detection_best.pth"
DEFAULT_DATASET_ROOT = "data/deepfake_images"
IMAGE_SIZE = 224


def _discover_class_names(dataset_root: str) -> list[str]:
    classes_file = Path(DEFAULT_MODEL_PATH).with_suffix(".classes.txt")
    if classes_file.exists():
        file_classes = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(file_classes) >= 2:
            return file_classes

    train_dir = Path(dataset_root) / "train"
    if not train_dir.exists():
        train_dir = Path(dataset_root) / "training"
    if train_dir.exists():
        class_dirs = [path.name for path in train_dir.iterdir() if path.is_dir()]
        if class_dirs:
            return sorted(class_dirs)

    root_dir = Path(dataset_root)
    root_class_dirs = [path.name for path in root_dir.iterdir() if path.is_dir()]
    if root_class_dirs:
        return sorted(root_class_dirs)
    return ["fake", "real"]


@st.cache_resource(show_spinner=False)
def _load_model(model_path: str, class_count: int) -> tuple[torch.nn.Module, torch.device]:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=class_count, freeze_backbone=False)
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def _predict_image(image: Image.Image, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    return probabilities


def main() -> None:
    st.set_page_config(page_title="Deepfake Detector", page_icon="🧪", layout="centered")
    st.title("Deepfake Detector")
    st.caption("Run image or video deepfake prediction using your trained model.")

    with st.sidebar:
        st.subheader("Settings")
        model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
        dataset_root = st.text_input("Dataset root (for auto class names)", value=DEFAULT_DATASET_ROOT)
        class_names_text = st.text_input(
            "Class names (comma-separated)",
            value=", ".join(_discover_class_names(dataset_root)),
            help="Order must match the class order used during training.",
        )
        max_video_frames = st.slider(
            "Max frames for video inference",
            min_value=8,
            max_value=96,
            value=32,
            step=8,
        )
        st.caption("Reduce false \"real\" on deepfakes")
        fake_threshold = st.slider(
            "P(fake) threshold (lower → more fake alerts)",
            min_value=0.25,
            max_value=0.55,
            value=0.42,
            step=0.01,
            help="Binary decision: predict fake when aggregated P(fake) ≥ this value (default below 0.5).",
        )
        frame_aggregation = st.selectbox(
            "Video frame aggregation",
            options=["mean_max", "max", "mean"],
            index=0,
            help="mean_max blends mean and max P(fake) per frame so a few artifact frames surface; "
            "max is strictest for fakes.",
        )
        mean_max_blend = st.slider(
            "mean_max blend (weight on mean)",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Rest goes to max(frame P(fake)). Higher max weight catches localized deepfake artifacts.",
        )

    class_names = [item.strip() for item in class_names_text.split(",") if item.strip()]
    if len(class_names) < 2:
        st.error("Please provide at least 2 class names.")
        st.stop()

    image_tab, video_tab = st.tabs(["Image Detection", "Video Detection"])

    with image_tab:
        uploaded_image = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            accept_multiple_files=False,
            key="image_uploader",
        )

        if uploaded_image:
            image = Image.open(io.BytesIO(uploaded_image.read())).convert("RGB")
            st.image(image, caption="Uploaded image", use_container_width=True)

            if st.button("Run image prediction", type="primary"):
                try:
                    model, device = _load_model(model_path=model_path, class_count=len(class_names))
                    probabilities = _predict_image(image=image, model=model, device=device)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Image prediction failed: {exc}")
                else:
                    predicted_label, decision, _p_fake = binary_decision_from_two_class_probs(
                        probabilities,
                        class_names,
                        fake_probability_threshold=fake_threshold,
                    )
                    confidence = float(
                        probabilities[class_names.index(predicted_label)].item()
                    )

                    st.success(
                        f"Image prediction: **{predicted_label}** ({confidence:.2%} confidence) | "
                        f"Decision: **{decision}**"
                    )
                    st.subheader("Class probabilities")
                    for idx, label in enumerate(class_names):
                        st.progress(
                            float(probabilities[idx].item()),
                            text=f"{label}: {probabilities[idx].item():.2%}",
                        )
        else:
            st.info("Choose an image file to run image prediction.")

    with video_tab:
        uploaded_video = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            accept_multiple_files=False,
            key="video_uploader",
        )

        if uploaded_video:
            st.video(uploaded_video)

            if st.button("Run video prediction", type="primary"):
                tmp_path: str | None = None
                try:
                    suffix = Path(uploaded_video.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                        tmp_file.write(uploaded_video.getbuffer())
                        tmp_path = tmp_file.name

                    result = predict_video(
                        video_path=tmp_path,
                        model_path=model_path,
                        class_names=class_names,
                        image_size=IMAGE_SIZE,
                        max_frames=max_video_frames,
                        fake_probability_threshold=fake_threshold,
                        frame_aggregation=frame_aggregation,
                        mean_max_weight=mean_max_blend,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Video prediction failed: {exc}")
                else:
                    st.success(
                        f"Video predicted class: **{result.predicted_label}** "
                        f"({result.confidence:.2%} confidence)"
                    )
                    st.info(
                        f"Final decision: **{result.decision}** | Frames used: **{result.total_frames_used}**"
                    )
                    st.subheader("Class probabilities")
                    for label, score in result.class_probabilities.items():
                        st.progress(float(score), text=f"{label}: {score:.2%}")
                finally:
                    if tmp_path and Path(tmp_path).exists():
                        Path(tmp_path).unlink(missing_ok=True)
        else:
            st.info("Choose a video file to run video prediction.")


if __name__ == "__main__":
    main()
