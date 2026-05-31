"""
Upload TrueSource models to the Hugging Face Hub.

Setup:
  pip install huggingface_hub
  huggingface-cli login

Examples:
  python scripts/upload_to_huggingface.py --repo-id YOUR_USERNAME/truesource-text --text-only
  python scripts/upload_to_huggingface.py --repo-id YOUR_USERNAME/truesource-image --image-only
  python scripts/upload_to_huggingface.py --repo-id YOUR_USERNAME/truesource --both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_MODEL_DIR = PROJECT_ROOT / "models" / "text_detection_best"
IMAGE_MODEL_FILE = PROJECT_ROOT / "models" / "image_detection_best.pth"
IMAGE_CLASSES_FILE = IMAGE_MODEL_FILE.with_suffix(".classes.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload TrueSource models to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hub repo id, e.g. yourusername/truesource-text-detector",
    )
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--private", action="store_true", help="Create/use a private repo")
    args = parser.parse_args()

    upload_text = args.text_only or args.both or (not args.text_only and not args.image_only)
    upload_image = args.image_only or args.both

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Install: pip install huggingface_hub")
        print("Then run: huggingface-cli login")
        sys.exit(1)

    api = HfApi()
    create_repo(args.repo_id, exist_ok=True, private=args.private)

    if upload_text:
        if not TEXT_MODEL_DIR.is_dir():
            print(f"Text model not found: {TEXT_MODEL_DIR}")
            sys.exit(1)
        print(f"Uploading text model from {TEXT_MODEL_DIR} ...")
        api.upload_folder(
            folder_path=str(TEXT_MODEL_DIR),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload DistilBERT human vs AI text detector",
        )
        print("Text model uploaded.")

    if upload_image:
        if not IMAGE_MODEL_FILE.is_file():
            print(f"Image weights not found: {IMAGE_MODEL_FILE}")
            sys.exit(1)
        print(f"Uploading {IMAGE_MODEL_FILE.name} ...")
        api.upload_file(
            path_or_fileobj=str(IMAGE_MODEL_FILE),
            path_in_repo=f"image/{IMAGE_MODEL_FILE.name}",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Upload ResNet18 deepfake image weights",
        )
        if IMAGE_CLASSES_FILE.is_file():
            api.upload_file(
                path_or_fileobj=str(IMAGE_CLASSES_FILE),
                path_in_repo=f"image/{IMAGE_CLASSES_FILE.name}",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message="Upload class labels",
            )
        print("Image model uploaded.")

    print(f"\nDone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
