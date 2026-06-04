import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [

    # Root files
    "README.md",
    "requirements.txt",
    "setup.py",
    ".gitignore",

    # Configs
    "configs/config.yaml",
    "configs/params.yaml",

    # Data folders
    "data/.gitkeep",

    # Models
    "models/.gitkeep",

    # Notebooks
    "notebooks/experiments.ipynb",

    # Source code
    "src/__init__.py",

    # Image detection module
    "src/image_detection/__init__.py",
    "src/image_detection/data_ingestion.py",
    "src/image_detection/data_transformation.py",
    "src/image_detection/model_trainer.py",
    "src/image_detection/model_evaluation.py",
    "src/image_detection/predict.py",

    # Video detection module
    "src/video_detection/__init__.py",
    "src/video_detection/frame_extractor.py",
    "src/video_detection/pipeline.py",

    # Text detection module
    "src/text_detection/__init__.py",
    "src/text_detection/data_ingestion.py",
    "src/text_detection/model_trainer.py",
    "src/text_detection/predict.py",

    # Common utils
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/exception.py",
    "src/utils/common.py",

    # Pipeline
    "src/pipeline/__init__.py",
    "src/pipeline/train_pipeline.py",
    "src/pipeline/predict_pipeline.py",

    # App
    "app/web_server.py",
    "app/inference_service.py",

    # Scripts
    "scripts/train_image.py",
    "scripts/train_text.py",
    "scripts/predict_text.py",
    
    # Tests
    "tests/verify_local.py",
    "tests/test_hf_api.py",

    # Main entry
    "run_app.py",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"{filename} already exists")