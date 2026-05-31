# TrueSource — Authenticity Intelligence Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**TrueSource** is an end-to-end AI platform for detecting synthetic and manipulated media. It identifies deepfakes in images and videos, detects AI-generated text, and provides both a powerful API and intuitive web interface for content authenticity verification.

## 🎯 Key Features

### 🖼️ **Image Deepfake Detection**
- Binary classification: Real vs. Fake images
- ResNet18-based deep learning model trained on balanced datasets
- Configurable confidence thresholds for flexible deployment
- Supports JPEG, PNG, and BMP formats

### 🎬 **Video Deepfake Detection**
- Frame extraction and multi-frame analysis
- Intelligent frame aggregation (mean/max pooling strategies)
- Temporal consistency evaluation
- Processes large video files with configurable frame limits

### 📝 **AI-Generated Text Detection**
- DistilBERT-based text classifier
- Human vs. AI-generated text classification
- Confidence scores and decision thresholds
- Handles variable-length text input (up to 256 tokens)

### 🌐 **Web Interface**
- Modern, responsive UI built with HTML/CSS/JavaScript
- Real-time model status and performance monitoring
- Multi-tab analysis interface for images, videos, and text
- REST API for programmatic access
- One-click model warmup and health checks

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Image Detection](#image-detection)
5. [Video Detection](#video-detection)
6. [Text Detection](#text-detection)
7. [Web Interface](#web-interface)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Model Details](#model-details)
11. [Troubleshooting](#troubleshooting)

---

## 🚀 Installation

### Prerequisites
- **Python 3.8+**
- **CUDA 11.7+** (optional, for GPU acceleration)
- **4GB+ RAM** (8GB+ recommended for video processing)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/TrueSource.git
cd TrueSource
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## ⚡ Quick Start

### 1. **Analyze an Image**
```bash
python predict_text.py --text "This is a sample text to analyze"
# or
python src/image_detection/predict.py --image path/to/image.jpg
```

### 2. **Start Web Server**
```bash
python run_web.py
# Visit http://localhost:5000
```

### 3. **Train Custom Models**
See detailed sections below for image, video, and text model training.

---

## 📁 Project Structure

```
TrueSource/
├── src/                          # Core ML modules
│   ├── image_detection/          # Image deepfake detection
│   │   ├── model_trainer.py      # Training pipeline
│   │   ├── model_evaluation.py   # Evaluation metrics
│   │   ├── data_ingestion.py     # Dataset loading
│   │   ├── data_transformation.py# Preprocessing
│   │   ├── inference_utils.py    # Prediction utilities
│   │   └── predict.py            # Single image prediction
│   ├── text_detection/           # AI text detection
│   │   ├── model_trainer.py      # DistilBERT fine-tuning
│   │   ├── model_evaluation.py   # Evaluation & threshold tuning
│   │   ├── data_ingestion.py     # Text dataset loading
│   │   ├── data_transformation.py# Tokenization & preprocessing
│   │   ├── inference_utils.py    # Inference helpers
│   │   └── predict.py            # Text prediction
│   ├── video_detection/          # Video deepfake detection
│   │   ├── frame_extractor.py    # Extract frames from video
│   │   ├── video_classifier.py   # Video-level prediction
│   │   └── pipeline.py           # Video processing pipeline
│   ├── pipeline/                 # Orchestration
│   │   ├── train_pipeline.py     # Full training pipeline
│   │   └── predict_pipeline.py   # Full prediction pipeline
│   └── utils/                    # Shared utilities
│       ├── logger.py             # Logging utilities
│       ├── exception.py          # Custom exceptions
│       └── common.py             # Common helpers
├── app/                          # Web application
│   ├── web_server.py             # Flask server
│   └── inference_service.py      # Model loading & inference
├── frontend/                     # Web UI
│   ├── index.html                # Main page
│   ├── css/style.css             # Styling
│   └── js/                       # Frontend logic
├── models/                       # Pre-trained models
│   ├── image_detection_best.pth  # Image model weights
│   └── text_detection_best/      # Text model directory
├── data/                         # Datasets
│   ├── img dataset/              # Image data
│   │   ├── train/                # Training images
│   │   ├── val/                  # Validation images
│   │   └── test/                 # Test images
│   └── text data/                # Text data
│       └── all.jsonl             # Text samples
├── configs/                      # Configuration files
│   ├── config.yaml               # Main config
│   ├── params.yaml               # Training parameters
│   └── users.json                # User management
├── scripts/                      # Utility scripts
│   ├── eval_text_model.py        # Model evaluation
│   └── upload_to_huggingface.py # Upload models
├── notebooks/                    # Jupyter notebooks
│   └── experiments.ipynb         # Experiments & exploration
├── main.py                       # Image model training entry point
├── train_text.py                 # Text model training entry point
├── predict_text.py               # Text prediction CLI
├── run_web.py                    # Web server entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🖼️ Image Detection

### Dataset Structure

Organize your image dataset as follows:

```
data/img dataset/
├── train/
│   ├── real/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── fake/
│       ├── deepfake1.jpg
│       ├── deepfake2.png
│       └── ...
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Training

```bash
# Basic training
python main.py --data_dir data/img\ dataset --epochs 10 --batch_size 32

# Advanced training with custom parameters
python main.py \
  --data_dir data/img\ dataset \
  --epochs 15 \
  --batch_size 16 \
  --learning_rate 2e-4 \
  --image_size 256 \
  --weight_decay 1e-4 \
  --label_smoothing 0.05
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/img dataset` | Path to dataset directory |
| `--epochs` | 10 | Number of training epochs |
| `--batch_size` | 32 | Batch size for training |
| `--image_size` | 224 | Input image size (square) |
| `--learning_rate` | 2e-4 | Learning rate for optimizer |
| `--weight_decay` | 1e-4 | L2 regularization coefficient |
| `--label_smoothing` | 0.05 | Label smoothing factor |
| `--num_workers` | 2 | Number of data loading workers |
| `--no_weighted_sampler` | False | Disable class-balanced sampling |
| `--no_group_aware_split` | False | Disable stratified split |

### Inference

```bash
# Predict single image
python src/image_detection/predict.py \
  --image path/to/image.jpg \
  --model_path models/image_detection_best.pth \
  --fake_threshold 0.42

# Output:
# {
#   "predicted_label": "real",
#   "confidence": 0.89,
#   "class_probabilities": {"real": 0.89, "fake": 0.11}
# }
```

---

## 🎬 Video Detection

### How It Works

1. **Frame Extraction**: Extracts uniformly spaced frames from video
2. **Frame Classification**: Analyzes each frame with the image model
3. **Aggregation**: Combines frame predictions using configurable strategies
4. **Decision**: Outputs video-level classification with confidence

### Inference

```bash
# Analyze video
python src/video_detection/video_classifier.py \
  --video path/to/video.mp4 \
  --model_path models/image_detection_best.pth \
  --max_frames 32 \
  --fake_threshold 0.42
```

### Frame Aggregation Strategies

- **mean**: Average of all frame probabilities
- **max**: Maximum fake probability across frames
- **mean_max**: Weighted combination (default: 45% mean + 55% max)

### Output

```json
{
  "predicted_label": "real",
  "confidence": 0.87,
  "decision": "real",
  "total_frames_used": 32,
  "class_probabilities": {"real": 0.87, "fake": 0.13}
}
```

---

## 📝 Text Detection

### Dataset Format

#### Option 1: JSONL Format (Recommended)

Create `data/text data/all.jsonl` with one JSON object per line:

```jsonl
{"text": "Machine learning is fascinating.", "label": "human"}
{"text": "Artificial intelligence powers modern technology.", "label": "human"}
{"text": "The rapid advancement of AI continues to reshape industries.", "label": "ai"}
{"text": "Natural language processing enables computers to understand text.", "label": "ai"}
```

#### Option 2: Text Files

Organize text files in folders:

```
data/text data/
├── human/
│   ├── document1.txt
│   ├── document2.txt
│   └── ...
└── ai/
    ├── generated1.txt
    ├── generated2.txt
    └── ...
```

### Training

```bash
# Basic training
python train_text.py --data_path "data/text data/all.jsonl" --epochs 5

# Advanced training
python train_text.py \
  --data_path "data/text data/all.jsonl" \
  --epochs 10 \
  --batch_size 16 \
  --learning_rate 2.0e-5 \
  --model_name distilbert-base-uncased \
  --max_length 512 \
  --early_stopping_patience 3
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `data/text data/all.jsonl` | Path to training data |
| `--epochs` | 5 | Number of training epochs |
| `--batch_size` | 8 | Batch size for training |
| `--learning_rate` | 2.0e-5 | Learning rate for optimizer |
| `--max_length` | 256 | Max token length |
| `--model_name` | `distilbert-base-uncased` | HuggingFace model ID |
| `--early_stopping_patience` | 3 | Epochs without improvement before stopping |
| `--model_path` | `models/text_detection_best` | Output model directory |

### Inference

```bash
# Analyze text string
python predict_text.py --text "This is a sample text to analyze"

# Analyze text file
python predict_text.py --file sample.txt

# Custom threshold
python predict_text.py \
  --text "Your text here" \
  --ai_threshold 0.6 \
  --model_path models/text_detection_best
```

### Output

```json
{
  "predicted_label": "human",
  "confidence": 0.92,
  "decision": "human",
  "probabilities": {
    "human": 0.92,
    "ai": 0.08
  }
}
```

### Threshold Tuning

Find optimal decision threshold on validation data:

```bash
python scripts/eval_text_model.py \
  --model_path models/text_detection_best \
  --tune_threshold
```

---

## 🌐 Web Interface

### Launch Server

```bash
python run_web.py
# Server starts on http://localhost:5000
```

### Features

1. **Home Tab**: Platform overview and model statistics
2. **Analyze Tab**: Upload images, videos, or paste text
3. **Features Tab**: Feature descriptions and capabilities
4. **About Tab**: Project information
5. **Contact Tab**: Support and feedback

### Web Interface Highlights

- **Real-time Status**: View loaded models and compute device
- **Adjustable Thresholds**: Control confidence thresholds for predictions
- **Batch Analysis**: Process multiple files sequentially
- **History**: Review recent predictions
- **Model Warmup**: Pre-load models for faster inference

---

## 🔌 API Reference

### Status Endpoint

```bash
GET /api/status
```

**Response:**
```json
{
  "status": "ready",
  "models_loaded": {"image": true, "text": true, "video": true},
  "device": "cuda"
}
```

### Model Warmup

```bash
POST /api/models/warmup
```

**Response:**
```json
{
  "message": "Models warmed up",
  "loaded": true
}
```

### Image Prediction

```bash
POST /api/predict/image
```

**Parameters:**
- `file` (multipart/form-data): Image file
- `fake_threshold` (float, optional): Confidence threshold (default: 0.42)

**Response:**
```json
{
  "predicted_label": "real",
  "confidence": 0.89,
  "decision": "real",
  "class_probabilities": {"real": 0.89, "fake": 0.11}
}
```

### Video Prediction

```bash
POST /api/predict/video
```

**Parameters:**
- `file` (multipart/form-data): Video file
- `fake_threshold` (float, optional): Confidence threshold (default: 0.42)
- `max_frames` (int, optional): Maximum frames to analyze (default: 32)

**Response:**
```json
{
  "predicted_label": "real",
  "confidence": 0.87,
  "decision": "real",
  "total_frames_used": 32,
  "class_probabilities": {"real": 0.87, "fake": 0.13}
}
```

### Text Prediction

```bash
POST /api/predict/text
```

**Parameters:**
- `text` (string): Text to analyze
- `ai_threshold` (float, optional): Confidence threshold (default: 0.5)

**Response:**
```json
{
  "predicted_label": "human",
  "confidence": 0.92,
  "decision": "human",
  "class_probabilities": {"human": 0.92, "ai": 0.08}
}
```

---

## ⚙️ Configuration

### Main Config (configs/config.yaml)

```yaml
dataset:
  data_path: "data/text data/all.jsonl"
  max_length: 256
  val_split: 0.1
  test_split: 0.1
  split_seed: 42

training:
  model_name: distilbert-base-uncased
  epochs: 5
  batch_size: 8
  learning_rate: 2.0e-5
  weight_decay: 0.01
  model_path: models/text_detection_best
  early_stopping_patience: 3
  use_class_weights: true
  classifier_dropout: 0.2

inference:
  ai_threshold: 0.5
  max_length: 256
```

### GPU Configuration

```bash
# Force CPU
export CUDA_VISIBLE_DEVICES=""

# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Mixed precision training (faster, less memory)
python train_text.py --use_fp16
```

---

## 🧠 Model Details

### Image Model

| Property | Value |
|----------|-------|
| **Architecture** | ResNet18 |
| **Input Size** | 224×224 pixels |
| **Classes** | Binary (Real/Fake) |
| **Output** | Class logits + probabilities |
| **Threshold** | 0.42 (optimized for balanced F1) |
| **Training** | ImageNet normalization + augmentation |

### Text Model

| Property | Value |
|----------|-------|
| **Architecture** | DistilBERT (base-uncased) |
| **Input** | Tokenized text (max 256 tokens) |
| **Classes** | Binary (Human/AI) |
| **Output** | Class logits + probabilities |
| **Threshold** | 0.5 (tunable) |
| **Training** | Fine-tuned on domain data |

### Video Model

| Property | Value |
|----------|-------|
| **Frame Processing** | ResNet18 (same as image model) |
| **Aggregation** | Mean + Max pooling |
| **Frames Analyzed** | Up to 32 (configurable) |
| **Decision** | Weighted combination of frame predictions |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**

```bash
# Reduce batch size
python train_text.py --batch_size 4 --max_length 128

# Use CPU
export CUDA_VISIBLE_DEVICES=""
python train_text.py
```

#### 2. **Models Not Loading**

```bash
# Check model files exist
ls -la models/

# Re-download model from HuggingFace
python scripts/upload_to_huggingface.py --download
```

#### 3. **Poor Prediction Accuracy**

- Ensure dataset is balanced (similar number of real/fake samples)
- Increase training epochs
- Tune confidence threshold based on validation metrics

```bash
python scripts/eval_text_model.py --tune_threshold
```

#### 4. **Video Processing Slow**

```bash
# Reduce frames analyzed
python src/video_detection/video_classifier.py --max_frames 16

# Use GPU
export CUDA_VISIBLE_DEVICES=0
```

#### 5. **Web Server Port Already in Use**

```bash
# Change port
python -c "from app.web_server import main; main(port=8080)"
```

---

## 📊 Evaluation & Metrics

### Image Model Evaluation

```bash
# Full evaluation on test set
python src/image_detection/model_evaluation.py --data_dir data/img\ dataset
```

### Text Model Evaluation

```bash
# Evaluate text model with metrics
python scripts/eval_text_model.py --model_path models/text_detection_best

# Tune threshold for optimal F1 score
python scripts/eval_text_model.py --model_path models/text_detection_best --tune_threshold
```

### Metrics Reported

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ResNet18** architecture from [TorchVision](https://pytorch.org/vision/)
- **DistilBERT** model from [HuggingFace Transformers](https://huggingface.co/distilbert-base-uncased)
- Community contributions and feedback



## Web App

we're woring on deployment...