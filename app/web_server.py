from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, redirect, request, send_from_directory
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference_service import (  # noqa: E402
    get_app_status,
    get_models_loaded,
    predict_image_upload,
    predict_text_input,
    predict_video_upload,
    warmup_models,
)

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config.update(MAX_CONTENT_LENGTH=100 * 1024 * 1024)


@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/login")
def login_redirect():
    return redirect("/")


@app.route("/<path:asset_path>")
def static_assets(asset_path: str):
    if asset_path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    target = FRONTEND_DIR / asset_path
    if target.is_file():
        return send_from_directory(FRONTEND_DIR, asset_path)
    return jsonify({"error": "Not found"}), 404


@app.get("/api/status")
def api_status():
    status = get_app_status()
    status["models_loaded"] = get_models_loaded()
    return jsonify(status)


@app.post("/api/models/warmup")
def api_warmup():
    try:
        loaded = warmup_models()
        return jsonify({"message": "Models warmed up", "loaded": loaded})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/predict/image")
def api_predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
        fake_threshold = float(request.form.get("fake_threshold", 0.42))
        result = predict_image_upload(image=image, fake_threshold=fake_threshold)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.post("/api/predict/video")
def api_predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    suffix = Path(uploaded.filename).suffix
    try:
        result = predict_video_upload(
            video_bytes=uploaded.read(),
            suffix=suffix,
            fake_threshold=float(request.form.get("fake_threshold", 0.42)),
            max_frames=int(request.form.get("max_frames", 32)),
            frame_aggregation=request.form.get("frame_aggregation", "mean_max"),
            mean_max_weight=float(request.form.get("mean_max_weight", 0.45)),
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.post("/api/predict/text")
def api_predict_text():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Text is required"}), 400

    ai_threshold = payload.get("ai_threshold")
    try:
        result = predict_text_input(
            text=text,
            ai_threshold=float(ai_threshold) if ai_threshold is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    host = os.environ.get("TRUESOURCE_HOST", "127.0.0.1")
    port = int(os.environ.get("TRUESOURCE_PORT", "5000"))
    debug = os.environ.get("TRUESOURCE_DEBUG", "1") == "1"
    print(f"TrueSource web app: http://{host}:{port}")
    print("Loading ML models into memory…")
    try:
        loaded = warmup_models()
        print(f"Models ready: {loaded}")
    except Exception as exc:  # noqa: BLE001
        print(f"Model warmup warning: {exc}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
