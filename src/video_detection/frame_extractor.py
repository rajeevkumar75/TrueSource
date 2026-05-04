from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image


def extract_video_frames(
    video_path: str,
    max_frames: int = 32,
    min_side: int = 224,
) -> list[Image.Image]:
    """
    Extract uniformly sampled frames from a video file.

    Args:
        video_path: Path to input video.
        max_frames: Maximum number of frames to return.
        min_side: Frames with a smaller side below this value are upscaled.
    """
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file.resolve()}")

    capture = cv2.VideoCapture(str(video_file))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_file.resolve()}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = max(frame_count, 1)
    target_frames = max(1, max_frames)

    step = max(frame_count // target_frames, 1)
    selected_indices = list(range(0, frame_count, step))[:target_frames]
    selected_set = set(selected_indices)

    extracted: list[Image.Image] = []
    current_index = 0

    while capture.isOpened() and len(extracted) < target_frames:
        ok, frame_bgr = capture.read()
        if not ok:
            break

        if current_index in selected_set:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            smaller_side = min(height, width)

            if smaller_side < min_side and smaller_side > 0:
                scale = min_side / smaller_side
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(
                    frame_rgb,
                    (new_width, new_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            extracted.append(Image.fromarray(frame_rgb))

        current_index += 1

    capture.release()

    if not extracted:
        raise ValueError(f"No frames could be extracted from: {video_file.resolve()}")

    return extracted
