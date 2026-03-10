"""
frame_extractor.py — Frame extraction helpers for Gruve Atlas video analysis.

Two extraction modes:
  1. extract_frames_with_timestamps() — 1-fps sampling, returns (path, seconds) pairs
     Used by GPT-4o, Qwen, Llama, Gemma (image-based VLMs).

  2. encode_video_base64() — encodes entire .mp4 as a base64 string
     Used by Gemini models (native video understanding via OpenRouter).
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# Constants (inlined — no dependency on external config)
# ---------------------------------------------------------------------------
FRAME_QUALITY   = 85    # JPEG quality 0–100
FRAME_MAX_WIDTH = 1024  # px — resize wider frames, preserve aspect ratio
MAX_FRAMES      = 10    # cap for 1-fps extraction (avoids huge payloads)


# ---------------------------------------------------------------------------
# Helper: encode a single JPEG frame as base64
# ---------------------------------------------------------------------------
def _encode_frame_b64(frame_path: str) -> str:
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Public: full-video base64 (Gemini path)
# ---------------------------------------------------------------------------
def encode_video_base64(video_path: str) -> str:
    """Read a local .mp4 and return its base64 representation."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Public: 1-fps frame extraction with timestamps (non-Gemini path)
# ---------------------------------------------------------------------------
def extract_frames_with_timestamps(
    video_path: str,
    output_dir: Path,
    fps: float = 1.0,
) -> list[tuple[str, float]]:
    """Extract frames at *fps* rate from *video_path*, save to *output_dir*.

    Returns a list of (absolute_frame_path, timestamp_seconds) tuples in
    chronological order, capped at MAX_FRAMES.

    Raises:
        FileNotFoundError: if video_path doesn't exist.
        ValueError: if the video can't be opened.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / video_fps   # seconds

    # Build (frame_index, timestamp_seconds) pairs at requested fps intervals
    step_secs = 1.0 / fps
    raw_pairs: list[tuple[int, float]] = []
    t = 0.0
    while t < duration and len(raw_pairs) < MAX_FRAMES:
        idx = int(t * video_fps)
        if idx < total_frames:
            raw_pairs.append((idx, round(t, 2)))
        t += step_secs

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, float]] = []

    for i, (frame_idx, timestamp) in enumerate(raw_pairs, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, bgr = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        if img.width > FRAME_MAX_WIDTH:
            scale    = FRAME_MAX_WIDTH / img.width
            new_size = (FRAME_MAX_WIDTH, int(img.height * scale))
            img      = img.resize(new_size, Image.LANCZOS)

        out_path = output_dir / f"frame_{i:04d}.jpg"
        img.save(str(out_path), format="JPEG", quality=FRAME_QUALITY)
        results.append((str(out_path), timestamp))

    cap.release()
    return results
