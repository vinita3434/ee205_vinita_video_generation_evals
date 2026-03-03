"""
extract_frames.py — Extract 5 evenly-spaced frames from a video clip using OpenCV.

Output: data/frames/{clip_id}/frame_1.jpg ... frame_5.jpg
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
from PIL import Image

# Allow imports from parent dir
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FRAME_COUNT, FRAME_FORMAT, FRAME_MAX_WIDTH, FRAME_QUALITY, FRAMES_DIR


def extract_frames(
    video_path: str,
    clip_id: str,
    source_url: str | None = None,
    output_dir: str | None = None,
) -> list[str]:
    """Extract FRAME_COUNT evenly-spaced frames from video_path.

    If the local file is missing and source_url is provided, the video is
    downloaded automatically before extraction.

    Frames are resized to max FRAME_MAX_WIDTH px wide (aspect preserved)
    and saved as JPEG at FRAME_QUALITY.

    Returns list of FRAME_COUNT absolute paths in chronological order.
    Raises FileNotFoundError if video_path doesn't exist (and can't be fetched).
    Raises ValueError if the video has fewer frames than FRAME_COUNT.
    """
    video_path = str(video_path)
    out_dir = Path(output_dir) if output_dir else FRAMES_DIR / clip_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(video_path).exists() and source_url:
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"  [download] {source_url} → {video_path}")
        urllib.request.urlretrieve(source_url, video_path)

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < FRAME_COUNT:
        cap.release()
        raise ValueError(
            f"Video has only {total_frames} frames — need at least {FRAME_COUNT}."
        )

    # Evenly-spaced indices: start, 25%, 50%, 75%, end
    if FRAME_COUNT == 1:
        indices = [total_frames // 2]
    else:
        indices = [int(i * (total_frames - 1) / (FRAME_COUNT - 1)) for i in range(FRAME_COUNT)]

    frame_paths: list[str] = []
    for frame_num, idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, bgr_frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Failed to read frame at index {idx} for clip '{clip_id}'.")

        # Convert BGR → RGB for Pillow
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)

        # Resize if wider than FRAME_MAX_WIDTH
        if img.width > FRAME_MAX_WIDTH:
            scale = FRAME_MAX_WIDTH / img.width
            new_size = (FRAME_MAX_WIDTH, int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        out_path = out_dir / f"frame_{frame_num}.jpg"
        img.save(str(out_path), format=FRAME_FORMAT, quality=FRAME_QUALITY)
        frame_paths.append(str(out_path))

    cap.release()
    return frame_paths


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python extract_frames.py <video_path> <clip_id>")
        sys.exit(1)
    paths = extract_frames(sys.argv[1], sys.argv[2])
    print(f"Extracted {len(paths)} frames:")
    for p in paths:
        print(f"  {p}")
