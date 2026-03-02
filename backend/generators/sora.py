import time
from pathlib import Path

from litellm import video_generation, video_status, video_content

from backend.config import OPENAI_API_KEY, OUTPUT_DIR

# Maps (quality, aspect_ratio) → Sora size string
_SORA_SIZES: dict[tuple[str, str], str] = {
    ("480p",  "16:9"): "854x480",
    ("480p",  "9:16"): "480x854",
    ("480p",  "1:1"):  "480x480",
    ("720p",  "16:9"): "1280x720",
    ("720p",  "9:16"): "720x1280",
    ("720p",  "1:1"):  "720x720",
    ("1080p", "16:9"): "1920x1080",
    ("1080p", "9:16"): "1080x1920",
    ("1080p", "1:1"):  "1080x1080",
}


def generate_sora(prompt: str, run_id: str, quality: str = "720p", aspect_ratio: str = "16:9") -> Path:
    """Generate video with OpenAI Sora 2 via LiteLLM. Returns path to saved MP4."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    out_path = OUTPUT_DIR / f"{run_id}_sora.mp4"

    size = _SORA_SIZES.get((quality, aspect_ratio), "1280x720")
    response = video_generation(
        model="openai/sora-2",
        prompt=prompt,
        seconds="8",
        size=size,
        api_key=OPENAI_API_KEY,
    )
    video_id = response.id

    while True:
        status_resp = video_status(video_id=video_id, api_key=OPENAI_API_KEY)
        if status_resp.status == "completed":
            break
        if status_resp.status == "failed":
            raise RuntimeError(f"Sora generation failed: {getattr(status_resp, 'error', 'unknown')}")
        time.sleep(10)

    video_bytes = video_content(video_id=video_id, api_key=OPENAI_API_KEY)
    out_path.write_bytes(video_bytes)
    return out_path
