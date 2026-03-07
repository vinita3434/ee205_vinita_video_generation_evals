import time
from pathlib import Path

from litellm import video_generation, video_status, video_content

from backend.config import OPENAI_API_KEY, OUTPUT_DIR


def generate_sora(prompt: str, run_id: str, quality: str = "720p", aspect_ratio: str = "16:9") -> Path:
    """Generate video with OpenAI Sora 2 via LiteLLM. Returns path to saved MP4."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    out_path = OUTPUT_DIR / f"{run_id}_sora.mp4"

    # Sora only supports 4 sizes — map quality + ratio to nearest valid option
    high_res = quality == "1080p"
    if aspect_ratio == "9:16":
        size = "1024x1792" if high_res else "720x1280"
    else:  # 16:9 or 1:1 (no square support, default to landscape)
        size = "1792x1024" if high_res else "1280x720"

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
