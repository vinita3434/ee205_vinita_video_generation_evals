from pathlib import Path

from fal_client import SyncClient

from backend.config import FAL_KEY, OUTPUT_DIR

KLING_ENDPOINT = "fal-ai/kling-video/v2.6/pro/text-to-video"


def generate_kling(prompt: str, run_id: str, aspect_ratio: str = "16:9") -> Path:
    """Generate video with Kling 2.6 via fal.ai. Returns path to saved MP4."""
    key = (FAL_KEY or "").strip()
    if not key:
        raise ValueError("FAL_KEY is not set. Add FAL_KEY to your .env (get one at https://fal.ai/dashboard/keys)")
    out_path = OUTPUT_DIR / f"{run_id}_kling.mp4"

    # Pass key explicitly so auth works even if env wasn't set when process started
    client = SyncClient(key=key)
    result = client.subscribe(
        KLING_ENDPOINT,
        arguments={
            "prompt": prompt,
            "duration": "5",
            "aspect_ratio": aspect_ratio,
            "generate_audio": False,
        },
        with_logs=True,
    )

    video_info = result.get("video")
    if not video_info:
        raise RuntimeError(f"Kling returned no video: {result}")
    url = video_info.get("url") if isinstance(video_info, dict) else getattr(video_info, "url", None)
    if not url:
        raise RuntimeError(f"No video URL in Kling result: {result}")

    import httpx
    r = httpx.get(url)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path
