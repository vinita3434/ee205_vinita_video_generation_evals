import time
from pathlib import Path

import httpx

from backend.config import NOVITA_API_KEY, OUTPUT_DIR

NOVITA_BASE = "https://api.novita.ai"

# Use Wan 2.1 (wan-t2v); Wan 2.2 (wan-2.2-t2v) often returns 403 without the right plan
NOVITA_WAN_ENDPOINT = f"{NOVITA_BASE}/v3/async/wan-t2v"


def generate_wan(prompt: str, run_id: str) -> Path:
    """Generate video with Wan 2.1 via Novita (wan-t2v). Returns path to saved MP4."""
    if not NOVITA_API_KEY:
        raise ValueError("NOVITA_API_KEY is not set")
    out_path = OUTPUT_DIR / f"{run_id}_wan.mp4"

    with httpx.Client(timeout=60.0) as client:
        # Start Wan 2.1 T2V task (documented request body)
        r = client.post(
            NOVITA_WAN_ENDPOINT,
            headers={
                "Authorization": f"Bearer {NOVITA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "prompt": prompt[:2000],
                "negative_prompt": "blur, distort, low quality, worst quality",
                "width": 1280,
                "height": 720,
                "seed": -1,
            },
        )
        if r.status_code == 403:
            try:
                err_body = r.json()
            except Exception:
                err_body = {"detail": r.text or "403 Forbidden"}
            msg = err_body.get("message") or err_body.get("detail") or err_body.get("error") or str(err_body)
            raise RuntimeError(
                f"Novita 403 Forbidden. Key valid but no permission for this API. "
                f"Details: {msg}. "
                "Check: novita.ai console → identity verification; confirm your plan includes Wan/Video."
            )
        r.raise_for_status()
        data = r.json()
        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"Novita did not return task_id: {data}")

        # Poll task result
        while True:
            r2 = client.get(
                f"{NOVITA_BASE}/v3/async/task-result",
                params={"task_id": task_id},
                headers={"Authorization": f"Bearer {NOVITA_API_KEY}"},
            )
            r2.raise_for_status()
            result = r2.json()
            task = result.get("task", {})
            status = task.get("status", "")
            if status == "TASK_STATUS_SUCCEED":
                break
            if status == "TASK_STATUS_FAILED":
                raise RuntimeError(f"Wan generation failed: {task.get('reason', result)}")
            time.sleep(10)

        videos = result.get("videos") or []
        if not videos:
            raise RuntimeError("Wan task succeeded but no video in response")
        video_url = videos[0].get("video_url")
        if not video_url:
            raise RuntimeError("No video_url in Wan response")

        # Download video
        r3 = client.get(video_url)
        r3.raise_for_status()
        out_path.write_bytes(r3.content)

    return out_path
