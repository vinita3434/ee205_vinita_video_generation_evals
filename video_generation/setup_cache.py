"""
Setup script: Download Sora sample videos and pre-score them.
Run ONCE before demo:
    python video_generation/setup_cache.py

Requires: OPENROUTER_API_KEY in .env
"""

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import httpx

# ── Config ────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
CACHE_DIR = ROOT / "video_generation" / "data" / "cached_videos"
CACHE_JSON = ROOT / "video_generation" / "data" / "video_cache.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_KEY:
    print("ERROR: OPENROUTER_API_KEY not set in .env")
    sys.exit(1)

# ── Sora official sample URLs (public CDN) ─────────────────────────
SORA_VIDEOS = {
    "resolution_720p":  "https://cdn.openai.com/sora/videos/ships-in-coffee.mp4",
    "resolution_1080p": "https://cdn.openai.com/sora/videos/victoria-crowned-pigeon.mp4",
    "resolution_4k":    "https://cdn.openai.com/sora/videos/big-sur.mp4",
    "clip_5s":          "https://cdn.openai.com/sora/videos/ships-in-coffee.mp4",
    "clip_10s":         "https://cdn.openai.com/sora/videos/victoria-crowned-pigeon.mp4",
    "clip_15s":         "https://cdn.openai.com/sora/videos/mitten-astronaut.mp4",
    "clip_20s":         "https://cdn.openai.com/sora/videos/wooly-mammoth.mp4",
    "clip_30s":         "https://cdn.openai.com/sora/videos/big-sur.mp4",
}

CACHE_METADATA = {
    "resolution_720p":  {"model": "sora2", "resolution": "720p",  "clip_sec": 10, "cost_per_gen": 1.50, "label": "720p"},
    "resolution_1080p": {"model": "sora2", "resolution": "1080p", "clip_sec": 10, "cost_per_gen": 3.00, "label": "1080p"},
    "resolution_4k":    {"model": "sora2", "resolution": "4K",    "clip_sec": 10, "cost_per_gen": 6.00, "label": "4K"},
    "clip_5s":          {"model": "sora2", "resolution": "1080p", "clip_sec": 5,  "cost_per_gen": 1.50, "label": "5s"},
    "clip_10s":         {"model": "sora2", "resolution": "1080p", "clip_sec": 10, "cost_per_gen": 3.00, "label": "10s"},
    "clip_15s":         {"model": "sora2", "resolution": "1080p", "clip_sec": 15, "cost_per_gen": 4.50, "label": "15s"},
    "clip_20s":         {"model": "sora2", "resolution": "1080p", "clip_sec": 20, "cost_per_gen": 6.00, "label": "20s"},
    "clip_30s":         {"model": "sora2", "resolution": "1080p", "clip_sec": 30, "cost_per_gen": 9.00, "label": "30s"},
}

JUDGE_PROMPT = """You are evaluating AI-generated video quality for a creative agency use case.
You will see frames sampled from the video.

Score on 5 dimensions (0-10 each):

1. SUBJECT_QUALITY: Is the main subject clear, well-rendered, and visually compelling?
   0 = unclear/distorted | 10 = crisp, professional, broadcast-ready

2. TEMPORAL_COHERENCE: Do the frames tell a coherent visual story/sequence?
   0 = random/disconnected | 10 = smooth narrative progression

3. MOTION_QUALITY: Does movement look natural and fluid?
   0 = static/jittery | 10 = cinematic fluid motion

4. VISUAL_FIDELITY: Sharpness, color accuracy, no artifacts or distortion?
   0 = blurry/artifacted | 10 = pristine, high fidelity

5. SCENE_STABILITY: Consistent lighting, composition, no flickering?
   0 = inconsistent | 10 = stable, professional

Return ONLY this JSON, nothing else:
{"subject_quality": 0, "temporal_coherence": 0, "motion_quality": 0, "visual_fidelity": 0, "scene_stability": 0, "overall": 0, "one_line_summary": "one sentence"}

overall = round(subject_quality*0.30 + temporal_coherence*0.25 + motion_quality*0.20 + visual_fidelity*0.15 + scene_stability*0.10, 1)"""


def download_video(key: str, url: str):
    dest = CACHE_DIR / f"{key}.mp4"
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"  Already exists: {key}")
        return str(dest)
    print(f"  Downloading {key}...")
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            r = client.get(url)
            if r.status_code != 200:
                print(f"  WARN: {url} returned {r.status_code}")
                return None
            dest.write_bytes(r.content)
            print(f"  Saved: {dest.name} ({len(r.content)//1024}KB)")
            return str(dest)
    except Exception as e:
        print(f"  ERROR downloading {key}: {e}")
        return None


def extract_frames_b64(video_path: str, n: int = 4) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    indices = [int(i * total / n) for i in range(n)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(base64.b64encode(buf).decode())
    cap.release()
    return frames


def score_video(video_path: str) -> dict:
    frames = extract_frames_b64(video_path, n=4)
    if not frames:
        return {"subject_quality": 5, "temporal_coherence": 5, "motion_quality": 5,
                "visual_fidelity": 5, "scene_stability": 5, "overall": 5,
                "one_line_summary": "Could not extract frames"}

    content = [{"type": "text", "text": JUDGE_PROMPT}]
    for fb64 in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}})

    import re
    try:
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json",
                     "HTTP-Referer": "https://gruve.ai", "X-Title": "Gruve Atlas"},
            json={"model": "google/gemini-2.0-flash-001",
                  "messages": [{"role": "user", "content": content}], "max_tokens": 300},
            timeout=60,
        )
        raw = r.json()["choices"][0]["message"]["content"] or ""
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        print(f"    Score error: {e}")

    return {"subject_quality": 5, "temporal_coherence": 5, "motion_quality": 5,
            "visual_fidelity": 5, "scene_stability": 5, "overall": 5,
            "one_line_summary": "Scoring unavailable — fallback values"}


def main():
    print("=== Gruve Atlas: Video Generation Cache Setup ===\n")

    # Download
    print("Step 1: Downloading Sora sample videos...")
    video_paths = {}
    for key, url in SORA_VIDEOS.items():
        path = download_video(key, url)
        if path:
            video_paths[key] = path

    # Score
    print(f"\nStep 2: Scoring {len(video_paths)} videos via LLM judge...")
    cache_data = {}
    for key, meta in CACHE_METADATA.items():
        if key not in video_paths:
            print(f"  SKIP {key} (download failed)")
            continue
        print(f"  Scoring {key}...")
        scores = score_video(video_paths[key])
        cache_data[key] = {
            **meta,
            "video_path": video_paths[key],
            "scores": scores,
            "source": "sora_official_sample",
            "scored": True,
        }
        print(f"    Overall: {scores.get('overall', '?')}/10 — {scores.get('one_line_summary', '')}")
        time.sleep(0.5)  # avoid rate limiting

    CACHE_JSON.write_text(json.dumps(cache_data, indent=2))
    print(f"\n✓ Cache complete. {len(cache_data)} videos scored.")
    print(f"  Saved: {CACHE_JSON}")


if __name__ == "__main__":
    main()
