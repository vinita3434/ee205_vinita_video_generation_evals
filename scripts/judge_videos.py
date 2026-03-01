#!/usr/bin/env python3
"""
Standalone judge script: send 2 videos + prompt to OpenRouter (Gemini) for evaluation.
Usage:
  python scripts/judge_videos.py "Your prompt" outputs/runid_kling.mp4 outputs/runid_sora.mp4
"""
import argparse
import base64
import os
import sys
from pathlib import Path

# Add project root for backend imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "google/gemini-2.0-flash-exp:free")


def video_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


def main():
    parser = argparse.ArgumentParser(description="Judge 2 videos (Kling, Sora) with Gemini via OpenRouter")
    parser.add_argument("prompt", help="Original prompt used to generate the videos")
    parser.add_argument("kling_video", help="Path to Kling 2.6 output video")
    parser.add_argument("sora_video", help="Path to OpenAI Sora 2 output video")
    args = parser.parse_args()

    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    judge_prompt = (
        f"Compare these 2 videos to this prompt: {args.prompt}\n\n"
        "Rate each video from 1-10 on:\n"
        "1. Temporal Stability (no flickering)\n"
        "2. Physics Accuracy\n\n"
        "Label each video as: Video A (Kling 2.6), Video B (OpenAI Sora 2). "
        "Respond with a clear table or structured ratings and a short summary."
    )

    try:
        video_a = video_to_data_url(args.kling_video)
        video_b = video_to_data_url(args.sora_video)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    content = [
        {"type": "text", "text": judge_prompt},
        {"type": "video_url", "video_url": {"url": video_a}},
        {"type": "video_url", "video_url": {"url": video_b}},
    ]

    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1024,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    print(message.get("content", ""))


if __name__ == "__main__":
    main()
