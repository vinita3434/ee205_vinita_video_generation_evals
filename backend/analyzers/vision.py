"""
vision.py — Video analysis runner for the web API.

Wraps video_eval/src logic but accepts a custom prompt per request
and runs two models (closed + open source) in parallel.
"""
from __future__ import annotations

import base64
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import litellm

# Add video_eval/ to path so its internal imports resolve correctly
VIDEO_EVAL_DIR = Path(__file__).resolve().parents[2] / "video_eval"
sys.path.insert(0, str(VIDEO_EVAL_DIR))
from src.extract_frames import extract_frames as _extract_frames  # noqa: E402

from backend.config import OPENROUTER_API_KEY  # noqa: E402

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Available vision models
# ---------------------------------------------------------------------------

VISION_MODELS = {
    "closed": [
        {"id": "openrouter/openai/gpt-4o",                  "name": "GPT-4o",             "type": "closed"},
        {"id": "openrouter/google/gemini-2.0-flash-001",    "name": "Gemini 2.0 Flash",   "type": "closed"},
    ],
    "open": [
        {"id": "openrouter/qwen/qwen-2.5-vl-7b-instruct",             "name": "Qwen 2.5 VL 7B",         "type": "open"},
        {"id": "openrouter/meta-llama/llama-3.2-11b-vision-instruct", "name": "Llama 3.2 Vision 11B",   "type": "open"},
        {"id": "openrouter/google/gemma-3-12b-it",                    "name": "Gemma 3 12B",             "type": "open"},
    ],
}

_MODELS_BY_ID: dict[str, dict] = {
    m["id"]: m
    for models in VISION_MODELS.values()
    for m in models
}


# ---------------------------------------------------------------------------
# Internal: run one model against extracted frames
# ---------------------------------------------------------------------------

def _encode_frame(frame_path: str) -> str:
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _run_one(
    run_id: str,
    frame_paths: list[str],
    model_cfg: dict,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Synchronous — call from a thread."""
    base = {
        "run_id":       run_id,
        "model_id":     model_cfg["id"],
        "model_name":   model_cfg["name"],
        "model_type":   model_cfg["type"],
        "output_text":  None,
        "latency_ms":   None,
        "ttft_ms":      None,
        "input_tokens":  None,
        "output_tokens": None,
        "cost_usd":     None,
        "error":        None,
    }
    try:
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_encode_frame(p)}",
                    "detail": "low",
                },
            }
            for p in frame_paths
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        t0 = time.perf_counter()
        full = litellm.completion(
            model=model_cfg["id"],
            messages=messages,
            stream=False,
            max_tokens=max_tokens,
            temperature=0.3,
            api_base="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        ttft_ms = None  # not available without streaming

        output_text = None
        input_tokens = output_tokens = None
        if full and full.choices:
            output_text = full.choices[0].message.content
        if full and full.usage:
            input_tokens  = full.usage.prompt_tokens
            output_tokens = full.usage.completion_tokens

        cost_usd: Optional[float] = None
        try:
            cost_usd = float(full._hidden_params.get("response_cost") or 0) or None
        except Exception:
            pass
        if cost_usd is None:
            try:
                cost_usd = litellm.completion_cost(completion_response=full)
            except Exception:
                pass

        return {
            **base,
            "output_text":  output_text,
            "latency_ms":   round(latency_ms, 1),
            "ttft_ms":      round(ttft_ms, 1) if ttft_ms is not None else None,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":     cost_usd,
        }

    except Exception as exc:
        return {**base, "error": str(exc)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_video(
    video_url: str,
    prompt: str,
    closed_model_id: str,
    open_model_id: str,
    n_frames: int = 3,
    max_tokens: int = 200,
) -> dict:
    """Download video from URL, extract frames, run two models in parallel.

    Returns a dict with run metadata and results list [closed, open].
    Raises ValueError for unknown model IDs.
    Raises on download/extraction failure.
    """
    closed_cfg = _MODELS_BY_ID.get(closed_model_id)
    open_cfg   = _MODELS_BY_ID.get(open_model_id)
    if not closed_cfg:
        raise ValueError(f"Unknown closed-source model: {closed_model_id}")
    if not open_cfg:
        raise ValueError(f"Unknown open-source model: {open_model_id}")

    run_id = str(uuid.uuid4())[:8]

    # Extract frames into video_eval/data/frames/<run_id>/
    # extract_frames auto-downloads from source_url if local file missing
    clips_dir = VIDEO_EVAL_DIR / "data" / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    video_path = str(clips_dir / f"{run_id}.mp4")

    all_frame_paths = _extract_frames(
        video_path=video_path,
        clip_id=run_id,
        source_url=video_url,
    )
    frame_paths = all_frame_paths[:n_frames]

    # Run both models in parallel threads
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(_run_one, run_id, frame_paths, closed_cfg, prompt, max_tokens): "closed",
            ex.submit(_run_one, run_id, frame_paths, open_cfg,   prompt, max_tokens): "open",
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    return {
        "run_id":    run_id,
        "video_url": video_url,
        "prompt":    prompt,
        "n_frames":  len(frame_paths),
        "results":   [results["closed"], results["open"]],
    }
