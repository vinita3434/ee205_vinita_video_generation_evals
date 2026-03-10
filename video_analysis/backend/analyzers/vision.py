"""
vision.py — Gruve Atlas video analysis runner.

Hybrid input strategy:
  • Gemini models   → encode full video as base64 (native video understanding)
  • All other VLMs  → extract frames at 1 fps with [t=Xs] timestamp labels

Both model paths run in parallel via ThreadPoolExecutor.
stream=False throughout (streaming breaks inside threads).
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import litellm

from backend.config import FRAMES_DIR, OPENROUTER_API_KEY
from backend.analyzers.frame_extractor import (
    _encode_frame_b64,
    encode_video_base64,
    extract_frames_with_timestamps,
)

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# Available vision models
# ---------------------------------------------------------------------------

VISION_MODELS = {
    "closed": [
        {"id": "openrouter/openai/gpt-4o",               "name": "GPT-4o",           "type": "closed"},
        {"id": "openrouter/google/gemini-2.0-flash-001", "name": "Gemini 2.0 Flash", "type": "closed"},
    ],
    "open": [
        {"id": "openrouter/meta-llama/llama-3.2-11b-vision-instruct", "name": "Llama 3.2 Vision 11B", "type": "open"},
        {"id": "openrouter/google/gemma-3-12b-it",                    "name": "Gemma 3 12B",           "type": "open"},
        {"id": "openrouter/qwen/qwen-2.5-vl-7b-instruct",             "name": "Qwen 2.5 VL 7B",       "type": "open"},
    ],
}

_MODELS_BY_ID: dict[str, dict] = {
    m["id"]: m
    for group in VISION_MODELS.values()
    for m in group
}


# ---------------------------------------------------------------------------
# Internal: build per-model message content
# ---------------------------------------------------------------------------

def _is_gemini(model_id: str) -> bool:
    return "gemini" in model_id.lower()


def _build_content_gemini(video_path: str, prompt: str) -> list[dict]:
    """Full video encoded as base64 — native Gemini video understanding."""
    video_b64 = encode_video_base64(video_path)
    return [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
        {"type": "text", "text": prompt},
    ]


def _build_content_frames(
    frame_data: list[tuple[str, float]],
    prompt: str,
) -> list[dict]:
    """Timestamped frame array — for GPT-4o, Qwen, Llama, Gemma."""
    content: list[dict] = []
    for frame_path, timestamp in frame_data:
        content.append({"type": "text", "text": f"[t={int(timestamp)}s]"})
        content.append({
            "type": "image_url",
            "image_url": {
                "url":    f"data:image/jpeg;base64,{_encode_frame_b64(frame_path)}",
                "detail": "low",
            },
        })
    content.append({"type": "text", "text": prompt})
    return content


# ---------------------------------------------------------------------------
# Cost fallback — per-token rates (USD) when API doesn't return cost
# ---------------------------------------------------------------------------

# (input_usd_per_token, output_usd_per_token)
_TOKEN_RATES: dict[str, tuple[float, float]] = {
    "openrouter/openai/gpt-4o":                              (5e-6,  15e-6),
    "openrouter/google/gemini-2.0-flash-001":                (1e-7,   4e-7),
    "openrouter/qwen/qwen-2.5-vl-7b-instruct":              (2e-7,   6e-7),
    "openrouter/meta-llama/llama-3.2-11b-vision-instruct":  (2e-7,   6e-7),
    "openrouter/google/gemma-3-12b-it":                     (2e-7,   6e-7),
}


def _token_cost(model_id: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    rates = _TOKEN_RATES.get(model_id)
    if not rates:
        return None
    return input_tokens * rates[0] + output_tokens * rates[1]


# ---------------------------------------------------------------------------
# Internal: run one model (called from a thread)
# ---------------------------------------------------------------------------

def _run_one(
    run_id:          str,
    model_cfg:       dict,
    prompt:          str,
    max_tokens:      int,
    video_path:      str,
    frame_data:      list[tuple[str, float]],  # pre-extracted; may be [] for Gemini
    input_modality:  str | None = None,
) -> dict:
    use_native = (input_modality == "native") if input_modality else _is_gemini(model_cfg["id"])

    base = {
        "run_id":        run_id,
        "model_id":      model_cfg["id"],
        "model_name":    model_cfg["name"],
        "model_type":    model_cfg["type"],
        "output_text":   None,
        "latency_ms":    None,
        "input_tokens":  None,
        "output_tokens": None,
        "cost_usd":      None,
        "error":         None,
        "input_mode":    "video" if use_native else "frames",
        "n_frames":      None if use_native else len(frame_data),
    }

    try:
        if use_native:
            content = _build_content_gemini(video_path, prompt)
        else:
            content = _build_content_frames(frame_data, prompt)

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
        # Fallback: compute from token counts if still null
        if cost_usd is None and input_tokens is not None and output_tokens is not None:
            cost_usd = _token_cost(model_cfg["id"], input_tokens, output_tokens)

        return {
            **base,
            "output_text":   output_text,
            "latency_ms":    round(latency_ms, 1),
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      cost_usd,
        }

    except Exception as exc:
        return {**base, "error": str(exc)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_video(
    video_path:       str,
    prompt:           str,
    closed_model_id:  str,
    open_model_id:    str,
    max_tokens:       int = 200,
    closed_modality:  str | None = None,
    open_modality:    str | None = None,
) -> dict:
    """Run two vision models against a local video file in parallel.

    Gemini models receive the full video as base64 by default.
    All other models receive 1-fps frames with [t=Xs] timestamp labels by default.
    Pass closed_modality / open_modality ('native' or 'frames') to override.

    Args:
        video_path:      Absolute path to a local .mp4 file.
        prompt:          User's analysis question.
        closed_model_id: Model ID from VISION_MODELS['closed'].
        open_model_id:   Model ID from VISION_MODELS['open'].
        max_tokens:      Max output tokens per model.
        closed_modality: Override input modality for the closed model ('native'|'frames'|None).
        open_modality:   Override input modality for the open model ('native'|'frames'|None).

    Returns:
        dict with run metadata and results list [closed_result, open_result].

    Raises:
        ValueError: Unknown model ID.
        FileNotFoundError: video_path doesn't exist.
    """
    closed_cfg = _MODELS_BY_ID.get(closed_model_id)
    open_cfg   = _MODELS_BY_ID.get(open_model_id)
    if not closed_cfg:
        raise ValueError(f"Unknown closed-source model: {closed_model_id!r}")
    if not open_cfg:
        raise ValueError(f"Unknown open-source model: {open_model_id!r}")

    run_id = str(uuid.uuid4())[:8]

    # Determine whether frames need to be extracted.
    # Extract frames if EITHER model will use frame-based input.
    if closed_modality is None and open_modality is None:
        # Default behaviour: extract frames unless BOTH models are Gemini
        needs_frames = not _is_gemini(closed_model_id) or not _is_gemini(open_model_id)
    else:
        needs_frames = closed_modality != "native" or open_modality != "native"

    frame_data: list[tuple[str, float]] = []
    if needs_frames:
        frame_dir  = FRAMES_DIR / run_id
        frame_data = extract_frames_with_timestamps(video_path, frame_dir, fps=1.0)

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {
            ex.submit(
                _run_one, run_id, closed_cfg, prompt, max_tokens,
                video_path, frame_data, closed_modality,
            ): "closed",
            ex.submit(
                _run_one, run_id, open_cfg, prompt, max_tokens,
                video_path, frame_data, open_modality,
            ): "open",
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    return {
        "run_id":     run_id,
        "video_path": video_path,
        "prompt":     prompt,
        "n_frames":   len(frame_data),
        "results":    [results["closed"], results["open"]],
    }
