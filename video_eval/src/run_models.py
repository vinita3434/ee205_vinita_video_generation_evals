"""
run_models.py — Send 5 frames + prompt to a vision model via LiteLLM/OpenRouter.
Captures output text, TTFT, total latency, token counts, and cost.
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path
from typing import Optional

import litellm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import OPENROUTER_API_KEY, PROMPT

litellm.suppress_debug_info = True

# Route all calls through OpenRouter
litellm.api_base = "https://openrouter.ai/api/v1"
litellm.api_key = OPENROUTER_API_KEY


def _encode_frame(frame_path: str) -> str:
    """Read a JPEG file and return its base64-encoded string."""
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def run_model(clip_id: str, frame_paths: list[str], model_cfg: dict) -> dict:
    """Send 5 frames + PROMPT to model_cfg and capture all metrics.

    Returns a dict with keys:
        clip_id, model_id, model_name, model_type,
        output_text, latency_ms, ttft_ms,
        input_tokens, output_tokens, cost_usd,
        error  (None on success, error string on failure)
    """
    base_result = {
        "clip_id":     clip_id,
        "model_id":    model_cfg["id"],
        "model_name":  model_cfg["name"],
        "model_type":  model_cfg["type"],
        "output_text": None,
        "latency_ms":  None,
        "ttft_ms":     None,
        "input_tokens":  None,
        "output_tokens": None,
        "cost_usd":    None,
        "error":       None,
    }

    try:
        # Encode all frames
        encoded_frames = [_encode_frame(p) for p in frame_paths]

        # Build multimodal content: images first (in order), then text prompt
        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low",  # cheapest OpenAI tier: 85 tokens/image flat
                },
            }
            for b64 in encoded_frames
        ]
        content.append({"type": "text", "text": PROMPT})

        messages = [{"role": "user", "content": content}]

        # Stream the response and capture timing
        collected_chunks: list = []
        ttft_ms: Optional[float] = None
        t_start = time.perf_counter()

        response = litellm.completion(
            model=model_cfg["id"],
            messages=messages,
            stream=True,
            max_tokens=200,
            temperature=0.3,
        )

        for chunk in response:
            if (
                ttft_ms is None
                and chunk.choices
                and chunk.choices[0].delta.content
            ):
                ttft_ms = (time.perf_counter() - t_start) * 1000
            collected_chunks.append(chunk)

        latency_ms = (time.perf_counter() - t_start) * 1000

        # Rebuild full response for token counts + cost
        from litellm.main import stream_chunk_builder
        full_response = stream_chunk_builder(collected_chunks, messages=messages)

        output_text: Optional[str] = None
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None

        if full_response and full_response.choices:
            output_text = full_response.choices[0].message.content

        if full_response and full_response.usage:
            input_tokens  = full_response.usage.prompt_tokens
            output_tokens = full_response.usage.completion_tokens

        # Cost extraction — 3-method waterfall
        cost_usd: Optional[float] = None
        try:
            cost_usd = float(full_response._hidden_params.get("response_cost") or 0) or None
        except Exception:
            pass
        if cost_usd is None:
            try:
                cost_usd = litellm.completion_cost(completion_response=full_response)
            except Exception:
                pass

        return {
            **base_result,
            "output_text":   output_text,
            "latency_ms":    round(latency_ms, 1),
            "ttft_ms":       round(ttft_ms, 1) if ttft_ms is not None else None,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      cost_usd,
        }

    except Exception as exc:
        return {**base_result, "error": str(exc)}
