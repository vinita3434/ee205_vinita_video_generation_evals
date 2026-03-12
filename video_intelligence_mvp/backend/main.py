import asyncio
import base64
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).parents[2] / ".env")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
CLIPS_DIR = Path(__file__).parents[2] / "video_analysis" / "data" / "test_clips"

MODEL_IDS = {
    "gpt4o":  "openai/gpt-4o",
    "gemini": "google/gemini-2.0-flash-001",
    "qwen":   "qwen/qwen2.5-vl-72b-instruct",
}
PRICING = {
    "gpt4o":  {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gemini": {"input": 0.10 / 1e6, "output":  0.40 / 1e6},
    "qwen":   {"input": 0.59 / 1e6, "output":  0.59 / 1e6},
}

RESULTS_FILE = Path(__file__).parents[1] / "data" / "results" / "benchmark_results.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Video processing ──────────────────────────────────────────────

def extract_frames_as_b64(clip_path: str, fps: int = 1) -> list[str]:
    cap = cv2.VideoCapture(clip_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, int(video_fps / fps))
    frames, frame_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(base64.b64encode(buf).decode("utf-8"))
        frame_idx += 1
    cap.release()
    return frames[:20]  # cap at 20 frames


def encode_video_as_b64(clip_path: str) -> tuple[str, str]:
    with open(clip_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    mime = "video/quicktime" if clip_path.endswith(".mov") else "video/mp4"
    return data, mime


def build_messages(model: str, prompt: str, clip_path: str) -> list[dict]:
    if model == "gemini":
        video_b64, mime = encode_video_as_b64(clip_path)
        return [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{video_b64}"}},
        ]}]
    else:
        frames = extract_frames_as_b64(clip_path, fps=1)
        content = [{"type": "text", "text": prompt}]
        for fb64 in frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}})
        return [{"role": "user", "content": content}]


# ── Request models ────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    id: str
    name: str
    model: str
    clip_length_sec: float
    daily_volume: int
    concurrency: int
    assembled_prompt: str
    use_case_params: dict = {}
    clip_name: Optional[str] = None       # local filename, e.g. "nba_highlight.mp4"
    clip_data_b64: Optional[str] = None   # base64-encoded uploaded file
    clip_mime_type: Optional[str] = None  # mime type for uploaded file


class BenchmarkRequest(BaseModel):
    consumer_type: str
    use_case: str
    scenarios: list[ScenarioRequest]


# ── Benchmark runner ──────────────────────────────────────────────

async def run_scenario_benchmark(scenario: ScenarioRequest) -> dict:
    base = {"id": scenario.id, "name": scenario.name, "model": scenario.model}

    if scenario.model not in MODEL_IDS:
        return {**base, "status": "error", "error": f"Unknown model: {scenario.model}"}
    if not OPENROUTER_API_KEY:
        return {**base, "status": "error", "error": "OPENROUTER_API_KEY not configured"}

    # Resolve clip to a file path
    tmp_file = None
    clip_label = "unknown"
    try:
        if scenario.clip_name:
            clip_path = str(CLIPS_DIR / scenario.clip_name)
            clip_label = scenario.clip_name
        elif scenario.clip_data_b64:
            ext = ".mov" if scenario.clip_mime_type == "video/quicktime" else ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(base64.b64decode(scenario.clip_data_b64))
            tmp.close()
            clip_path = tmp.name
            tmp_file = tmp.name
            clip_label = f"uploaded{ext}"
        else:
            return {**base, "status": "error", "error": "No clip provided"}

        messages = build_messages(scenario.model, scenario.assembled_prompt, clip_path)
    except Exception as e:
        if tmp_file:
            try: os.unlink(tmp_file)
            except Exception: pass
        return {**base, "status": "error", "error": f"Clip processing failed: {e}"}

    model_id = MODEL_IDS[scenario.model]
    p = PRICING[scenario.model]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://gruve.ai",
        "X-Title": "Gruve Atlas",
    }
    payload = {"model": model_id, "messages": messages, "max_tokens": 500, "stream": True}

    t_start = time.time()
    t_first_token = None
    full_text = ""
    output_token_count = 0

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST", f"{OPENROUTER_BASE}/chat/completions",
                headers=headers, json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    return {**base, "status": "error",
                            "error": f"API {resp.status_code}: {body[:200].decode(errors='replace')}"}
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            if t_first_token is None:
                                t_first_token = time.time()
                            full_text += delta
                            output_token_count += 1
                    except Exception:
                        continue
    except Exception as e:
        return {**base, "status": "error", "error": str(e)}
    finally:
        if tmp_file:
            try: os.unlink(tmp_file)
            except Exception: pass

    t_end = time.time()
    ttft_ms = round((t_first_token - t_start) * 1000) if t_first_token else None
    total_sec = max(t_end - t_start, 0.001)
    tps = round(output_token_count / total_sec, 1)

    clip_sec = scenario.clip_length_sec
    if scenario.model == "gemini":
        video_tokens = int(clip_sec * 258)
    else:
        n_frames = min(int(clip_sec), 20)
        video_tokens = n_frames * 255

    prompt_tokens = int(len(scenario.assembled_prompt.split()) * 1.3)
    input_tokens = video_tokens + prompt_tokens
    output_tokens = output_token_count

    video_cost  = video_tokens  * p["input"]
    prompt_cost = prompt_tokens * p["input"]
    output_cost = output_tokens * p["output"]
    cost_per_query = video_cost + prompt_cost + output_cost

    daily = scenario.daily_volume
    qacs = round(min(100, tps / (cost_per_query * 10)), 1) if cost_per_query > 0 else 0
    total_toks = max(input_tokens + output_tokens, 1)

    return {
        **base,
        "ttft_ms": ttft_ms,
        "tps": tps,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_per_query": round(cost_per_query, 6),
        "daily_cost":     round(cost_per_query * daily, 2),
        "monthly_cost":   round(cost_per_query * daily * 30, 2),
        "annual_cost":    round(cost_per_query * daily * 365, 2),
        "qacs": qacs,
        "cost_breakdown": {
            "video_tokens":  video_tokens,
            "video_cost":    round(video_cost, 6),
            "video_pct":     round(video_tokens / max(input_tokens, 1) * 100, 1),
            "prompt_tokens": prompt_tokens,
            "prompt_cost":   round(prompt_cost, 6),
            "prompt_pct":    round(prompt_tokens / max(input_tokens, 1) * 100, 1),
            "output_tokens": output_tokens,
            "output_cost":   round(output_cost, 6),
            "output_pct":    round(output_tokens / total_toks * 100, 1),
        },
        "response_text": full_text[:500],
        "clip_used": clip_label,
        "status": "complete",
    }


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "port": 8002}


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    results = await asyncio.gather(
        *[run_scenario_benchmark(s) for s in req.scenarios],
        return_exceptions=True,
    )
    cleaned = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            cleaned.append({"id": req.scenarios[i].id, "name": req.scenarios[i].name,
                            "model": req.scenarios[i].model, "status": "error", "error": str(r)})
        else:
            cleaned.append(r)

    run = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "consumer_type": req.consumer_type,
        "use_case": req.use_case,
        "scenarios": cleaned,
    }
    existing = []
    if RESULTS_FILE.exists():
        try:
            existing = json.loads(RESULTS_FILE.read_text())
        except Exception:
            pass
    existing.append(run)
    RESULTS_FILE.write_text(json.dumps(existing, indent=2))
    return run


@app.get("/results")
async def get_results():
    if not RESULTS_FILE.exists():
        return []
    try:
        return json.loads(RESULTS_FILE.read_text())
    except Exception:
        return []
