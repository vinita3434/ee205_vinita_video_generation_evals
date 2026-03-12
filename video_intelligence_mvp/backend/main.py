import asyncio
import base64
import json
import os
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).parents[2] / ".env")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE    = "https://openrouter.ai/api/v1"
CLIPS_DIR          = Path(__file__).parents[2] / "video_analysis" / "data" / "test_clips"

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

# Clip name → filename mapping (handles alias differences)
CLIP_FILE_MAP = {
    "nba_highlight": "nba_highlight.mp4",
    "nfl_play":      "nfl_highlight.mp4",   # actual file is nfl_highlight.mp4
    "nfl_highlight": "nfl_highlight.mp4",
    "action_sideways":   "action_attributes__am_i_standing_sideways.mp4",
    "action_jump":       "action_detection__did_i_jump_a_moment.mp4",
    "action_window":     "action_detection__do_i_open_the_window.mp4",
    "object_holding":    "object_referencing__what_am_i_holding_in.mp4",
    "object_have":       "object_referencing__what_do_i_have_in.mp4",
}

# Temp upload store: clip_id → Path
_temp_clips: dict[str, Path] = {}

RESULTS_FILE = Path(__file__).parents[1] / "data" / "results" / "benchmark_results.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Video helpers ─────────────────────────────────────────────────

def extract_frames_as_b64(clip_path: str, fps: int = 1,
                           max_duration_sec: int = None) -> list[str]:
    cap = cv2.VideoCapture(clip_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval  = max(1, int(video_fps / fps))

    hard_cap = 20
    if max_duration_sec:
        duration_cap = int(max_duration_sec * fps)
        max_frames   = min(hard_cap, duration_cap)
    else:
        max_frames = hard_cap

    frames, frame_idx = [], 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames.append(base64.b64encode(buf).decode("utf-8"))
        frame_idx += 1
    cap.release()
    return frames


def get_video_for_gemini(clip_path: str, duration_sec: int) -> tuple[str, str]:
    mime = "video/quicktime" if clip_path.endswith(".mov") else "video/mp4"
    source = clip_path

    if shutil.which("ffmpeg"):
        out_path = f"/tmp/gruve_trim_{duration_sec}s_{Path(clip_path).name}"
        ret = os.system(
            f"ffmpeg -i '{clip_path}' -t {duration_sec} -c copy "
            f"'{out_path}' -y -loglevel quiet 2>/dev/null"
        )
        if ret == 0 and Path(out_path).exists():
            source = out_path

    with open(source, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, mime


def resolve_clip_path(clip_path_or_name: str) -> str:
    """
    Resolve clip_path from request to an absolute filesystem path.
    Accepts:
      - TEMP:{clip_id}    → look up in _temp_clips
      - a known clip name → look up in CLIP_FILE_MAP → CLIPS_DIR
      - otherwise assume it's already a filesystem path
    """
    if clip_path_or_name.startswith("TEMP:"):
        clip_id = clip_path_or_name[5:]
        p = _temp_clips.get(clip_id)
        if p is None:
            raise FileNotFoundError(f"Temp clip {clip_id} not found (may have expired)")
        return str(p)

    if clip_path_or_name in CLIP_FILE_MAP:
        return str(CLIPS_DIR / CLIP_FILE_MAP[clip_path_or_name])

    # Try as bare filename
    candidate = CLIPS_DIR / clip_path_or_name
    if candidate.exists():
        return str(candidate)

    raise FileNotFoundError(f"Clip not found: {clip_path_or_name}")


def build_messages(model: str, prompt: str, clip_path: str,
                   duration_sec: int) -> tuple[list[dict], int]:
    """Returns (messages, video_token_count)."""
    if model == "gemini":
        video_b64, mime = get_video_for_gemini(clip_path, duration_sec)
        video_tokens    = int(duration_sec * 258)
        messages        = [{"role": "user", "content": [
            {"type": "text",      "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{video_b64}"}},
        ]}]
    else:
        frames          = extract_frames_as_b64(clip_path, fps=1,
                                                max_duration_sec=duration_sec)
        video_tokens    = len(frames) * 255
        content         = [{"type": "text", "text": prompt}]
        for fb64 in frames:
            content.append({"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}})
        messages = [{"role": "user", "content": content}]

    return messages, video_tokens


# ── Request models ────────────────────────────────────────────────

class ScenarioRequest(BaseModel):
    id:                   str
    name:                 str
    model:                str            # gpt4o | gemini | qwen
    clip_path:            str            # clip name OR "TEMP:{clip_id}"
    analyze_duration_sec: int            # seconds to analyze
    assembled_prompt:     str
    daily_volume:         int
    use_case_params:      dict = {}
    concurrency:          int = 1


class BenchmarkRequest(BaseModel):
    consumer_type: str
    use_case:      str
    scenarios:     list[ScenarioRequest]


# ── Benchmark runner ──────────────────────────────────────────────

async def run_one_scenario(scenario: ScenarioRequest) -> dict:
    base = {"id": scenario.id, "name": scenario.name, "model": scenario.model}

    if scenario.model not in MODEL_IDS:
        return {**base, "status": "error", "error": f"Unknown model: {scenario.model}"}
    if not OPENROUTER_API_KEY:
        return {**base, "status": "error", "error": "OPENROUTER_API_KEY not configured"}

    try:
        clip_path = resolve_clip_path(scenario.clip_path)
    except FileNotFoundError as e:
        return {**base, "status": "error", "error": f"Clip processing failed: {e}"}

    try:
        messages, video_tokens = build_messages(
            scenario.model, scenario.assembled_prompt,
            clip_path, scenario.analyze_duration_sec
        )
    except Exception as e:
        return {**base, "status": "error", "error": f"Clip processing failed: {e}"}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://gruve.ai",
        "X-Title":       "Gruve Atlas",
    }
    payload = {
        "model":      MODEL_IDS[scenario.model],
        "messages":   messages,
        "max_tokens": 500,
        "stream":     True,
    }

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
                        data  = json.loads(chunk)
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            if t_first_token is None:
                                t_first_token = time.time()
                            full_text          += delta
                            output_token_count += 1
                    except Exception:
                        continue
    except Exception as e:
        return {**base, "status": "error", "error": str(e)}

    t_end      = time.time()
    ttft_ms    = round((t_first_token - t_start) * 1000) if t_first_token else None
    elapsed    = max(t_end - t_start, 0.001)
    tps        = round(output_token_count / elapsed, 1)

    prompt_tokens = int(len(scenario.assembled_prompt.split()) * 1.3)
    input_tokens  = int(video_tokens + prompt_tokens)
    output_tokens = output_token_count
    total_tokens  = max(input_tokens + output_tokens, 1)

    p            = PRICING[scenario.model]
    video_cost   = video_tokens  * p["input"]
    prompt_cost  = prompt_tokens * p["input"]
    output_cost  = output_tokens * p["output"]
    cost_per_q   = video_cost + prompt_cost + output_cost
    daily        = scenario.daily_volume
    qacs         = round(min(100, tps / (cost_per_q * 10)), 1) if cost_per_q > 0 else 0

    return {
        **base,
        "clip_used":            Path(clip_path).name,
        "analyze_duration_sec": scenario.analyze_duration_sec,
        "ttft_ms":              ttft_ms,
        "tps":                  tps,
        "input_tokens":         input_tokens,
        "output_tokens":        output_tokens,
        "cost_per_query":       round(cost_per_q,  6),
        "daily_cost":           round(cost_per_q * daily,       2),
        "monthly_cost":         round(cost_per_q * daily * 30,  2),
        "annual_cost":          round(cost_per_q * daily * 365, 2),
        "qacs":                 qacs,
        "cost_breakdown": {
            "video_tokens":  int(video_tokens),
            "video_cost":    round(video_cost,   6),
            "video_pct":     round(video_tokens  / max(input_tokens, 1) * 100, 1),
            "prompt_tokens": int(prompt_tokens),
            "prompt_cost":   round(prompt_cost,  6),
            "prompt_pct":    round(prompt_tokens / max(input_tokens, 1) * 100, 1),
            "output_tokens": output_tokens,
            "output_cost":   round(output_cost,  6),
            "output_pct":    round(output_tokens / total_tokens * 100, 1),
        },
        "response_text": full_text[:600],
        "status":        "complete",
    }


# ── Endpoints ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "port": 8002}


@app.post("/upload-clip")
async def upload_clip(file: UploadFile = File(...)):
    """Accept a video file upload, save to temp, return clip_id."""
    ext    = Path(file.filename).suffix.lower() or ".mp4"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    data   = await file.read()
    tmp.write(data)
    tmp.close()

    clip_id = str(uuid.uuid4())
    _temp_clips[clip_id] = Path(tmp.name)

    # Try to get duration via ffprobe
    duration = None
    if shutil.which("ffprobe"):
        import subprocess
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", tmp.name],
                stderr=subprocess.DEVNULL
            )
            duration = float(out.decode().strip())
        except Exception:
            pass

    return {
        "clip_id":  clip_id,
        "clip_path": f"TEMP:{clip_id}",
        "filename": file.filename,
        "size_bytes": len(data),
        "duration_sec": duration,
    }


@app.get("/clip-exists/{clip_name}")
async def clip_exists(clip_name: str):
    """Check if a sample clip exists on the server."""
    try:
        resolve_clip_path(clip_name)
        return {"exists": True}
    except FileNotFoundError:
        return {"exists": False}


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    results = await asyncio.gather(
        *[run_one_scenario(s) for s in req.scenarios],
        return_exceptions=True,
    )
    cleaned = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            cleaned.append({
                "id":     req.scenarios[i].id,
                "name":   req.scenarios[i].name,
                "model":  req.scenarios[i].model,
                "status": "error",
                "error":  str(r),
            })
        else:
            cleaned.append(r)

    run = {
        "run_id":        str(uuid.uuid4()),
        "timestamp":     datetime.utcnow().isoformat(),
        "consumer_type": req.consumer_type,
        "use_case":      req.use_case,
        "scenarios":     cleaned,
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
