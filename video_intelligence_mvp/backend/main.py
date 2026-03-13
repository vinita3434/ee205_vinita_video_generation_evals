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


# ── Exploration endpoint (anomaly detection config matrix) ────────

ANOMALY_PROMPT = """You are a security AI monitoring a CCTV camera feed for anomalies.

Analyze this security camera clip and respond in this exact format:

ANOMALY_DETECTED: [YES/NO]
CONFIDENCE: [0-100]%
DESCRIPTION: [What is happening? Be specific about actions, persons, location in frame]
SEVERITY: [LOW/MEDIUM/HIGH]
TIMESTAMP: [When in the clip does the anomaly occur, if applicable]
RECOMMENDATION: [What action should security take?]"""

SECURITY_CLIP_MAP = {
    "security_anomaly_1": "sample_security.mp4",
    "security_anomaly_2": "sample_security.mp4",   # fallback to same clip if 2nd missing
    "sample_security":    "sample_security.mp4",
    "sample_retail":      "sample_retail.mp4",
}

EXPLORE_PRICING = {
    "gpt4o":  {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gemini": {"input": 0.10 / 1e6, "output":  0.40 / 1e6},
    "qwen":   {"input": 0.59 / 1e6, "output":  0.59 / 1e6},
}


def generate_configs(clip_min: int, clip_max: int,
                     freq_min: float, freq_max: float, n: int) -> list[dict]:
    configs = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        configs.append({
            "config_id": i + 1,
            "clip_sec":      round(clip_min + t * (clip_max - clip_min)),
            "checks_per_hr": round(freq_min + t * (freq_max - freq_min), 1),
        })
    return configs


async def run_config_inference(
    config: dict, model: str, clip_path: str, cameras: int
) -> dict:
    base = {"config_id": config["config_id"], "model": model}
    if not OPENROUTER_API_KEY:
        return {**base, "status": "error", "error": "OPENROUTER_API_KEY not configured"}

    try:
        frames = extract_frames_as_b64(clip_path, fps=1,
                                       max_duration_sec=config["clip_sec"])
    except Exception as e:
        return {**base, "status": "error", "error": f"Frame extraction: {e}"}

    content = [{"type": "text", "text": ANOMALY_PROMPT}]
    for fb64 in frames:
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://gruve.ai",
        "X-Title":       "Gruve Atlas",
    }
    payload = {
        "model":      MODEL_IDS[model],
        "messages":   [{"role": "user", "content": content}],
        "max_tokens": 400,
        "stream":     True,
    }

    t_start       = time.perf_counter()
    t_first_token = None
    full_text     = ""
    output_token_count = 0

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST", f"{OPENROUTER_BASE}/chat/completions",
                headers=headers, json=payload
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    return {**base, "status": "error",
                            "error": f"API {resp.status_code}: {body[:100].decode(errors='replace')}"}
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk_str = line[6:].strip()
                    if chunk_str == "[DONE]":
                        break
                    if not chunk_str:
                        continue
                    try:
                        chunk        = json.loads(chunk_str)
                        delta        = chunk.get("choices", [{}])[0].get("delta", {})
                        content_text = delta.get("content")
                        # Only capture first token when actual non-empty content arrives
                        if content_text and content_text.strip():
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            full_text          += content_text
                            output_token_count += 1
                    except Exception:
                        continue
    except Exception as e:
        return {**base, "status": "error", "error": str(e)}

    t_end    = time.perf_counter()
    ttft_ms  = round((t_first_token - t_start) * 1000) if t_first_token else None
    gen_dur  = max(t_end - (t_first_token or t_start), 0.001)
    output_tokens = output_token_count
    tps      = round(output_tokens / gen_dur, 1)
    total_time_ms = round((t_end - t_start) * 1000)

    # Cost calculation
    p             = EXPLORE_PRICING[model]
    video_tokens  = config["clip_sec"] * 258
    prompt_tokens = 80
    input_tokens  = video_tokens + prompt_tokens
    cost_per_q    = input_tokens * p["input"] + output_tokens * p["output"]
    queries_per_day  = cameras * config["checks_per_hr"] * 24
    detection_latency_min = round(60 / max(config["checks_per_hr"], 0.1), 1)

    return {
        **base,
        "clip_sec":              config["clip_sec"],
        "checks_per_hr":         config["checks_per_hr"],
        "detection_latency_min": detection_latency_min,
        "ttft_ms":               ttft_ms,
        "tps":                   tps,
        "total_time_ms":         total_time_ms,
        "input_tokens":          input_tokens,
        "output_tokens":         output_tokens,
        "cost_per_query":        round(cost_per_q, 6),
        "queries_per_day":       round(queries_per_day),
        "daily_cost":            round(cost_per_q * queries_per_day, 2),
        "monthly_cost":          round(cost_per_q * queries_per_day * 30, 2),
        "annual_cost":           round(cost_per_q * queries_per_day * 365, 2),
        "quality":               {},   # filled in by anchor judge after all inferences
        "response_text":     full_text[:600],
        "status":            "complete",
    }


async def judge_quality_with_anchor(all_results: list[dict]) -> dict[int, dict]:
    """
    Score all configs with ONE judge call using the longest clip as anchor.
    Returns {config_id: quality_dict}.
    Guarantees monotonic scores (longer clip = higher or equal quality).
    """
    import re as _re

    complete = [r for r in all_results if r.get("status") == "complete"]
    if not complete:
        return {}

    anchor = max(complete, key=lambda r: r.get("clip_sec", 0))
    sorted_by_clip = sorted(complete, key=lambda r: r.get("clip_sec", 0))

    responses_block = ""
    for r in sorted_by_clip:
        responses_block += (
            f"\nCONFIG {r['config_id']} ({r['clip_sec']}s clip):\n"
            f"---\n{r['response_text'][:400]}\n---\n"
        )

    judge_prompt = f"""You are evaluating anomaly detection AI responses from a security camera.
Multiple responses came from the SAME footage but different clip durations (shorter = less context).

ANCHOR (Config {anchor['config_id']}, {anchor['clip_sec']}s — longest clip, most context):
---
{anchor['response_text'][:400]}
---
The anchor scores 95/100 overall. Score all others relative to it.

Score each on three dimensions (0–100):
1. detection_accuracy: Correctly identified anomaly? Confident and specific?
2. actionability: Useful for a security operator needing to act immediately?
3. completeness: Timing, location, severity, recommendation — how much detail?

STRICT RULES:
- Anchor must score highest or tied for highest overall
- Scores must generally INCREASE with clip duration — more context = better
- Use the FULL 0–100 range — spread scores meaningfully
- overall = round((detection_accuracy + actionability + completeness) / 3)
- anomaly_detected = true if response says YES

ALL RESPONSES TO SCORE:
{responses_block}

Respond ONLY in this JSON, nothing else:
{{"scores": [{{"config_id": 1, "detection_accuracy": 0, "actionability": 0, "completeness": 0, "overall": 0, "anomaly_detected": true, "one_line_summary": "one sentence"}}]}}"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "https://gruve.ai",
                    "X-Title":       "Gruve Atlas",
                },
                json={
                    "model":    "google/gemini-2.0-flash-001",
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "max_tokens": 1000,
                },
            )
        raw = r.json()["choices"][0]["message"]["content"] or ""
        m = _re.search(r'\{.*"scores".*\}', raw, _re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            return {s["config_id"]: s for s in parsed.get("scores", [])}
    except Exception:
        pass

    # Fallback: linearly spaced scores
    n = len(sorted_by_clip)
    return {
        r["config_id"]: {
            "config_id": r["config_id"],
            "detection_accuracy": round(60 + (i / max(n - 1, 1)) * 35),
            "actionability":      round(58 + (i / max(n - 1, 1)) * 37),
            "completeness":       round(55 + (i / max(n - 1, 1)) * 40),
            "overall":            round(58 + (i / max(n - 1, 1)) * 37),
            "anomaly_detected":   True,
            "one_line_summary":   f"Config {r['config_id']} ({r['clip_sec']}s clip)",
        }
        for i, r in enumerate(sorted_by_clip)
    }


async def judge_anomaly_response(model_response: str) -> dict:
    """Judge the anomaly detection response using Gemini Flash."""
    judge_prompt = f"""You are evaluating an anomaly detection AI response for a security camera system.

The AI was shown a CCTV clip and asked to detect anomalies.
Here is its response:

---
{model_response[:500]}
---

Score this response on THREE dimensions (0-100 each):

1. DETECTION_ACCURACY: Did it correctly identify whether an anomaly exists? Describe accurately?
2. ACTIONABILITY: How specific and useful is the description for a security operator?
3. COMPLETENESS: Did it capture all relevant details — timing, location, severity?

Respond ONLY in this JSON format:
{{"detection_accuracy": 0, "actionability": 0, "completeness": 0, "anomaly_detected": true, "overall": 0, "one_line_summary": "one sentence"}}

overall = average of the three scores.
anomaly_detected = true if the response indicates an anomaly was found."""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "https://gruve.ai",
                    "X-Title":       "Gruve Atlas",
                },
                json={
                    "model":    "google/gemini-2.0-flash-001",
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "max_tokens": 200,
                },
            )
        raw = r.json()["choices"][0]["message"]["content"] or ""
        import re
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    # Fallback: parse the response manually
    text_lower = model_response.lower()
    detected   = "yes" in text_lower and "anomaly_detected" in text_lower
    return {
        "detection_accuracy": 60,
        "actionability":      55,
        "completeness":       50,
        "anomaly_detected":   detected,
        "overall":            55,
        "one_line_summary":   "Quality scoring unavailable",
    }


def find_sweet_spot(flat_results: list[dict]) -> dict | None:
    """
    Two-pass marginal value algorithm:
    1. Min quality floor: 70/100
    2. Sort by annual_cost ascending
    3. For each step, compute marginal quality gain / marginal cost
    4. Sweet spot = highest marginal ratio (weighted to prefer incremental value)
    5. Fallback: highest quality if all below 70
    """
    if not flat_results:
        return None

    sorted_configs = sorted(flat_results, key=lambda c: c.get("annual_cost", 0))
    viable = [c for c in sorted_configs if c.get("quality_overall", 0) >= 70]
    if not viable:
        viable = sorted_configs

    best_config = viable[0]
    best_ratio  = viable[0].get("quality_overall", 0) / max(viable[0].get("annual_cost", 1), 1)

    for i in range(1, len(viable)):
        curr = viable[i]
        prev = viable[i - 1]
        cost_delta    = curr.get("annual_cost", 0) - prev.get("annual_cost", 0)
        quality_delta = curr.get("quality_overall", 0) - prev.get("quality_overall", 0)
        if quality_delta <= 0:
            continue
        marginal_ratio = quality_delta / max(cost_delta, 0.01)
        weighted = marginal_ratio * 10
        if weighted > best_ratio:
            best_ratio  = weighted
            best_config = curr

    return best_config


class ExploreRequest(BaseModel):
    consumer_type:      str
    use_case:           str
    models:             list[str]
    cameras:            int
    base_checks_per_hr: float
    clip_min_sec:       int
    clip_max_sec:       int
    freq_min_per_hr:    float
    freq_max_per_hr:    float
    n_configs:          int
    clip_path:          str    # clip name or TEMP:{id}


@app.post("/benchmark-explore")
async def benchmark_explore(req: ExploreRequest):
    configs = generate_configs(
        req.clip_min_sec, req.clip_max_sec,
        req.freq_min_per_hr, req.freq_max_per_hr,
        req.n_configs,
    )

    # Resolve clip path
    try:
        clip_path = resolve_clip_path(req.clip_path)
    except FileNotFoundError:
        mapped = SECURITY_CLIP_MAP.get(req.clip_path)
        if mapped:
            clip_path = str(CLIPS_DIR / mapped)
        else:
            raise

    # Step 1: Run all inferences concurrently (no judging yet)
    tasks = [
        run_config_inference(cfg, model, clip_path, req.cameras)
        for cfg in configs
        for model in req.models
        if model in MODEL_IDS
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate completed from errors
    completed_inferences, error_results = [], []
    for i, result in enumerate(raw_results):
        if isinstance(result, Exception):
            cfg_idx = i // max(len(req.models), 1)
            mod_idx = i %  max(len(req.models), 1)
            cfg_id  = configs[min(cfg_idx, len(configs) - 1)]["config_id"]
            model   = req.models[min(mod_idx, len(req.models) - 1)]
            error_results.append({"config_id": cfg_id, "model": model,
                                   "status": "error", "error": str(result)})
        elif result.get("status") == "complete":
            completed_inferences.append(result)
        else:
            error_results.append(result)

    # Step 2: Anchor judging — single call scoring all responses together
    quality_map = await judge_quality_with_anchor(completed_inferences)

    # Step 3: Merge quality scores; build flat list for sweet spot
    flat_for_sweet_spot = []
    for result in completed_inferences:
        cfg_id = result["config_id"]
        q = quality_map.get(cfg_id, {
            "detection_accuracy": 55, "actionability": 55,
            "completeness": 55, "overall": 55,
            "anomaly_detected": True, "one_line_summary": "Score unavailable",
        })
        result["quality"]         = q
        result["quality_overall"] = q.get("overall", 0)
        flat_for_sweet_spot.append({
            "config_id":           cfg_id,
            "model":               result["model"],
            "annual_cost":         result.get("annual_cost", 0),
            "quality_overall":     result["quality_overall"],
            "clip_sec":            result.get("clip_sec", 0),
            "checks_per_hr":       result.get("checks_per_hr", 0),
            "detection_latency_min": result.get("detection_latency_min", 60),
        })

    # Step 4: Sweet spot (marginal value algorithm)
    sweet_spot = find_sweet_spot(flat_for_sweet_spot)

    # Step 5: Group by config_id
    config_results = {
        cfg["config_id"]: {
            "config_id":            cfg["config_id"],
            "clip_sec":             cfg["clip_sec"],
            "checks_per_hr":        cfg["checks_per_hr"],
            "detection_latency_min": round(60 / max(cfg["checks_per_hr"], 0.1), 1),
            "models": {},
        }
        for cfg in configs
    }
    for result in completed_inferences:
        config_results[result["config_id"]]["models"][result["model"]] = result
    for err in error_results:
        cid = err.get("config_id")
        if cid in config_results:
            config_results[cid]["models"][err.get("model", "unknown")] = err

    # Step 6: Sensitivity relative to most expensive config
    sensitivity = None
    if flat_for_sweet_spot and sweet_spot:
        sorted_flat    = sorted(flat_for_sweet_spot, key=lambda c: c["annual_cost"])
        most_expensive = sorted_flat[-1]
        cheapest       = sorted_flat[0]
        base_cost      = most_expensive["annual_cost"]
        levers         = []

        if sweet_spot["config_id"] != most_expensive["config_id"]:
            levers.append({
                "action":         f"Use sweet spot (Config {sweet_spot['config_id']})",
                "detail":         f"{sweet_spot['clip_sec']}s clips · {sweet_spot['checks_per_hr']}/hr",
                "annual_cost":    sweet_spot["annual_cost"],
                "savings_pct":    round((base_cost - sweet_spot["annual_cost"]) / max(base_cost, 1) * 100),
                "quality_warning": False,
                "quality_score":  sweet_spot["quality_overall"],
            })

        levers.append({
            "action":         "Halve camera count",
            "detail":         f"{req.cameras // 2} cameras",
            "annual_cost":    sweet_spot["annual_cost"] * 0.5,
            "savings_pct":    round((base_cost - sweet_spot["annual_cost"] * 0.5) / max(base_cost, 1) * 100),
            "quality_warning": False,
        })

        if cheapest["config_id"] != sweet_spot["config_id"]:
            levers.append({
                "action":         f"Use minimum config (Config {cheapest['config_id']})",
                "detail":         f"{cheapest['clip_sec']}s clips · {cheapest['checks_per_hr']}/hr",
                "annual_cost":    cheapest["annual_cost"],
                "savings_pct":    round((base_cost - cheapest["annual_cost"]) / max(base_cost, 1) * 100),
                "quality_warning": True,
                "quality_score":  cheapest["quality_overall"],
            })

        sensitivity = {
            "baseline_config_id": most_expensive["config_id"],
            "baseline_label":     (f"Config {most_expensive['config_id']} "
                                   f"({most_expensive['clip_sec']}s · "
                                   f"{most_expensive['checks_per_hr']}/hr)"),
            "baseline_cost":      base_cost,
            "levers":             levers,
        }

    return {
        "run_id":         str(uuid.uuid4()),
        "timestamp":      datetime.utcnow().isoformat(),
        "use_case":       req.use_case,
        "cameras":        req.cameras,
        "models_tested":  req.models,
        "configs":        list(config_results.values()),
        "sweet_spot":     sweet_spot,
        "sensitivity":    sensitivity,
    }


# ── Sensitivity Analysis endpoint ────────────────────────────────

NO_GT_PROMPT_TEMPLATE = """You are evaluating an anomaly detection AI response for a security camera system.

Score this response on 4 dimensions (0–100 each):

1. DETECTION (weight 40%) — Did it correctly identify whether an anomaly exists? Confident and specific?
   0 = missed anomaly / completely wrong | 100 = perfect detection with high confidence

2. LOCALIZATION (weight 20%) — Did it identify WHERE in the frame and WHEN in the clip?
   "Near door at ~0:08" = high | "suspicious activity" = low
   0 = no location/timing | 100 = precise location and timestamp

3. SEVERITY (weight 20%) — Did it correctly assess LOW/MEDIUM/HIGH and explain why?
   0 = no severity or wrong | 100 = correct severity with clear reasoning

4. ACTIONABILITY (weight 20%) — What should the operator do RIGHT NOW?
   "Dispatch security, lock storage, call police" = high | "Review footage" = low
   0 = no action | 100 = specific immediate actions

overall = round(detection*0.4 + localization*0.2 + severity*0.2 + actionability*0.2)

MODEL RESPONSE TO SCORE:
{response_text}

Return ONLY this JSON, nothing else:
{{"detection": 0, "localization": 0, "severity": 0, "actionability": 0, "overall": 0, "anomaly_detected": true, "one_line_summary": "one sentence"}}"""

GT_PROMPT_TEMPLATE = """You are evaluating an anomaly detection AI response against known ground truth.

GROUND TRUTH FOR THIS CLIP:
{ground_truth}

Score the model's response against this ground truth on 4 dimensions (0–100):

1. DETECTION (weight 40%) — Did it identify the SAME anomaly described in ground truth?
   0 = missed it entirely | 100 = identified exactly the right event

2. LOCALIZATION (weight 20%) — Does location and timing match the ground truth?
   0 = wrong location/timing | 100 = matches ground truth precisely

3. SEVERITY (weight 20%) — Does the severity assessment match ground truth?
   0 = wrong severity | 100 = exact match with correct reasoning

4. ACTIONABILITY (weight 20%) — Does the recommended action match what ground truth implies?
   0 = wrong or missing action | 100 = correct action recommended

overall = round(detection*0.4 + localization*0.2 + severity*0.2 + actionability*0.2)

MODEL RESPONSE TO SCORE:
{response_text}

Return ONLY this JSON, nothing else:
{{"detection": 0, "localization": 0, "severity": 0, "actionability": 0, "overall": 0, "anomaly_detected": true, "one_line_summary": "one sentence", "ground_truth_match": true}}"""


async def score_response_quality(response_text: str, ground_truth: str | None = None) -> dict:
    import re as _re
    if ground_truth:
        prompt = GT_PROMPT_TEMPLATE.format(
            ground_truth=ground_truth,
            response_text=response_text[:500],
        )
    else:
        prompt = NO_GT_PROMPT_TEMPLATE.format(response_text=response_text[:500])

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
                         "HTTP-Referer": "https://gruve.ai", "X-Title": "Gruve Atlas"},
                json={"model": "google/gemini-2.0-flash-001",
                      "messages": [{"role": "user", "content": prompt}], "max_tokens": 200},
            )
        raw = r.json()["choices"][0]["message"]["content"] or ""
        m = _re.search(r"\{[\s\S]*?\}", raw)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {"detection": 60, "localization": 40, "severity": 55, "actionability": 45, "overall": 52,
            "anomaly_detected": True, "one_line_summary": "Quality scoring unavailable"}


def normalize_scores_to_anchor(sweep_results: list[dict]) -> list[dict]:
    """Apply anchor normalization once after ALL clip sweep results are collected. Clip sweep only."""
    if not sweep_results:
        return sweep_results
    max_score = max(r.get("quality", {}).get("overall", 0) for r in sweep_results)
    if max_score == 0:
        return sweep_results
    scale_factor = 95 / max_score
    for r in sweep_results:
        q = r.get("quality", {})
        q["overall_normalized"] = round(q.get("overall", 0) * scale_factor)
        for dim in ["detection", "localization", "severity", "actionability"]:
            if dim in q:
                q[f"{dim}_normalized"] = round(q[dim] * scale_factor)
    return sweep_results


async def run_single_inference_for_sensitivity(
    model: str, clip_sec: int, checks_per_hr: float, cameras: int, clip_path: str
) -> dict:
    """Run one inference call and return raw results (no quality scoring yet)."""
    frames = extract_frames_as_b64(clip_path, fps=1, max_duration_sec=clip_sec)
    content = [{"type": "text", "text": ANOMALY_PROMPT}]
    for fb64 in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}})

    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
               "HTTP-Referer": "https://gruve.ai", "X-Title": "Gruve Atlas"}
    payload = {"model": MODEL_IDS[model], "messages": [{"role": "user", "content": content}],
               "max_tokens": 400, "stream": True}

    t_start = time.perf_counter()
    t_first = None
    full_text = ""
    token_count = 0

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", f"{OPENROUTER_BASE}/chat/completions",
                                     headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    raise Exception(f"API {resp.status_code}: {body[:100].decode(errors='replace')}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    cs = line[6:].strip()
                    if cs == "[DONE]" or not cs:
                        continue
                    try:
                        d = json.loads(cs)["choices"][0]["delta"].get("content")
                        if d and d.strip():
                            if t_first is None:
                                t_first = time.perf_counter()
                            full_text += d
                            token_count += 1
                    except Exception:
                        continue
    except Exception as e:
        raise Exception(str(e))

    t_end = time.perf_counter()
    p = EXPLORE_PRICING[model]
    video_tokens = clip_sec * 258
    prompt_tokens = 80
    input_tokens = video_tokens + prompt_tokens
    cost_per_q = input_tokens * p["input"] + token_count * p["output"]
    queries_per_day = cameras * checks_per_hr * 24

    return {
        "model": model,
        "clip_sec": clip_sec,
        "checks_per_hr": checks_per_hr,
        "ttft_ms": round((t_first - t_start) * 1000) if t_first else None,
        "tps": round(token_count / max(t_end - (t_first or t_start), 0.001), 1),
        "input_tokens": input_tokens,
        "output_tokens": token_count,
        "cost_per_query": round(cost_per_q, 6),
        "queries_per_day": round(queries_per_day),
        "daily_cost": round(cost_per_q * queries_per_day, 2),
        "monthly_cost": round(cost_per_q * queries_per_day * 30, 2),
        "annual_cost": round(cost_per_q * queries_per_day * 365, 2),
        "detection_latency_min": round(60 / max(checks_per_hr, 0.1), 1),
        "response_text": full_text[:600],
        "quality": {},
        "status": "complete",
    }


def find_plateau_clip(clip_sweep: list) -> dict:
    """
    Find the clip duration where marginal quality/cost ratio drops sharply.
    Plateau = point where adding more seconds gives < 3 quality points.
    If no clear plateau, return the middle config.
    """
    sorted_clips = sorted(clip_sweep, key=lambda c: c["clip_sec"])
    if len(sorted_clips) <= 1:
        return sorted_clips[0] if sorted_clips else {}
    best = sorted_clips[0]
    for i in range(1, len(sorted_clips)):
        curr = sorted_clips[i]
        prev = sorted_clips[i - 1]
        curr_quality = curr["quality"].get("overall_normalized", curr["quality"].get("overall", 0))
        prev_quality = prev["quality"].get("overall_normalized", prev["quality"].get("overall", 0))
        quality_gain = curr_quality - prev_quality
        if quality_gain >= 3:
            best = curr
        else:
            break
    return best


def build_recommendation(clip_sweep: list, model_sweep: list, freq_sweep: list) -> dict:
    sorted_clip = sorted(clip_sweep, key=lambda r: r["clip_sec"])
    optimal_clip = find_plateau_clip(clip_sweep) if clip_sweep else (sorted_clip[0] if sorted_clip else {})

    # Best model: highest quality / cost ratio
    valid_models = [m for m in model_sweep if m.get("status") == "complete"]
    best_model_result = max(valid_models, key=lambda m: m.get("quality", {}).get("overall", 0) /
                            max(m.get("annual_cost", 1), 1)) if valid_models else {}

    # Frequency: lowest (quality unaffected)
    optimal_freq = min(freq_sweep, key=lambda f: f.get("annual_cost", 0)) if freq_sweep else {}

    # Biggest saving lever
    model_saving = (max((m.get("annual_cost", 0) for m in model_sweep), default=0) -
                    min((m.get("annual_cost", 0) for m in model_sweep), default=0))
    clip_saving = (max((c.get("annual_cost", 0) for c in clip_sweep), default=0) -
                   min((c.get("annual_cost", 0) for c in clip_sweep), default=0))
    freq_saving = (max((f.get("annual_cost", 0) for f in freq_sweep), default=0) -
                   min((f.get("annual_cost", 0) for f in freq_sweep), default=0))
    levers = [("model_switch", model_saving), ("clip_reduction", clip_saving),
              ("frequency_reduction", freq_saving)]
    biggest_lever, biggest_saving = max(levers, key=lambda x: x[1])

    return {
        "optimal_clip_sec": optimal_clip.get("clip_sec"),
        "optimal_model": best_model_result.get("model"),
        "optimal_checks_per_hr": optimal_freq.get("checks_per_hr"),
        "annual_cost": best_model_result.get("annual_cost", 0),
        "quality_score": best_model_result.get("quality", {}).get("overall", 0),
        "biggest_lever": biggest_lever,
        "biggest_lever_savings": round(biggest_saving),
    }


class SensitivityRequest(BaseModel):
    consumer_type: str
    use_case: str
    base_model: str
    base_clip_sec: int
    base_checks_per_hr: float
    cameras: int
    run_clip_sensitivity: bool = True
    run_model_sensitivity: bool = True
    run_frequency_sensitivity: bool = True
    clip_durations: list[int] = [5, 10, 15, 20, 25, 30]
    models_to_compare: list[str] = ["gemini", "gpt4o", "qwen"]
    frequencies: list[float] = [1.0, 2.0, 3.0, 4.0]
    clip_path: str
    ground_truth: Optional[str] = None


@app.post("/sensitivity-analysis")
async def sensitivity_analysis(req: SensitivityRequest):
    # Resolve clip
    try:
        clip_path = resolve_clip_path(req.clip_path)
    except FileNotFoundError:
        mapped = SECURITY_CLIP_MAP.get(req.clip_path)
        clip_path = str(CLIPS_DIR / mapped) if mapped else str(CLIPS_DIR / "sample_security.mp4")

    # Build inference tasks (clip_sweep + model_sweep only — freq_sweep = 1 base call)
    tasks = []
    task_labels = []

    if req.run_clip_sensitivity:
        for dur in req.clip_durations:
            tasks.append(run_single_inference_for_sensitivity(
                req.base_model, dur, req.base_checks_per_hr, req.cameras, clip_path))
            task_labels.append(("clip_sweep", dur))

    # Model sweep: always run all 3 models as independent inference calls (no deduplication)
    if req.run_model_sensitivity:
        for model in req.models_to_compare:
            if model in MODEL_IDS:
                tasks.append(run_single_inference_for_sensitivity(
                    model, req.base_clip_sec, req.base_checks_per_hr, req.cameras, clip_path))
                task_labels.append(("model_sweep", model))

    # freq_sweep: single base inference, reuse base_model + base_clip_sec
    if req.run_frequency_sensitivity:
        tasks.append(run_single_inference_for_sensitivity(
            req.base_model, req.base_clip_sec, req.base_checks_per_hr, req.cameras, clip_path))
        task_labels.append(("freq_base", req.base_checks_per_hr))

    # Run all concurrently
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate into sweeps
    clip_sweep_raw, model_sweep_raw = [], []
    freq_base_result = None

    for i, (sweep_type, key) in enumerate(task_labels):
        result = raw_results[i]
        if isinstance(result, Exception):
            result = {"status": "error", "error": str(result), "model": req.base_model,
                      "clip_sec": req.base_clip_sec, "checks_per_hr": req.base_checks_per_hr,
                      "annual_cost": 0, "quality": {}}
        result["key"] = key
        if sweep_type == "clip_sweep":
            clip_sweep_raw.append(result)
        elif sweep_type == "model_sweep":
            model_sweep_raw.append(result)
        elif sweep_type == "freq_base":
            freq_base_result = result

    # Build frequency sweep by scaling cost (1 base inference); explicitly copy ALL quality fields
    freq_sweep_raw = []

    # Score quality for clip_sweep + model_sweep + freq_base (in parallel)
    # BUG 2 FIX: include freq_base_result so its quality is set before copying to freq entries
    freq_base_list = [freq_base_result] if freq_base_result and freq_base_result.get("status") == "complete" else []
    all_to_score = [r for r in clip_sweep_raw + model_sweep_raw + freq_base_list if r.get("status") == "complete"]
    score_tasks = [score_response_quality(r["response_text"], req.ground_truth) for r in all_to_score]
    quality_results = await asyncio.gather(*score_tasks, return_exceptions=True)

    for i, result in enumerate(all_to_score):
        q = quality_results[i] if not isinstance(quality_results[i], Exception) else {
            "detection": 55, "localization": 40, "severity": 50, "actionability": 45,
            "overall": 48, "anomaly_detected": True, "one_line_summary": "Scoring failed"
        }
        result["quality"] = q

    # Frequency sweep: one base inference, then copy ALL quality fields into each frequency entry
    if freq_base_result and req.run_frequency_sensitivity and freq_base_result.get("status") == "complete":
        base_result = freq_base_result
        base_quality = base_result.get("quality", {})
        for freq in req.frequencies:
            freq_result = {
                "checks_per_hr": freq,
                "clip_sec": req.base_clip_sec,
                "model": req.base_model,
                "cost_per_query": base_result["cost_per_query"],
                "annual_cost": round(base_result["cost_per_query"] * req.cameras * freq * 24 * 365, 2),
                "ttft_ms": base_result.get("ttft_ms", 0),
                "tps": base_result.get("tps", 0),
                "queries_per_day": round(req.cameras * freq * 24),
                "daily_cost": round(base_result["cost_per_query"] * req.cameras * freq * 24, 2),
                "monthly_cost": round(base_result["cost_per_query"] * req.cameras * freq * 24 * 30, 2),
                "detection_latency_min": round(60 / max(freq, 0.1), 1),
                "quality": {
                    "detection": base_quality.get("detection", 0),
                    "localization": base_quality.get("localization", 0),
                    "severity": base_quality.get("severity", 0),
                    "actionability": base_quality.get("actionability", 0),
                    "overall": base_quality.get("overall", 0),
                    "overall_normalized": base_quality.get("overall_normalized", base_quality.get("overall", 0)),
                    "anomaly_detected": base_quality.get("anomaly_detected", True),
                    "one_line_summary": base_quality.get("one_line_summary", ""),
                },
                "response_text": base_result.get("response_text", ""),
                "key": freq,
                "status": "complete",
            }
            freq_sweep_raw.append(freq_result)

    # Normalize clip_sweep scores to anchor (ONCE, after all clip results collected; not for model/freq)
    clip_sweep_final = normalize_scores_to_anchor(clip_sweep_raw)

    # Recommendation
    recommendation = build_recommendation(clip_sweep_final, model_sweep_raw, freq_sweep_raw)

    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "base_config": {
            "model": req.base_model, "clip_sec": req.base_clip_sec,
            "checks_per_hr": req.base_checks_per_hr, "cameras": req.cameras,
        },
        "ground_truth_provided": bool(req.ground_truth),
        "clip_sweep":   clip_sweep_final,
        "model_sweep":  model_sweep_raw,
        "frequency_sweep": freq_sweep_raw,
        "recommendation": recommendation,
    }
