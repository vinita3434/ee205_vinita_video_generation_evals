import base64
import concurrent.futures
import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Project root (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

from backend.config import OUTPUT_DIR, OPENROUTER_API_KEY, JUDGE_MODEL, KLING_COST_PER_S, SORA_COST_PER_S
from backend.generators import generate_sora, generate_kling

app = FastAPI(title="Video Gen Evals", description="One prompt → 2 models (Kling, Sora) → Judge")


@app.exception_handler(Exception)
def catch_all(_request, exc):
    """Return 502 with real error message so the UI can show it (skip HTTPException)."""
    from fastapi.responses import JSONResponse
    if isinstance(exc, HTTPException):
        raise exc
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=502,
        content={"detail": str(exc)},
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    quality: str = "720p"   # 480p | 720p | 1080p  (Sora only; Kling is always Pro)
    ratio: str = "16:9"     # 16:9 | 9:16 | 1:1


# Actual output durations (seconds) — Kling 5s, Sora 8s (from generators)
KLING_DURATION_S = 5.0
SORA_DURATION_S = 8.0


class GenerateResponse(BaseModel):
    run_id: str
    kling_path: str
    sora_path: str
    kling_latency_s: float
    sora_latency_s: float
    kling_duration_s: float  # output video length for speed/cost math
    sora_duration_s: float
    kling_cost_usd: float
    sora_cost_usd: float


class JudgeRequest(BaseModel):
    prompt: str
    kling_path: str
    sora_path: str


def _generate_one(model: str, prompt: str, run_id: str, quality: str, ratio: str) -> tuple[str, Path, float]:
    start = time.time()
    if model == "kling":
        path = generate_kling(prompt, run_id, aspect_ratio=ratio)
    else:
        path = generate_sora(prompt, run_id, quality=quality, aspect_ratio=ratio)
    return model, path, time.time() - start


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate 2 videos from the same prompt (Kling 2.6, Sora 2)."""
    run_id = str(uuid.uuid4())[:8]
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_generate_one, "kling", prompt, run_id, request.quality, request.ratio): "kling",
            executor.submit(_generate_one, "sora", prompt, run_id, request.quality, request.ratio): "sora",
        }
        for fut in concurrent.futures.as_completed(futures):
            model = futures[fut]
            try:
                _, path, latency = fut.result()
                results[model] = (path, latency)
            except Exception as e:
                err_msg = str(e)
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=502, detail=f"{model}: {err_msg}")

    return GenerateResponse(
        run_id=run_id,
        kling_path=results["kling"][0].name,
        sora_path=results["sora"][0].name,
        kling_latency_s=round(results["kling"][1], 2),
        sora_latency_s=round(results["sora"][1], 2),
        kling_duration_s=KLING_DURATION_S,
        sora_duration_s=SORA_DURATION_S,
        kling_cost_usd=round(KLING_DURATION_S * KLING_COST_PER_S, 4),
        sora_cost_usd=round(SORA_DURATION_S * SORA_COST_PER_S, 4),
    )


# --- Frame extraction for judge (STEP 1) ---

def _get_video_duration_seconds(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True,
        timeout=10,
        text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return 5.0  # fallback
    return max(0.1, float(out.stdout.strip()))


def _extract_frames(video_path: Path) -> tuple[str, list[Path]]:
    """
    Extract 5 frames at 0%, 25%, 50%, 75%, 100% of video as JPEGs.
    Returns (temp_dir, [path0, path1, path2, path3, path4]). Caller must clean up temp_dir.
    """
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=503, detail="ffmpeg is required for frame extraction. Install ffmpeg.")
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(str(video_path))
    duration = _get_video_duration_seconds(video_path)
    # Frame times: first, 25%, 50%, 75%, last (cap last to duration - 0.05 to avoid past end)
    times = [
        0.0,
        duration * 0.25,
        duration * 0.5,
        duration * 0.75,
        max(0.0, duration - 0.05),
    ]
    temp_dir = tempfile.mkdtemp(prefix="video_judge_frames_")
    temp_path = Path(temp_dir)
    frame_paths: list[Path] = []
    for i, t in enumerate(times):
        out_file = temp_path / f"frame_{i}.jpg"
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
                "-vframes", "1", "-q:v", "2", str(out_file),
            ],
            capture_output=True,
            timeout=15,
        )
        if out_file.is_file():
            frame_paths.append(out_file)
    if len(frame_paths) != 5:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=502, detail=f"Frame extraction failed: got {len(frame_paths)}/5 frames")
    return temp_dir, frame_paths


def _frame_paths_to_data_urls(frame_paths: list[Path]) -> list[str]:
    """Convert list of JPEG paths to data:image/jpeg;base64,... URLs."""
    urls = []
    for p in frame_paths:
        b = p.read_bytes()
        b64 = base64.b64encode(b).decode("utf-8")
        urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


# STEP 2 — Judge rubrics (one API call per metric; each call gets 10 images: 5 from A, 5 from B)
METRIC_RUBRICS = [
    (
        "subject_consistency",
        "Subject Consistency",
        "You are evaluating frames from an AI-generated video in chronological order. "
        "Score Subject Consistency from 1-10. "
        "Criteria: Do the main subjects (people, objects) maintain the same appearance, "
        "color, and shape across all frames? "
        "10 = perfectly consistent. 1 = subject changes drastically across frames. "
        "First 5 images are Video A (Kling 2.6), next 5 are Video B (Sora 2), each in chronological order. "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    ),
    (
        "background_consistency",
        "Background Consistency",
        "You are evaluating frames from an AI-generated video in chronological order. "
        "Score Background Consistency from 1-10. "
        "Criteria: Does the environment and background remain stable across frames? "
        "10 = perfectly stable background. 1 = background changes or flickers heavily. "
        "First 5 images are Video A (Kling 2.6), next 5 are Video B (Sora 2), each in chronological order. "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    ),
    (
        "motion_smoothness",
        "Motion Smoothness",
        "You are evaluating frames from an AI-generated video in chronological order. "
        "Score Motion Smoothness from 1-10. "
        "Criteria: Do objects and subjects move naturally and fluidly between frames? "
        "Look for unnatural jumps, jerky movement, or frozen subjects. "
        "10 = perfectly smooth motion. 1 = very jerky or unnatural movement. "
        "First 5 images are Video A (Kling 2.6), next 5 are Video B (Sora 2), each in chronological order. "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    ),
    (
        "prompt_fidelity",
        "Prompt Fidelity",
        None,  # filled with prompt at runtime
    ),
    (
        "aesthetic_quality",
        "Aesthetic Quality",
        "You are evaluating frames from an AI-generated video. "
        "Score Aesthetic Quality from 1-10. "
        "Criteria: How visually appealing and cinematic are the frames? "
        "Consider lighting, composition, color grading, and overall visual polish. "
        "10 = extremely high quality visuals. 1 = poor visual quality. "
        "First 5 images are Video A (Kling 2.6), next 5 are Video B (Sora 2). "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    ),
    (
        "dynamic_degree",
        "Dynamic Degree",
        "You are evaluating frames from an AI-generated video in chronological order. "
        "Score Dynamic Degree from 1-10. "
        "Criteria: Is there meaningful motion happening in the video, "
        "or does it look like a static image with minimal movement? "
        "10 = rich dynamic motion throughout. 1 = nearly static, no meaningful motion. "
        "First 5 images are Video A (Kling 2.6), next 5 are Video B (Sora 2), each in chronological order. "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    ),
]


def _call_judge_metric(content: list, metric_name: str) -> dict:
    """One OpenRouter call with content (text + 10 images). Returns parsed {video_a: {score, reason}, video_b: {score, reason}}."""
    import httpx
    r = None
    last_exc = None
    for attempt in range(1, 5):
        try:
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": JUDGE_MODEL,
                    "messages": [{"role": "user", "content": content}],
                    "max_tokens": 512,
                },
                timeout=120.0,
            )
            if r.status_code in (502, 503, 504) and attempt < 4:
                time.sleep(5 * attempt)
                continue
            r.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            last_exc = e
            if e.response.status_code in (502, 503, 504) and attempt < 4:
                time.sleep(5 * attempt)
                continue
            raise
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_exc = e
            if attempt < 4:
                time.sleep(5 * attempt)
                continue
            raise HTTPException(status_code=502, detail=f"Judge ({metric_name}) failed after {attempt} attempts: {e!s}")
    if r is None or not r.is_success:
        raise HTTPException(status_code=502, detail=last_exc and str(last_exc) or "Judge request failed")
    raw = (r.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {"video_a": {"score": 0, "reason": "Parse error"}, "video_b": {"score": 0, "reason": "Parse error"}}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"video_a": {"score": 0, "reason": "Parse error"}, "video_b": {"score": 0, "reason": "Parse error"}}


@app.post("/judge")
def judge(request: JudgeRequest):
    """
    STEP 1: Extract 5 frames (0%, 25%, 50%, 75%, 100%) from each video.
    STEP 2: One API call per metric (6 calls), each with 10 images (5 from A, 5 from B) + rubric.
    STEP 3: Composite = average of 6 metric scores per video.
    STEP 4/5: Performance and cost are from /generate response (frontend combines).
    Clean up temp frame files after.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not set")

    prompt_text = request.prompt.strip()
    kling_full = OUTPUT_DIR / request.kling_path
    sora_full = OUTPUT_DIR / request.sora_path
    if not kling_full.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.kling_path}")
    if not sora_full.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.sora_path}")

    # STEP 1 — Frame extraction
    temp_dir_a = temp_dir_b = None
    try:
        temp_dir_a, frames_a = _extract_frames(kling_full)
        temp_dir_b, frames_b = _extract_frames(sora_full)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        urls_a = _frame_paths_to_data_urls(frames_a)
        urls_b = _frame_paths_to_data_urls(frames_b)
        # Build content for each metric: [text, img1a..img5a, img1b..img5b]
        image_parts_a = [{"type": "image_url", "image_url": {"url": u}} for u in urls_a]
        image_parts_b = [{"type": "image_url", "image_url": {"url": u}} for u in urls_b]

        # Prompt Fidelity rubric (insert prompt)
        prompt_fidelity_text = (
            "You are evaluating frames from an AI-generated video. "
            f"The original prompt was: {prompt_text!r}. "
            "Score Prompt Fidelity from 1-10. "
            "Criteria: How accurately does the video reflect the objects, actions, "
            "environment, and style described in the prompt? "
            "10 = perfectly matches prompt. 1 = completely ignores prompt. "
            "First 5 images are Video A (Kling 2.6), next 5 are Video B (Sora 2). "
            "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
        )

        results_a: dict = {}
        results_b: dict = {}

        for key, _label, rubric_template in METRIC_RUBRICS:
            if rubric_template is None:
                rubric_text = prompt_fidelity_text
            else:
                rubric_text = rubric_template
            content = [{"type": "text", "text": rubric_text}] + image_parts_a + image_parts_b
            parsed = _call_judge_metric(content, key)
            va = parsed.get("video_a") or {}
            vb = parsed.get("video_b") or {}
            results_a[key] = {"score": float(va.get("score", 0)), "reason": va.get("reason") or ""}
            results_b[key] = {"score": float(vb.get("score", 0)), "reason": vb.get("reason") or ""}

        # STEP 3 — Composite score
        def overall(metrics: dict) -> float:
            vals = [m["score"] for m in metrics.values()]
            return round(sum(vals) / 6, 1) if vals else 0.0

        overall_a = overall(results_a)
        overall_b = overall(results_b)

        return {
            "model": JUDGE_MODEL,
            "video_a": results_a,
            "video_b": results_b,
            "overall_quality_a": overall_a,
            "overall_quality_b": overall_b,
        }
    finally:
        # Clean up temp frame files
        if temp_dir_a:
            shutil.rmtree(temp_dir_a, ignore_errors=True)
        if temp_dir_b:
            shutil.rmtree(temp_dir_b, ignore_errors=True)


@app.get("/video/{filename}")
def serve_video(filename: str):
    """Serve a generated video file from OUTPUT_DIR (filename only, e.g. runid_wan.mp4)."""
    full = OUTPUT_DIR / filename
    if not full.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(full, media_type="video/mp4")


# Serve frontend at /
if FRONTEND_DIR.is_dir():
    @app.get("/")
    def index():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return {"message": "Video Gen Evals API", "docs": "/docs"}
