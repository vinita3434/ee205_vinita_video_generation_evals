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
    model: str | None = None  # "sora" | "kling" = run this model twice for Version A & B; None = run both models once


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
    """Generate 2 videos from the same prompt. If model is set, run that model twice (Version A & B); else run Kling + Sora once each."""
    run_id = str(uuid.uuid4())[:8]
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    single_model = request.model if request.model in ("sora", "kling") else None
    results = {}

    if single_model:
        # Run the same model twice for Version A and Version B
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(_generate_one, single_model, prompt, run_id + "_a", request.quality, request.ratio): "a",
                executor.submit(_generate_one, single_model, prompt, run_id + "_b", request.quality, request.ratio): "b",
            }
            for fut in concurrent.futures.as_completed(futures):
                key = futures[fut]
                try:
                    _, path, latency = fut.result()
                    results[key] = (path, latency)
                except Exception as e:
                    err_msg = str(e)
                    import traceback
                    traceback.print_exc()
                    raise HTTPException(status_code=502, detail=f"{single_model} ({key}): {err_msg}")
        # Map to response: Version A = first run, Version B = second run; response reuses sora/kling fields
        dur = SORA_DURATION_S if single_model == "sora" else KLING_DURATION_S
        cost_per_s = SORA_COST_PER_S if single_model == "sora" else KLING_COST_PER_S
        return GenerateResponse(
            run_id=run_id,
            sora_path=results["a"][0].name,
            kling_path=results["b"][0].name,
            sora_latency_s=round(results["a"][1], 2),
            kling_latency_s=round(results["b"][1], 2),
            sora_duration_s=dur,
            kling_duration_s=dur,
            sora_cost_usd=round(dur * cost_per_s, 4),
            kling_cost_usd=round(dur * cost_per_s, 4),
        )
    else:
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


# Weighted panel: 3 judges via OpenRouter
JUDGE_PANEL = [
    ("gemini", "google/gemini-pro-1.5", "Gemini 1.5 Pro"),
    ("gpt4o", "openai/gpt-4o", "GPT-4o"),
    ("claude", "anthropic/claude-3.5-sonnet", "Claude 3.5 Sonnet"),
]
# Per-metric weights: metric_key -> judge_key -> weight (sum = 1.0)
PANEL_WEIGHTS = {
    "identity_preservation": {"gemini": 0.60, "gpt4o": 0.25, "claude": 0.15},
    "scene_coherence": {"gemini": 0.50, "gpt4o": 0.25, "claude": 0.25},
    "temporal_fluency": {"gemini": 0.60, "gpt4o": 0.20, "claude": 0.20},
    "semantic_alignment": {"gpt4o": 0.50, "gemini": 0.30, "claude": 0.20},
    "cinematic_quality": {"claude": 0.50, "gpt4o": 0.30, "gemini": 0.20},
}
VARIANCE_THRESHOLD = 2.0  # flag if max - min judge score > this

# 5 metrics for weighted panel (same frames + rubric sent to all 3 judges)
PANEL_METRIC_RUBRICS = [
    (
        "identity_preservation",
        "Identity Preservation",
        "You are evaluating frames from an AI-generated video in chronological order. "
        "Score Identity Preservation from 1-10. "
        "Criteria: Do the main subjects (people, objects) maintain the same appearance, "
        "color, and shape across all frames? "
        "10 = perfectly consistent. 1 = subject changes drastically across frames. "
        "First 5 images are Video A, next 5 are Video B, each in chronological order. "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    ),
    (
        "scene_coherence",
        "Scene Coherence",
        "You are evaluating frames from an AI-generated video in chronological order. "
        "Score Scene Coherence from 1-10. "
        "Criteria: Does the environment and background remain stable across frames? "
        "10 = perfectly stable. 1 = background changes or flickers heavily. "
        "First 5 images are Video A, next 5 are Video B, each in chronological order. "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    ),
    (
        "temporal_fluency",
        "Temporal Fluency",
        "You are evaluating frames from an AI-generated video in chronological order. "
        "Score Temporal Fluency from 1-10. "
        "Criteria: Do objects move naturally and fluidly between frames? "
        "Look for unnatural jumps, jerky movement, or frozen subjects. "
        "10 = perfectly smooth. 1 = very jerky or unnatural. "
        "First 5 images are Video A, next 5 are Video B, each in chronological order. "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    ),
    (
        "semantic_alignment",
        "Semantic Alignment",
        None,  # filled with prompt at runtime
    ),
    (
        "cinematic_quality",
        "Cinematic Quality",
        "You are evaluating frames from an AI-generated video. "
        "Score Cinematic Quality from 1-10. "
        "Criteria: How visually appealing and cinematic are the frames? "
        "Consider lighting, composition, color grading, and overall visual polish. "
        "10 = extremely high quality. 1 = poor visual quality. "
        "First 5 images are Video A, next 5 are Video B. "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    ),
]


def _call_one_judge(content: list, model_id: str, metric_name: str) -> dict:
    """One OpenRouter call with given model. Returns {video_a: {score, reason}, video_b: {score, reason}}."""
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
                    "model": model_id,
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
    Weighted panel: 3 judges (Gemini 1.5 Pro, GPT-4o, Claude 3.5 Sonnet).
    STEP 1: Extract 5 frames per video.
    STEP 2: For each of 5 metrics, send same frames + rubric to all 3 judges in parallel.
    STEP 3: Weighted score per metric; high-variance flag if judge spread > 2.
    STEP 4: overall_quality = average of 5 weighted scores; consensus reason from majority judge.
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

    temp_dir_a = temp_dir_b = None
    try:
        temp_dir_a, frames_a = _extract_frames(kling_full)
        temp_dir_b, frames_b = _extract_frames(sora_full)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    semantic_alignment_text = (
        "You are evaluating frames from an AI-generated video. "
        f"The original prompt was: {prompt_text!r}. "
        "Score Semantic Alignment from 1-10. "
        "Criteria: How accurately does the video reflect the objects, actions, "
        "environment, and style described in the prompt? "
        "10 = perfectly matches prompt. 1 = completely ignores prompt. "
        "First 5 images are Video A, next 5 are Video B. "
        "Return only a JSON: {\"video_a\": {\"score\": X, \"reason\": \"one sentence\"}, \"video_b\": {\"score\": X, \"reason\": \"one sentence\"}}",
    )

    try:
        urls_a = _frame_paths_to_data_urls(frames_a)
        urls_b = _frame_paths_to_data_urls(frames_b)
        image_parts_a = [{"type": "image_url", "image_url": {"url": u}} for u in urls_a]
        image_parts_b = [{"type": "image_url", "image_url": {"url": u}} for u in urls_b]

        results_a: dict = {}
        results_b: dict = {}

        for key, _label, rubric_template in PANEL_METRIC_RUBRICS:
            rubric_text = semantic_alignment_text if rubric_template is None else rubric_template
            content = [{"type": "text", "text": rubric_text}] + image_parts_a + image_parts_b

            # Call all 3 judges in parallel for this metric
            panel_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futures = {
                    ex.submit(_call_one_judge, content, model_id, key): (jkey, display_name)
                    for jkey, model_id, display_name in JUDGE_PANEL
                }
                for fut in concurrent.futures.as_completed(futures):
                    jkey, _ = futures[fut]
                    try:
                        parsed = fut.result()
                        panel_results.append((jkey, parsed))
                    except Exception:
                        panel_results.append((jkey, {"video_a": {"score": 0, "reason": "Error"}, "video_b": {"score": 0, "reason": "Error"}}))

            weights = PANEL_WEIGHTS[key]
            scores_a = {}
            scores_b = {}
            reasons_a = {}
            reasons_b = {}
            for jkey, parsed in panel_results:
                va = parsed.get("video_a") or {}
                vb = parsed.get("video_b") or {}
                sa = float(va.get("score", 0))
                sb = float(vb.get("score", 0))
                scores_a[jkey] = sa
                scores_b[jkey] = sb
                reasons_a[jkey] = va.get("reason") or ""
                reasons_b[jkey] = vb.get("reason") or ""

            weighted_a = sum(scores_a.get(j, 0) * weights.get(j, 0) for j in weights)
            weighted_b = sum(scores_b.get(j, 0) * weights.get(j, 0) for j in weights)
            spread_a = max(scores_a.values()) - min(scores_a.values()) if scores_a else 0
            spread_b = max(scores_b.values()) - min(scores_b.values()) if scores_b else 0
            high_variance_a = spread_a > VARIANCE_THRESHOLD
            high_variance_b = spread_b > VARIANCE_THRESHOLD

            results_a[key] = {
                "gemini_score": scores_a.get("gemini"),
                "gpt4o_score": scores_a.get("gpt4o"),
                "claude_score": scores_a.get("claude"),
                "weighted_score": round(weighted_a, 1),
                "reason_gemini": reasons_a.get("gemini", ""),
                "reason_gpt4o": reasons_a.get("gpt4o", ""),
                "reason_claude": reasons_a.get("claude", ""),
                "high_variance": high_variance_a,
                "variance_note": "Judges disagree on this metric — consider human review" if high_variance_a else None,
            }
            results_b[key] = {
                "gemini_score": scores_b.get("gemini"),
                "gpt4o_score": scores_b.get("gpt4o"),
                "claude_score": scores_b.get("claude"),
                "weighted_score": round(weighted_b, 1),
                "reason_gemini": reasons_b.get("gemini", ""),
                "reason_gpt4o": reasons_b.get("gpt4o", ""),
                "reason_claude": reasons_b.get("claude", ""),
                "high_variance": high_variance_b,
                "variance_note": "Judges disagree on this metric — consider human review" if high_variance_b else None,
            }

        # Overall = average of 5 weighted scores
        def overall(metrics: dict) -> float:
            vals = [m["weighted_score"] for m in metrics.values()]
            return round(sum(vals) / 5, 1) if vals else 0.0

        overall_a = overall(results_a)
        overall_b = overall(results_b)

        # Consensus reason: majority judge (Gemini) reason from a high-weight metric. video_a = Kling, video_b = Sora.
        def _consensus(results: dict) -> str:
            r = results.get("identity_preservation") or results.get("temporal_fluency") or {}
            if isinstance(r, dict):
                out = r.get("reason_gemini") or r.get("reason_gpt4o") or r.get("reason_claude") or ""
                if out:
                    return out
            for v in results.values():
                if isinstance(v, dict):
                    out = v.get("reason_gemini") or v.get("reason_gpt4o") or v.get("reason_claude") or ""
                    if out:
                        return out
            return ""

        overall_reason_a = _consensus(results_b)  # Model A = Sora = video_b
        overall_reason_b = _consensus(results_a)  # Model B = Kling = video_a

        return {
            "panel_models": [display_name for _j, _mid, display_name in JUDGE_PANEL],
            "video_a": results_a,
            "video_b": results_b,
            "overall_quality_a": overall_a,
            "overall_quality_b": overall_b,
            "overall_reason_a": overall_reason_a,
            "overall_reason_b": overall_reason_b,
        }
    finally:
        if temp_dir_a:
            shutil.rmtree(temp_dir_a, ignore_errors=True)
        if temp_dir_b:
            shutil.rmtree(temp_dir_b, ignore_errors=True)


# ── Scenario-based generation + judge (new Video Gen flow) ───────

GEN_PRICING = {
    "sora2":   {"per_second": 0.15},
    "kling26": {"per_second": 0.08},
}
RES_MULTIPLIER = {"480p": 0.5, "720p": 1.0, "1080p": 2.0, "4k": 4.0}
FPS_MULTIPLIER = {16: 0.8, 20: 1.0, 24: 1.3}

USE_CASE_SEED_PROMPTS = {
    "brand_storytelling":    "A cinematic brand story video showing a product journey",
    "client_pitch":          "A professional concept video for a client presentation",
    "social_content":        "An engaging short-form social media video",
    "employee_onboarding":   "A clear and professional employee training video",
    "product_demo":          "A clean product demonstration and explainer video",
    "executive_comms":       "A polished executive communication video",
    "performance_ad":        "A high-converting performance advertisement video",
    "product_showcase":      "A dynamic product showcase video",
    "ab_creative":           "A creative test variant video for A/B testing",
    "lookbook":              "A fashion lookbook video with editorial styling",
    "product_detail":        "A detailed product styling video",
    "runway_editorial":      "A runway and editorial fashion video",
    "short_form":            "An engaging short-form entertainment video",
    "news_broadcast":        "A professional news broadcast package video",
    "trailer_promo":         "A cinematic trailer and promotional video",
}


def build_gen_prompt(use_case: str, quality_buckets: dict) -> str:
    base = USE_CASE_SEED_PROMPTS.get(use_case, "A professional video")
    quality_instructions = []
    if quality_buckets.get("visual_fidelity", 5) >= 7:
        quality_instructions.append("cinematic quality, sharp details, professional color grading")
    if quality_buckets.get("motion_quality", 5) >= 7:
        quality_instructions.append("smooth motion, no flickering, fluid transitions")
    if quality_buckets.get("subject_quality", 5) >= 7:
        quality_instructions.append("consistent subjects, accurate semantic alignment")
    if quality_buckets.get("scene_stability", 5) >= 7:
        quality_instructions.append("stable background, controlled dynamic elements")
    if quality_instructions:
        return f"{base}. Quality requirements: {', '.join(quality_instructions)}."
    return f"{base}. Fast generation, standard quality."


EVAL_JUDGE_RUBRICS = [
    ("subject_quality", "Subject Quality",
     "Evaluate the generated video frames for Subject Quality (0.0–1.0). "
     "Criteria: Accuracy and consistency of subjects across frames — are they realistic, correctly rendered, semantically correct? "
     "Do NOT reference resolution, fps, or any input parameters — evaluate OUTPUT only. "
     "Return ONLY JSON: {\"score_a\": 0.0, \"score_b\": 0.0, \"reasoning\": \"one sentence\"}"),
    ("motion_quality", "Motion Quality",
     "Evaluate the generated video frames for Motion Quality (0.0–1.0). "
     "Criteria: Smoothness and naturalness of movement — no flickering, no unnatural jumps, fluid transitions. "
     "Do NOT reference resolution, fps, or any input parameters — evaluate OUTPUT only. "
     "Return ONLY JSON: {\"score_a\": 0.0, \"score_b\": 0.0, \"reasoning\": \"one sentence\"}"),
    ("visual_fidelity", "Visual Fidelity",
     "Evaluate the generated video frames for Visual Fidelity (0.0–1.0). "
     "Criteria: Overall image quality — lighting, composition, color grading, sharpness, aesthetic polish. "
     "Do NOT reference resolution, fps, or any input parameters — evaluate OUTPUT only. "
     "Return ONLY JSON: {\"score_a\": 0.0, \"score_b\": 0.0, \"reasoning\": \"one sentence\"}"),
    ("scene_stability", "Scene Stability",
     "Evaluate the generated video frames for Scene Stability (0.0–1.0). "
     "Criteria: Consistency of background and environment — does the scene remain stable without unwanted changes? "
     "Do NOT reference resolution, fps, or any input parameters — evaluate OUTPUT only. "
     "Return ONLY JSON: {\"score_a\": 0.0, \"score_b\": 0.0, \"reasoning\": \"one sentence\"}"),
    ("overall_quality", "Overall Quality",
     "Evaluate these generated video frames for Overall Quality (0.0–1.0). "
     "Give your holistic assessment of the visual output quality across all dimensions. "
     "Do NOT reference resolution, fps, or any input parameters — evaluate OUTPUT only. "
     "Return ONLY JSON: {\"score_a\": 0.0, \"score_b\": 0.0, \"reasoning\": \"one sentence\"}"),
]

EVAL_JUDGE_MODELS = [
    ("claude", "anthropic/claude-3-5-sonnet"),
    ("gemini", "google/gemini-2.0-flash-001"),
    ("gpt4o",  "openai/gpt-4o"),
]


def _call_eval_judge(frames_a_urls: list, frames_b_urls: list,
                     rubric: str, model_id: str) -> dict:
    import httpx
    content = (
        [{"type": "text", "text": "Scenario A frames (chronological):"}]
        + [{"type": "image_url", "image_url": {"url": u}} for u in frames_a_urls]
        + [{"type": "text", "text": "Scenario B frames (chronological):"}]
        + [{"type": "image_url", "image_url": {"url": u}} for u in frames_b_urls]
        + [{"type": "text", "text": rubric}]
    )
    try:
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
            json={"model": model_id, "messages": [{"role": "user", "content": content}], "max_tokens": 300},
            timeout=90.0,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"] or ""
        m = re.search(r"\{[\s\S]*?\}", raw)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {"score_a": 0.0, "score_b": 0.0, "reasoning": "Unavailable"}


def run_eval_judge_panel(path_a: Path, path_b: Path) -> dict:
    """Extract frames from both videos, run 3 judges × 5 dimensions in parallel."""
    temp_dir_a = temp_dir_b = None
    try:
        temp_dir_a, frames_a = _extract_frames(path_a)
        temp_dir_b, frames_b = _extract_frames(path_b)
        urls_a = _frame_paths_to_data_urls(frames_a)
        urls_b = _frame_paths_to_data_urls(frames_b)
    except Exception as e:
        return {"error": str(e)}

    dimensions = [r[0] for r in EVAL_JUDGE_RUBRICS]
    # All 15 calls (5 dims × 3 judges) in parallel
    futures_map = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        for dim_key, _dim_label, rubric in EVAL_JUDGE_RUBRICS:
            for judge_key, model_id in EVAL_JUDGE_MODELS:
                fut = ex.submit(_call_eval_judge, urls_a, urls_b, rubric, model_id)
                futures_map[fut] = (dim_key, judge_key)

        raw_results: dict[str, dict[str, dict]] = {d: {} for d in dimensions}
        for fut in concurrent.futures.as_completed(futures_map):
            dim_key, judge_key = futures_map[fut]
            try:
                raw_results[dim_key][judge_key] = fut.result()
            except Exception:
                raw_results[dim_key][judge_key] = {"score_a": 0.0, "score_b": 0.0, "reasoning": "Error"}

    # Build per-dimension consensus
    result_a, result_b = {}, {}
    disagreements = []
    for dim_key in dimensions:
        scores_a = {j: raw_results[dim_key].get(j, {}).get("score_a", 0.0) for j, _ in EVAL_JUDGE_MODELS}
        scores_b = {j: raw_results[dim_key].get(j, {}).get("score_b", 0.0) for j, _ in EVAL_JUDGE_MODELS}
        vals_a = list(scores_a.values())
        vals_b = list(scores_b.values())
        consensus_a = round(sum(vals_a) / len(vals_a), 3) if vals_a else 0.0
        consensus_b = round(sum(vals_b) / len(vals_b), 3) if vals_b else 0.0
        # Disagreement: std dev > 0.2
        mean_a = consensus_a
        std_a = (sum((s - mean_a) ** 2 for s in vals_a) / len(vals_a)) ** 0.5 if vals_a else 0
        if std_a > 0.2:
            disagreements.append(dim_key)
        result_a[dim_key] = {"consensus": consensus_a, "individual": scores_a}
        result_b[dim_key] = {"consensus": consensus_b, "individual": scores_b}

    overall_a = round(sum(v["consensus"] for v in result_a.values()) / len(result_a), 3)
    overall_b = round(sum(v["consensus"] for v in result_b.values()) / len(result_b), 3)

    if temp_dir_a:
        shutil.rmtree(temp_dir_a, ignore_errors=True)
    if temp_dir_b:
        shutil.rmtree(temp_dir_b, ignore_errors=True)

    return {
        "scenario_a": result_a,
        "scenario_b": result_b,
        "overall_a": overall_a,
        "overall_b": overall_b,
        "disagreements": disagreements,
        "judge_count": len(EVAL_JUDGE_MODELS),
    }


class EvalScenario(BaseModel):
    id: str                        # "A" | "B"
    name: str
    model: str                     # "sora2" | "kling26"
    resolution: str = "720p"       # 480p | 720p | 1080p | 4k
    fps: int = 20                  # 16 | 20 | 24
    clip_length_sec: int = 8
    ttft_threshold_sec: int = 30
    quality_buckets: dict = {}


class GenerateEvalRequest(BaseModel):
    use_case: str
    scenarios: list[EvalScenario]  # exactly 2


@app.post("/generate-eval")
def generate_eval(request: GenerateEvalRequest):
    """
    Generate 2 scenario videos, measure generation time, run 3-model judge panel.
    Returns combined results with cost breakdown and quality scores.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not set")
    if len(request.scenarios) != 2:
        raise HTTPException(status_code=400, detail="Exactly 2 scenarios required")

    run_id = str(uuid.uuid4())[:8]
    scenario_results = []

    # Map model IDs to generation functions
    def model_to_gen(s: EvalScenario) -> tuple[str, str]:
        """Returns (gen_model_key, quality_str) for existing generators."""
        if s.model == "sora2":
            q = "1080p" if s.resolution in ("1080p", "4k") else "720p"
            return "sora", q
        return "kling", s.resolution

    # Generate both scenarios concurrently
    prompts = {}
    for s in request.scenarios:
        prompts[s.id] = build_gen_prompt(request.use_case, s.quality_buckets)

    gen_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futures = {}
        for s in request.scenarios:
            gen_key, quality = model_to_gen(s)
            fut = ex.submit(_generate_one, gen_key, prompts[s.id], f"{run_id}_{s.id}", quality, "16:9")
            futures[fut] = s
        for fut in concurrent.futures.as_completed(futures):
            s = futures[fut]
            try:
                _, path, gen_time = fut.result()
                gen_results[s.id] = {"path": path, "time": gen_time, "error": None}
            except Exception as e:
                gen_results[s.id] = {"path": None, "time": None, "error": str(e)}

    # Build scenario result objects
    completed_paths = {}
    for s in request.scenarios:
        gr = gen_results.get(s.id, {})
        cost_per_gen = (
            s.clip_length_sec
            * RES_MULTIPLIER.get(s.resolution, 1.0)
            * FPS_MULTIPLIER.get(s.fps, 1.0)
            * GEN_PRICING.get(s.model, {"per_second": 0.10})["per_second"]
        )
        gen_time = gr.get("time")
        result = {
            "id": s.id,
            "name": s.name,
            "model": s.model,
            "resolution": s.resolution,
            "fps": s.fps,
            "clip_length_sec": s.clip_length_sec,
            "ttft_threshold_sec": s.ttft_threshold_sec,
            "generation_time_sec": round(gen_time, 2) if gen_time else None,
            "ttft_exceeded": (gen_time or 0) > s.ttft_threshold_sec,
            "video_path": gr["path"].name if gr.get("path") else None,
            "cost_per_generation": round(cost_per_gen, 4),
            "cost_at_10": round(cost_per_gen * 10, 2),
            "cost_at_100": round(cost_per_gen * 100, 2),
            "cost_at_1000": round(cost_per_gen * 1000, 2),
            "prompt_used": prompts[s.id],
            "quality_buckets": s.quality_buckets,
            "judge_scores": None,
            "status": "error" if gr.get("error") else "generated",
            "error": gr.get("error"),
        }
        scenario_results.append(result)
        if gr.get("path"):
            completed_paths[s.id] = gr["path"]

    # Run judge if both scenarios generated successfully
    judge_panel = None
    if len(completed_paths) == 2:
        ids = [s.id for s in request.scenarios]
        try:
            path_a = completed_paths[ids[0]]
            path_b = completed_paths[ids[1]]
            judge_panel = run_eval_judge_panel(path_a, path_b)
            # Attach scores per scenario
            for res in scenario_results:
                if res["id"] == ids[0] and "scenario_a" in judge_panel:
                    res["judge_scores"] = judge_panel["scenario_a"]
                    res["judge_overall"] = judge_panel.get("overall_a")
                    res["status"] = "complete"
                elif res["id"] == ids[1] and "scenario_b" in judge_panel:
                    res["judge_scores"] = judge_panel["scenario_b"]
                    res["judge_overall"] = judge_panel.get("overall_b")
                    res["status"] = "complete"
        except Exception as e:
            judge_panel = {"error": str(e)}

    return {
        "run_id": run_id,
        "use_case": request.use_case,
        "scenarios": scenario_results,
        "judge_panel": judge_panel,
    }


# ── Video Gen sensitivity endpoints ──────────────────────────────

VIDEO_CACHE_JSON = PROJECT_ROOT / "video_generation" / "data" / "video_cache.json"
VIDEOGEN_JUDGE_PROMPT = """You are evaluating AI-generated video quality for a creative agency use case.
You will see frames sampled from the video.

Score on 5 dimensions (0-10 each):
1. SUBJECT_QUALITY: Is the main subject clear, well-rendered, and visually compelling?
2. TEMPORAL_COHERENCE: Do the frames tell a coherent visual story/sequence?
3. MOTION_QUALITY: Does movement look natural and fluid?
4. VISUAL_FIDELITY: Sharpness, color accuracy, no artifacts?
5. SCENE_STABILITY: Consistent lighting, composition, no flickering?

Return ONLY this JSON:
{"subject_quality": 0, "temporal_coherence": 0, "motion_quality": 0, "visual_fidelity": 0, "scene_stability": 0, "overall": 0, "one_line_summary": "one sentence"}
overall = round(subject_quality*0.30 + temporal_coherence*0.25 + motion_quality*0.20 + visual_fidelity*0.15 + scene_stability*0.10, 1)"""


def score_video_path(video_path: Path) -> dict:
    """Extract frames from a video file and score via LLM judge (via OpenRouter)."""
    try:
        temp_dir, frame_paths = _extract_frames(video_path)
    except Exception as e:
        return {"subject_quality": 5, "temporal_coherence": 5, "motion_quality": 5,
                "visual_fidelity": 5, "scene_stability": 5, "overall": 5.0,
                "one_line_summary": f"Scoring unavailable: {e}"}

    try:
        urls = _frame_paths_to_data_urls(frame_paths)
        content = [{"type": "text", "text": VIDEOGEN_JUDGE_PROMPT}]
        for url in urls:
            content.append({"type": "image_url", "image_url": {"url": url}})

        import httpx as _httpx
        r = _httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json",
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
        pass
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {"subject_quality": 5, "temporal_coherence": 5, "motion_quality": 5,
            "visual_fidelity": 5, "scene_stability": 5, "overall": 5.0,
            "one_line_summary": "Scoring unavailable"}


@app.get("/videogen/load-cache")
def videogen_load_cache():
    """Return pre-scored video cache data (no API calls). Requires setup_cache.py to have been run."""
    if not VIDEO_CACHE_JSON.exists():
        return {"error": "Cache not found. Run: python video_generation/setup_cache.py",
                "available": False}
    try:
        data = json.loads(VIDEO_CACHE_JSON.read_text())
        return {"available": True, "data": data, "count": len(data)}
    except Exception as e:
        return {"error": str(e), "available": False}


class VideoGenComparisonRequest(BaseModel):
    prompt: str | None = None
    clip_sec: int = 10
    resolution: str = "1080p"       # 720p | 1080p | 4K
    fps: int = 24
    generations_per_month: int = 200
    subject_quality_min: float = 7.0
    temporal_coherence_min: float = 6.0
    latency_threshold_sec: int = 60


@app.post("/videogen/generate-comparison")
def videogen_generate_comparison(req: VideoGenComparisonRequest):
    """Generate Sora 2 + Kling 2.6 with the same prompt, score both, return comparison."""
    prompt = req.prompt or (
        "Cinematic brand video. Dynamic product launch, modern sleek aesthetic, "
        "professional studio quality, smooth camera movement."
    )
    run_id = str(uuid.uuid4())[:8]
    quality = "1080p" if req.resolution in ("1080p", "4K") else "720p"

    RES_MULT = {"720p": 1.0, "1080p": 2.0, "4K": 4.0}
    mult = RES_MULT.get(req.resolution, 2.0)
    sora_cpg  = round(req.clip_sec * 0.15 * mult, 2)
    kling_cpg = round(req.clip_sec * 0.08 * mult, 2)

    # Generate both concurrently
    sora_result  = {"path": None, "gen_time": None, "error": None}
    kling_result = {"path": None, "gen_time": None, "error": None}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut_sora  = ex.submit(_generate_one, "sora",  prompt, f"{run_id}_sora",  quality, "16:9")
        fut_kling = ex.submit(_generate_one, "kling", prompt, f"{run_id}_kling", quality, "16:9")

        for fut, key in [(fut_sora, "sora"), (fut_kling, "kling")]:
            target = sora_result if key == "sora" else kling_result
            try:
                _, path, gen_time = fut.result()
                target["path"] = path
                target["gen_time"] = gen_time
            except Exception as e:
                target["error"] = str(e)

    def build_model_result(result: dict, cpg: float, model_name: str) -> dict:
        if result["error"] or not result["path"]:
            return {
                "status": "error", "error": result["error"] or "Generation failed",
                "cost_per_gen": cpg,
                "monthly_cost": round(cpg * req.generations_per_month),
                "annual_cost":  round(cpg * req.generations_per_month * 12),
                "gen_time_sec": None, "scores": None,
                "meets_quality_bar": False, "latency_ok": False,
                "video_b64": None,
            }
        path: Path = result["path"]
        gen_time = result["gen_time"]
        scores = score_video_path(path)
        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        meets = (scores.get("subject_quality", 0) >= req.subject_quality_min and
                 scores.get("temporal_coherence", 0) >= req.temporal_coherence_min)
        return {
            "status": "complete",
            "gen_time_sec": round(gen_time, 1),
            "cost_per_gen": cpg,
            "monthly_cost": round(cpg * req.generations_per_month),
            "annual_cost":  round(cpg * req.generations_per_month * 12),
            "scores": scores,
            "meets_quality_bar": meets,
            "latency_ok": gen_time <= req.latency_threshold_sec,
            "video_b64": video_b64,
            "error": None,
        }

    sora_out  = build_model_result(sora_result,  sora_cpg,  "Sora 2")
    kling_out = build_model_result(kling_result, kling_cpg, "Kling 2.6")

    # Recommendation
    sm = sora_out["meets_quality_bar"]
    km = kling_out["meets_quality_bar"]
    if sm and km:
        rec = "kling"
        ratio = round((sora_cpg / max(kling_cpg, 0.01) - 1) * 100)
        reason = f"Both models meet your quality bar. Kling 2.6 is {ratio}% cheaper."
    elif sm and not km:
        rec = "sora"
        reason = "Only Sora 2 meets your quality thresholds. Quality justifies the premium."
    elif km and not sm:
        rec = "kling"
        reason = "Kling 2.6 meets your quality bar at lower cost."
    else:
        rec = "neither"
        reason = "Neither model meets your quality bar. Consider lowering thresholds or using higher resolution."

    return {
        "run_id": run_id,
        "prompt_used": prompt,
        "config": {"clip_sec": req.clip_sec, "resolution": req.resolution,
                   "fps": req.fps, "generations_per_month": req.generations_per_month},
        "sora":  sora_out,
        "kling": kling_out,
        "recommendation": rec,
        "recommendation_reason": reason,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


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
