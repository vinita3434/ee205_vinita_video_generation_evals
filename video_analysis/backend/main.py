"""
main.py — Gruve Atlas: standalone video analysis FastAPI service.

Run from repo root:
    cd video_analysis
    uvicorn backend.main:app --port 8001 --reload

Endpoints:
    GET  /                    → frontend/index.html
    GET  /vision-models       → available closed + open VLMs
    GET  /test-videos         → clips in data/test_clips/
    POST /analyze             → JSON body  {video_path, prompt, ...}
    POST /analyze/upload      → multipart  {file, prompt, ...}
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.config import DATA_DIR, FRAMES_DIR, OPENROUTER_API_KEY, TEST_CLIPS_DIR
from backend.analyzers.vision import VISION_MODELS, analyze_video

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

_PKG_ROOT    = Path(__file__).resolve().parent.parent  # video_analysis/
FRONTEND_DIR = _PKG_ROOT / "frontend"

app = FastAPI(
    title="Gruve Atlas — Video Analysis",
    description="Benchmark closed vs open-source VLMs on arbitrary video clips.",
)


@app.exception_handler(Exception)
def _catch_all(_request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    import traceback
    traceback.print_exc()
    return JSONResponse(status_code=502, content={"detail": str(exc)})


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if FRONTEND_DIR.is_dir():
    @app.get("/")
    def index():
        f = FRONTEND_DIR / "index.html"
        return FileResponse(f) if f.is_file() else {"message": "Gruve Atlas API", "docs": "/docs"}


# ---------------------------------------------------------------------------
# GET /vision-models
# ---------------------------------------------------------------------------

@app.get("/vision-models")
def list_vision_models():
    """Return available closed + open-source vision models."""
    return VISION_MODELS


# ---------------------------------------------------------------------------
# GET /test-videos
# ---------------------------------------------------------------------------

@app.get("/test-videos")
def list_test_videos():
    """Return .mp4 clips with ground truth QA pairs where available.

    Each item: { id, label, path, domain, ground_truth[] }
    """
    import json as _json
    clips = sorted(TEST_CLIPS_DIR.glob("*.mp4"))

    # Index ground truth by filename
    gt_by_file: dict[str, list] = {}
    gt_file = DATA_DIR / "ground_truth.json"
    if gt_file.is_file():
        for entry in _json.loads(gt_file.read_text()):
            fname = entry.get("filename", "")
            gt_by_file.setdefault(fname, []).append(entry)

    result = []
    for clip in clips:
        stem    = clip.stem
        label   = stem.replace("_", " ").replace("-", " ").title()
        gt_list = gt_by_file.get(clip.name, [])
        domain  = gt_list[0].get("domain", "qivd") if gt_list else "qivd"
        result.append({
            "id":           clip.name,
            "label":        label,
            "path":         str(clip),
            "domain":       domain,
            "ground_truth": gt_list,
        })
    return result


# ---------------------------------------------------------------------------
# POST /analyze  (JSON body — benchmark panel: pre-loaded test clips)
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    video_path:       str
    prompt:           str        = "Describe what is happening in this video. Be specific about actions, objects, and any text visible."
    closed_model:     str        = "openrouter/openai/gpt-4o"
    open_model:       str        = "openrouter/google/gemini-2.0-flash-001"
    max_tokens:       int        = 200
    closed_modality:  str | None = None
    open_modality:    str | None = None


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyze a pre-loaded test clip by its absolute path."""
    _check_api_key()
    video_path = Path(request.video_path)
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {request.video_path}")

    return await _run_analysis(
        video_path       = str(video_path),
        prompt           = request.prompt,
        closed_model     = request.closed_model,
        open_model       = request.open_model,
        max_tokens       = request.max_tokens,
        closed_modality  = request.closed_modality,
        open_modality    = request.open_modality,
    )


# ---------------------------------------------------------------------------
# POST /analyze/upload  (multipart — custom analysis tab)
# ---------------------------------------------------------------------------

@app.post("/analyze/upload")
async def analyze_upload(
    file:         UploadFile = File(...),
    prompt:       str        = Form("Describe what is happening in this video. Be specific about actions, objects, and any text visible."),
    closed_model: str        = Form("openrouter/openai/gpt-4o"),
    open_model:   str        = Form("openrouter/meta-llama/llama-3.2-11b-vision-instruct"),
    max_tokens:   int        = Form(200),
):
    """Accept a video file upload, save temporarily, then analyze."""
    _check_api_key()
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video (video/*).")

    # Write upload to a temp file so vision.py can open it with OpenCV
    suffix     = Path(file.filename or "upload.mp4").suffix or ".mp4"
    upload_id  = str(uuid.uuid4())[:8]
    tmp_dir    = Path(tempfile.gettempdir()) / "gruve_atlas_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path   = tmp_dir / f"{upload_id}{suffix}"

    try:
        with open(tmp_path, "wb") as dst:
            shutil.copyfileobj(file.file, dst)

        result = await _run_analysis(
            video_path   = str(tmp_path),
            prompt       = prompt,
            closed_model = closed_model,
            open_model   = open_model,
            max_tokens   = max_tokens,
        )
        # Swap internal path for a safe display name
        result["video_path"] = file.filename or tmp_path.name
        return result

    finally:
        file.file.close()
        # Best-effort cleanup after response
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Shared analysis runner
# ---------------------------------------------------------------------------

def _check_api_key():
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not set.")


# ---------------------------------------------------------------------------
# POST /append-run  — live benchmark update after each UI run
# ---------------------------------------------------------------------------

class AppendRunRequest(BaseModel):
    clip_id:      str
    filename:     str
    question:     str
    ground_truth: str
    category:     str = ""
    domain:       str = "qivd"
    results:      list[dict]   # [{model_name, model_id, model_type, verdict, latency_ms, cost_usd, output_text, error}]


@app.post("/append-run")
def append_run(req: AppendRunRequest):
    """Append a single-clip live result to scored_results.json and recompute aggregates.

    Called by the frontend after each RUN BENCHMARK + judge cycle.
    Creates scored_results.json if it doesn't exist yet.
    """
    import json as _json
    from datetime import datetime, timezone

    scored_file = DATA_DIR / "results" / "scored_results.json"
    scored_file.parent.mkdir(parents=True, exist_ok=True)

    data = _json.loads(scored_file.read_text()) if scored_file.is_file() else {
        "models": {}, "run_timestamp": None, "n_clips": 0, "judge_model": "anthropic/claude-haiku-4-5"
    }

    data["run_timestamp"] = datetime.now(timezone.utc).isoformat()

    for r in req.results:
        name = r.get("model_name") or r.get("model_id", "unknown")
        if name not in data["models"]:
            data["models"][name] = {
                "accuracy": 0, "avg_latency_ms": 0, "total_cost_usd": 0,
                "avg_cost_usd": 0, "qacs": 0, "n_scored": 0, "per_clip": [],
            }
        m = data["models"][name]

        verdict = r.get("verdict", "UNKNOWN")
        score   = 1.0 if verdict == "CORRECT" else 0.0 if verdict == "INCORRECT" else None

        new_clip = {
            "clip_id":      req.clip_id,
            "category":     req.category,
            "domain":       req.domain,
            "question":     req.question,
            "ground_truth": req.ground_truth,
            "score":        score,
            "verdict":      verdict,
            "latency_ms":   r.get("latency_ms"),
            "cost_usd":     r.get("cost_usd"),
            "output_text":  (r.get("output_text") or "")[:300],
            "error":        r.get("error"),
        }
        # Replace existing clip entry or append
        idx = next((i for i, c in enumerate(m["per_clip"]) if c["clip_id"] == req.clip_id), -1)
        if idx >= 0:
            m["per_clip"][idx] = new_clip
        else:
            m["per_clip"].append(new_clip)

        # Recompute aggregates
        scored = [c for c in m["per_clip"] if c["score"] is not None]
        n      = len(scored)
        acc    = sum(c["score"] for c in scored) / n if n else 0.0
        lats   = [c["latency_ms"] for c in m["per_clip"] if c["latency_ms"] is not None]
        costs  = [c["cost_usd"]   for c in m["per_clip"] if c["cost_usd"]   is not None]
        avg_lat    = sum(lats) / len(lats) if lats else 0.0
        total_cost = sum(costs)
        n_clips    = len(m["per_clip"])
        cpq        = total_cost / n_clips if n_clips else 0.0
        qacs       = min(100.0, (acc * 100) / (cpq * 10_000)) if cpq > 0 else (100.0 if acc > 0 else 0.0)

        m.update({
            "accuracy":       round(acc, 4),
            "avg_latency_ms": round(avg_lat, 1),
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_usd":   round(cpq, 6),
            "qacs":           round(qacs, 2),
            "n_scored":       n,
        })

    data["n_clips"] = max((len(m["per_clip"]) for m in data["models"].values()), default=0)
    scored_file.write_text(_json.dumps(data, indent=2))
    return data


# ---------------------------------------------------------------------------
# GET /benchmark-results
# ---------------------------------------------------------------------------

@app.get("/benchmark-results")
def get_benchmark_results():
    """Return scored benchmark results from data/results/scored_results.json.

    Returns 404 if the file doesn't exist yet (run run_benchmark.py + score_results.py first).
    """
    scored_file = DATA_DIR / "results" / "scored_results.json"
    if not scored_file.is_file():
        raise HTTPException(
            status_code=404,
            detail="No benchmark results yet. Run: python3 video_analysis/scripts/run_benchmark.py && python3 video_analysis/scripts/score_results.py",
        )
    import json
    return json.loads(scored_file.read_text())


# ---------------------------------------------------------------------------
# POST /judge-answer  — real-time LLM judge for Section 2 live results
# ---------------------------------------------------------------------------

class JudgeAnswerRequest(BaseModel):
    question:     str
    model_output: str
    ground_truth: str


@app.post("/judge-answer")
async def judge_answer_endpoint(request: JudgeAnswerRequest):
    """Ask Claude Haiku if model_output correctly answers the question."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not set.")

    prompt = (
        f'Is "{request.model_output}" a correct answer to the question '
        f'"{request.question}" where the ground truth answer is "{request.ground_truth}"?\n'
        "Be lenient — partial matches, synonyms, and reasonable paraphrases count as correct.\n"
        "Reply with only: CORRECT or INCORRECT"
    )

    import httpx as _httpx
    try:
        async with _httpx.AsyncClient() as client:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                json={
                    "model":       "anthropic/claude-haiku-4-5",
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  10,
                    "temperature": 0,
                },
                timeout=30.0,
            )
            r.raise_for_status()
            text    = r.json()["choices"][0]["message"]["content"].strip().upper()
            verdict = "CORRECT" if "CORRECT" in text else "INCORRECT"
    except Exception:
        verdict = "UNKNOWN"

    return {"verdict": verdict}


# ---------------------------------------------------------------------------
# Shared analysis runner
# ---------------------------------------------------------------------------

async def _run_analysis(
    video_path:      str,
    prompt:          str,
    closed_model:    str,
    open_model:      str,
    max_tokens:      int,
    closed_modality: str | None = None,
    open_modality:   str | None = None,
) -> dict:
    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: analyze_video(
                video_path       = video_path,
                prompt           = prompt,
                closed_model_id  = closed_model,
                open_model_id    = open_model,
                max_tokens       = max_tokens,
                closed_modality  = closed_modality,
                open_modality    = open_modality,
            ),
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Analysis failed: {exc}")
