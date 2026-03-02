import base64
import concurrent.futures
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


class GenerateResponse(BaseModel):
    run_id: str
    kling_path: str
    sora_path: str
    kling_latency_s: float
    sora_latency_s: float
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
        kling_cost_usd=round(5 * KLING_COST_PER_S, 4),   # 5s clip
        sora_cost_usd=round(8 * SORA_COST_PER_S, 4),     # 8s clip
    )


def _video_to_data_url(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:video/mp4;base64,{b64}"


@app.post("/judge")
def judge(request: JudgeRequest):
    """Send the 2 videos + prompt to OpenRouter (Gemini) for evaluation."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY is not set")

    prompt_text = request.prompt.strip()
    judge_prompt = (
        f"Compare these 2 videos to this prompt: {prompt_text}\n\n"
        "Rate each video from 1-10 on:\n"
        "1. Temporal Stability (no flickering)\n"
        "2. Physics Accuracy\n\n"
        "Label each video as: Video A (Kling 2.6), Video B (OpenAI Sora 2). "
        "Respond with a clear table or structured ratings and a short summary."
    )

    try:
        video_a = _video_to_data_url(str(OUTPUT_DIR / request.kling_path))
        video_b = _video_to_data_url(str(OUTPUT_DIR / request.sora_path))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    content = [
        {"type": "text", "text": judge_prompt},
        {"type": "video_url", "video_url": {"url": video_a}},
        {"type": "video_url", "video_url": {"url": video_b}},
    ]

    import httpx
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
    return {"result": message.get("content", ""), "model": JUDGE_MODEL}


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
