import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env from repo root (two levels up from this file)
load_dotenv(Path(__file__).parents[2] / ".env")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MODEL_IDS = {
    "gpt4o": "openai/gpt-4o",
    "gemini": "google/gemini-2.0-flash-001",
    "qwen": "qwen/qwen2.5-vl-72b-instruct",
}

PRICING = {
    "gpt4o":  {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
    "gemini": {"input": 0.10 / 1e6, "output":  0.40 / 1e6},
    "qwen":   {"input": 0.59 / 1e6, "output":  0.59 / 1e6},
}

RESULTS_FILE = Path(__file__).parents[1] / "data" / "results" / "benchmark_results.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)


class ScenarioRequest(BaseModel):
    id: str
    name: str
    model: str
    clip_length_sec: float
    daily_volume: int
    concurrency: int
    assembled_prompt: str
    use_case_params: dict = {}


class BenchmarkRequest(BaseModel):
    consumer_type: str
    use_case: str
    scenarios: list[ScenarioRequest]


async def run_scenario(scenario: ScenarioRequest) -> dict:
    model_key = scenario.model
    if model_key not in MODEL_IDS:
        return {"id": scenario.id, "name": scenario.name, "model": model_key,
                "status": "error", "error": f"Unknown model: {model_key}"}

    if not OPENROUTER_API_KEY:
        return {"id": scenario.id, "name": scenario.name, "model": model_key,
                "status": "error", "error": "OPENROUTER_API_KEY not configured"}

    model_id = MODEL_IDS[model_key]
    pricing = PRICING[model_key]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Gruve Atlas",
    }
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": scenario.assembled_prompt}],
        "max_tokens": 300,
    }

    try:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers=headers,
                json=payload,
            )
        ttft_ms = round((time.perf_counter() - start) * 1000)

        if resp.status_code != 200:
            return {"id": scenario.id, "name": scenario.name, "model": model_key,
                    "status": "error", "error": f"API {resp.status_code}: {resp.text[:200]}"}

        data = resp.json()
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})

        input_tokens = usage.get("prompt_tokens", int(scenario.clip_length_sec * 258))
        output_tokens = usage.get("completion_tokens", max(1, len(content.split())))
        total_time_s = max(ttft_ms / 1000, 0.001)
        tps = round(output_tokens / total_time_s, 1)

        cost_per_query = input_tokens * pricing["input"] + output_tokens * pricing["output"]
        qacs = round(min(100, tps / (cost_per_query * 10)), 1) if cost_per_query > 0 else 0
        monthly_cost = round(cost_per_query * scenario.daily_volume * 30, 2)

        return {
            "id": scenario.id,
            "name": scenario.name,
            "model": model_key,
            "ttft_ms": ttft_ms,
            "tps": tps,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_per_query": round(cost_per_query, 6),
            "monthly_cost": monthly_cost,
            "qacs": qacs,
            "status": "complete",
            "response_preview": content[:200],
        }
    except Exception as e:
        return {"id": scenario.id, "name": scenario.name, "model": model_key,
                "status": "error", "error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok", "port": 8002}


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest):
    results = await asyncio.gather(*[run_scenario(s) for s in req.scenarios])

    run = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "consumer_type": req.consumer_type,
        "use_case": req.use_case,
        "scenarios": list(results),
    }

    all_results = []
    if RESULTS_FILE.exists():
        try:
            all_results = json.loads(RESULTS_FILE.read_text())
        except Exception:
            all_results = []
    all_results.append(run)
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2))
    return run


@app.get("/results")
async def get_results():
    if not RESULTS_FILE.exists():
        return []
    try:
        return json.loads(RESULTS_FILE.read_text())
    except Exception:
        return []
