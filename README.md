# Gruve Atlas — Video Model Benchmarking

Two tools in one repo: **Video Generation Evals** (Kling vs Sora) and **Video Analysis Evals** (benchmarking vision models on real video understanding).

## API Keys

Create a `.env` file in the repo root:

```
OPENAI_API_KEY=...       # Sora 2 via LiteLLM
FAL_KEY=...              # Kling 2.6 Pro
OPENROUTER_API_KEY=...   # Judge + video analysis models
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Video Generation Evals

One prompt → 2 videos (Kling 2.6, Sora 2) → judged by Gemini.

```bash
uvicorn backend.main:app --reload --port 8000
```

Open **http://localhost:8000** → enter prompt → Generate → Run Judge.

---

## Video Analysis Evals (Gruve Atlas)

Benchmark vision models (GPT-4o, Gemini, Llama, Qwen) on real video clips using an LLM judge and QACS scoring.

```bash
uvicorn video_analysis.backend.main:app --reload --port 8001
```

Open `video_analysis/frontend/index.html` in your browser.

**What it does:**
- Select a test clip and comparison mode (Frames vs Frames / Native vs Native / Mixed)
- Click **RUN BENCHMARK** to run two models side-by-side
- Section 2: live results with accuracy scores and cost
- Section 3: historical benchmark table (pre-seeded with baseline data)
- Section 4: Decision Engine — adjust sliders to rank models by your priorities

**Test data** (already in repo): 7 clips in `video_analysis/data/test_clips/` (QIVD + NBA/NFL), ground truth in `video_analysis/data/ground_truth.json`.

---

## Project Layout

```
backend/               Video gen FastAPI app (port 8000)
frontend/index.html    Video gen UI
video_analysis/
  backend/             Video analysis FastAPI app (port 8001)
  frontend/            Benchmark dashboard UI
  data/
    test_clips/        7 .mp4 test videos
    ground_truth.json  Questions + expected answers per clip
    results/           Baseline scored_results.json (pre-seeded)
  scripts/             Benchmark runner, LLM scorer, clip downloaders
video_eval/            Alternate eval pipeline (frame-based)
scripts/               Video gen judge CLI
```
