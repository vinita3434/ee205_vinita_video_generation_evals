# Video Gen Evals

One prompt → **2 videos** (Kling 2.6, OpenAI Sora 2) → **Judge** with Gemini via OpenRouter (Temporal Stability & Physics Accuracy, 1–10).

## What you need

### API keys

| Key | Used for | Where to get it |
|-----|----------|------------------|
| `OPENAI_API_KEY` | Sora 2 (LiteLLM) | [OpenAI API](https://platform.openai.com/api-keys) |
| `FAL_KEY` | Kling 2.6 Pro | [fal.ai Dashboard](https://fal.ai/dashboard/keys) |
| `OPENROUTER_API_KEY` | Judge (Gemini) | [OpenRouter](https://openrouter.ai/keys) — *you have credits here* |

### Setup

1. Clone and enter the repo:
   ```bash
   cd ee205_vinita_video_generation_evals
   ```

2. Create a virtualenv and install deps:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

3. Copy env and add your keys:
   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY, FAL_KEY, OPENROUTER_API_KEY
   ```

## Run the platform

1. From repo root, start the server (backend + UI at `/`):
   ```bash
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   npm run dev:all
   ```
   Or without npm: `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`

2. Open **http://localhost:8000** in your browser.

3. Enter a **prompt** → click **GENERATE 2 VIDEOS** → when both are ready, click **RUN JUDGE (GEMINI)** to get ratings (Temporal Stability & Physics Accuracy, 1–10).

## Judge script (CLI)

Without the UI, you can run the judge on existing videos:

```bash
python scripts/judge_videos.py "Your prompt" outputs/runid_kling.mp4 outputs/runid_sora.mp4
```

Uses `OPENROUTER_API_KEY` and optional `JUDGE_MODEL` from `.env`.

## Project layout

- `backend/` — FastAPI app: `/generate`, `/judge`, `/video/{filename}`
- `frontend/index.html` — Single-page UI (prompt, 3 video slots, judge button)
- `scripts/judge_videos.py` — Standalone judge script
- `outputs/` — Generated videos (created automatically)
