# video_eval — Video Analysis Evaluation Pipeline

Benchmarks open-source vs closed-source vision models on a **play-by-play commentary generation** task. Extracts frames from video clips, sends them to each model with a fixed prompt, and scores outputs using ROUGE-L, BERTScore, and an optional LLM judge.

## Models Evaluated
| Model | Type | Provider |
|-------|------|----------|
| GPT-4o | Closed | OpenRouter |
| Gemini 1.5 Flash | Closed | OpenRouter |
| Llama 3.2 Vision 11B | Open | OpenRouter |

## Setup

```bash
cd video_eval

# Install dependencies (torch ~2GB on first install)
pip install -r requirements.txt

# Add your OpenRouter API key to the parent .env file:
# OPENROUTER_API_KEY=sk-or-...
```

## Adding Video Clips

1. Drop `.mp4` files into `data/clips/`
2. Add an entry to `data/ground_truth.json` for each clip:
```json
{
  "clip_id": "nba_001",
  "source": "youtube_cc",
  "category": "nba",
  "video_file": "nba_001.mp4",
  "duration_sec": 22,
  "ground_truth": "LeBron drives baseline...",
  "verified": true
}
```

Two test clips are pre-configured for the sanity check (downloaded automatically on first run if using the download script).

## Running

```bash
# Sanity check — 2 clips × 3 models, no judge (~$0.05)
python main.py

# Full run — all clips × 3 models, no judge
python main.py --full

# Full run + LLM judge (~$0.65 total for 10 clips)
python main.py --full --judge
```

## Viewing Results

```bash
# Raw JSON
cat results/runs.json | python -m json.tool

# MLflow dashboard
mlflow ui --backend-store-uri ./mlruns
# → http://localhost:5000
```

## Project Structure

```
video_eval/
├── data/
│   ├── clips/              # drop .mp4 files here
│   ├── frames/             # auto-created during extraction
│   └── ground_truth.json   # ground truth captions per clip
├── src/
│   ├── extract_frames.py   # OpenCV frame extraction (5 frames, max 1024px, JPEG q85)
│   ├── run_models.py       # LiteLLM multimodal runner
│   ├── score.py            # ROUGE-L + BERTScore + LLM judge
│   └── log_results.py      # MLflow + runs.json logging
├── results/
│   └── runs.json           # one row per (clip × model)
├── config.py               # all controlled variables
├── main.py                 # orchestrator
└── requirements.txt
```

## Metrics Tracked (per clip × model)
| Metric | Description |
|--------|-------------|
| `latency_ms` | End-to-end response time |
| `ttft_ms` | Time to first token |
| `input_tokens` | Prompt token count |
| `output_tokens` | Response token count |
| `cost_usd` | API cost |
| `rouge_l` | ROUGE-L F1 vs ground truth |
| `bert_score_f1` | BERTScore F1 (distilbert) |
| `llm_judge_score` | GPT-4o judge score (1–5, optional) |
