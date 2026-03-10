"""
score_results.py — Score raw benchmark results using an LLM judge.

Reads:   data/results/raw_results.json
Writes:  data/results/scored_results.json

Scoring (per model output):
  LLM judge (Claude Haiku via OpenRouter) decides CORRECT / INCORRECT.
  CORRECT = 1.0, INCORRECT = 0.0
  Entries with answer == "TBD" are skipped (score = None).

QACS formula (0–100 scale):
  accuracy_pct   = accuracy × 100
  cost_per_query = total_cost_usd / n_clips
  cost_norm      = cost_per_query × 10 000   (normalises to per-10K queries)
  quality_weight = 1.0
  QACS           = min(100, (accuracy_pct × quality_weight) / cost_norm)
  If cost is 0 and accuracy > 0: QACS = 100

Usage (from repo root):
    python3 video_analysis/scripts/score_results.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

# ── Config ─────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

OPENROUTER_API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()
JUDGE_MODEL        = "anthropic/claude-haiku-4-5"
RESULTS_DIR        = REPO_ROOT / "video_analysis" / "data" / "results"
RAW_FILE           = RESULTS_DIR / "raw_results.json"
SCORED_FILE        = RESULTS_DIR / "scored_results.json"

TBD_MARKER = "TBD"


# ── LLM judge ──────────────────────────────────────────────────────────────

def judge_answer(question: str, model_output: str, ground_truth: str) -> float:
    """Return 1.0 (CORRECT) or 0.0 (INCORRECT) via Claude Haiku."""
    if not OPENROUTER_API_KEY:
        print("  [!] OPENROUTER_API_KEY not set — defaulting to 0.0")
        return 0.0

    prompt = (
        f"Is \"{model_output}\" a correct answer to the question \"{question}\" "
        f"where the ground truth answer is \"{ground_truth}\"?\n"
        "Be lenient — partial matches, synonyms, and reasonable paraphrases count as correct.\n"
        "Reply with only: CORRECT or INCORRECT"
    )

    for attempt in range(3):
        try:
            r = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model":       JUDGE_MODEL,
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  10,
                    "temperature": 0,
                },
                timeout=30.0,
            )
            r.raise_for_status()
            text    = r.json()["choices"][0]["message"]["content"].strip().upper()
            verdict = 1.0 if "CORRECT" in text else 0.0
            return verdict
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  [!] Judge call failed: {e} — defaulting to 0.0")
                return 0.0
    return 0.0


# ── QACS ───────────────────────────────────────────────────────────────────

def compute_qacs(accuracy: float, cost_per_query: float, quality_weight: float = 1.0) -> float:
    accuracy_pct = accuracy * 100
    if cost_per_query <= 0:
        return 100.0 if accuracy_pct > 0 else 0.0
    cost_norm = cost_per_query * 10_000
    return min(100.0, (accuracy_pct * quality_weight) / cost_norm)


# ── Load raw results ────────────────────────────────────────────────────────

if not RAW_FILE.exists():
    print(f"ERROR: {RAW_FILE} not found — run run_benchmark.py first.")
    sys.exit(1)

raw_results = json.loads(RAW_FILE.read_text())
print(f"[✓] Loaded {len(raw_results)} raw results")
print(f"[…] Using LLM judge: {JUDGE_MODEL}\n")

# ── Score ───────────────────────────────────────────────────────────────────

model_data: dict[str, list[dict]] = {}

for entry in raw_results:
    clip_id      = entry["clip_id"]
    gt           = entry["ground_truth"]
    question     = gt.get("question", "")
    ground_truth = gt.get("answer", "") or gt.get("short_answer", "")
    is_tbd       = TBD_MARKER in str(ground_truth)

    if entry.get("error") or not entry.get("results"):
        for r in entry.get("results", []):
            name = r.get("model_name") or r.get("model_id", "unknown")
            model_data.setdefault(name, []).append({
                "clip_id":      clip_id,
                "category":     gt.get("category", ""),
                "domain":       gt.get("domain", "qivd"),
                "question":     question,
                "ground_truth": ground_truth,
                "score":        None,
                "verdict":      "ERROR",
                "latency_ms":   None,
                "cost_usd":     None,
                "output_text":  None,
                "error":        entry.get("error", "no results"),
            })
        continue

    for r in entry["results"]:
        name        = r.get("model_name") or r.get("model_id", "unknown")
        output_text = r.get("output_text") or ""
        latency_ms  = r.get("latency_ms")
        cost_usd    = r.get("cost_usd")
        err         = r.get("error")

        if err:
            score, verdict = None, "ERROR"
        elif is_tbd:
            score, verdict = None, "TBD"
            print(f"  SKIP (TBD)  {clip_id[:32]:32s}  {name}")
        else:
            print(f"  Judging    {clip_id[:32]:32s}  {name[:22]:22s}  …", end="", flush=True)
            score   = judge_answer(question, output_text, ground_truth)
            verdict = "CORRECT" if score == 1.0 else "INCORRECT"
            print(f"  {verdict}")

        model_data.setdefault(name, []).append({
            "clip_id":      clip_id,
            "category":     gt.get("category", ""),
            "domain":       gt.get("domain", "qivd"),
            "question":     question,
            "ground_truth": ground_truth,
            "score":        score,
            "verdict":      verdict,
            "latency_ms":   latency_ms,
            "cost_usd":     cost_usd,
            "output_text":  output_text[:300],
            "error":        err,
        })

# ── Aggregate ───────────────────────────────────────────────────────────────

scored_models: dict[str, dict] = {}

for model_name, clips in model_data.items():
    scored_clips = [c for c in clips if c["score"] is not None]
    n_scored     = len(scored_clips)

    accuracy    = (sum(c["score"] for c in scored_clips) / n_scored) if n_scored else 0.0
    latencies   = [c["latency_ms"] for c in clips if c["latency_ms"] is not None]
    costs       = [c["cost_usd"]   for c in clips if c["cost_usd"]   is not None]

    avg_latency    = sum(latencies) / len(latencies) if latencies else 0.0
    total_cost     = sum(costs)
    cost_per_query = total_cost / len(clips) if clips else 0.0
    qacs           = compute_qacs(accuracy, cost_per_query)

    scored_models[model_name] = {
        "accuracy":       round(accuracy, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_usd":   round(cost_per_query, 6),
        "qacs":           round(qacs, 2),
        "n_scored":       n_scored,
        "per_clip":       clips,
    }

    n_correct = sum(1 for c in scored_clips if c["score"] == 1.0)
    print(f"\n  ── {model_name} ──")
    print(f"     accuracy:    {accuracy:.0%}  ({n_correct}/{n_scored} correct)")
    print(f"     avg latency: {avg_latency:.0f} ms")
    print(f"     total cost:  ${total_cost:.6f}")
    print(f"     QACS:        {qacs:.1f}/100")

# ── Save ────────────────────────────────────────────────────────────────────

output = {
    "models":        scored_models,
    "run_timestamp": datetime.now(timezone.utc).isoformat(),
    "n_clips":       len(raw_results),
    "judge_model":   JUDGE_MODEL,
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCORED_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False))
print(f"\n[✓] Scored results saved → {SCORED_FILE}")
print("    Refresh http://localhost:8001 to see updated benchmark panel.")
