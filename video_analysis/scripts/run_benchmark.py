"""
run_benchmark.py — End-to-end benchmark runner for Gruve Atlas.

For each clip in ground_truth.json, calls POST /analyze and saves raw
model responses to data/results/raw_results.json.

Usage (from repo root):
    python3 video_analysis/scripts/run_benchmark.py
"""
import json
import sys
import time
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[2]
DATA_DIR       = REPO_ROOT / "video_analysis" / "data"
TEST_CLIPS_DIR = DATA_DIR / "test_clips"
GT_FILE        = DATA_DIR / "ground_truth.json"
RESULTS_DIR    = DATA_DIR / "results"
RAW_FILE       = RESULTS_DIR / "raw_results.json"

API_BASE       = "http://localhost:8001"
CLOSED_MODEL   = "openrouter/openai/gpt-4o"               # frames @ 1fps + [t=Xs] labels
OPEN_MODEL     = "openrouter/google/gemini-2.0-flash-001"  # full video (native understanding)
MAX_TOKENS      = 150
REQUEST_TIMEOUT = 180  # seconds per clip

# Skip entries whose ground truth is TBD (sports clips not yet filled in)
TBD_MARKER = "TBD"

# ── Load ground truth ─────────────────────────────────────────────────────────
if not GT_FILE.exists():
    print(f"ERROR: {GT_FILE} not found — run export_qivd_samples.py first.")
    sys.exit(1)

ground_truth = json.loads(GT_FILE.read_text())
print(f"[✓] Loaded {len(ground_truth)} clips from ground_truth.json")

# ── Check server is up ────────────────────────────────────────────────────────
try:
    r = requests.get(f"{API_BASE}/vision-models", timeout=5)
    r.raise_for_status()
    print(f"[✓] Server reachable at {API_BASE}")
except Exception:
    print(f"ERROR: Cannot reach {API_BASE} — is the server running?")
    print("      cd video_analysis && uvicorn backend.main:app --port 8001 --reload")
    sys.exit(1)

# ── Run benchmark ─────────────────────────────────────────────────────────────
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
raw_results = []

for idx, item in enumerate(ground_truth, 1):
    clip_path = TEST_CLIPS_DIR / item["filename"]
    answer    = item.get("answer", "") or item.get("short_answer", "")
    is_tbd    = TBD_MARKER in str(answer)

    print(f"\n[{idx}/{len(ground_truth)}] {item['filename']}")
    print(f"  category: {item.get('category', '?')}  |  domain: {item.get('domain', 'qivd')}")
    print(f"  question: {item['question']}")
    print(f"  answer:   {answer[:80]}")

    if is_tbd:
        print(f"  SKIP — ground truth is TBD. Fill in ground_truth.json first.")
        raw_results.append({
            "clip_id":      item["clip_id"],
            "ground_truth": item,
            "error":        "ground truth TBD",
            "results":      [],
        })
        continue

    # Use the question directly — LLM judge evaluates the answer, not timestamp
    prompt = item["question"] + " Answer concisely in 1-2 sentences."

    if not clip_path.exists():
        print(f"  SKIP — file not found: {clip_path}")
        raw_results.append({
            "clip_id":      item["clip_id"],
            "ground_truth": item,
            "error":        f"File not found: {clip_path}",
            "results":      [],
        })
        continue

    payload = {
        "video_path":   str(clip_path),
        "prompt":       prompt,
        "closed_model": CLOSED_MODEL,
        "open_model":   OPEN_MODEL,
        "max_tokens":   MAX_TOKENS,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(f"{API_BASE}/analyze", json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        print(f"  ✓ completed in {elapsed:.0f} ms total (wall time)")
        for r in data.get("results", []):
            name = r.get("model_name", r.get("model_id", "?"))
            lat  = r.get("latency_ms")
            out  = (r.get("output_text") or "")[:120]
            err  = r.get("error")
            if err:
                print(f"    {name}: ERROR — {err}")
            else:
                print(f"    {name} [{lat:.0f}ms]: {out}")
        raw_results.append({
            "clip_id":      item["clip_id"],
            "ground_truth": item,
            "prompt":       prompt,
            **data,
        })
    except requests.exceptions.Timeout:
        print(f"  ERROR — request timed out after {REQUEST_TIMEOUT}s")
        raw_results.append({
            "clip_id":      item["clip_id"],
            "ground_truth": item,
            "error":        "timeout",
            "results":      [],
        })
    except Exception as exc:
        msg = str(exc)
        try:
            msg = resp.json().get("detail", msg)
        except Exception:
            pass
        print(f"  ERROR — {msg}")
        raw_results.append({
            "clip_id":      item["clip_id"],
            "ground_truth": item,
            "error":        msg,
            "results":      [],
        })

# ── Save ──────────────────────────────────────────────────────────────────────
RAW_FILE.write_text(json.dumps(raw_results, indent=2, ensure_ascii=False))
print(f"\n[✓] Raw results saved → {RAW_FILE}")
print(f"    {len([r for r in raw_results if not r.get('error')])} / {len(raw_results)} clips succeeded")
print("\nNext: python3 video_analysis/scripts/score_results.py")
