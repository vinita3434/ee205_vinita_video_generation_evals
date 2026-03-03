"""
main.py — Orchestrator for the video analysis evaluation pipeline.

Usage:
    python main.py                    # sanity check: 2 clips × 3 models (no judge)
    python main.py --full             # full run: all clips × 3 models (no judge)
    python main.py --full --judge     # full run with LLM judge (~$0.005 extra per run)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow

# Ensure config and src are importable
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CLIPS_DIR,
    GROUND_TRUTH_FILE,
    MLFLOW_TRACKING_URI,
    MODELS,
)
from src.extract_frames import extract_frames
from src.log_results import log_results
from src.run_models import run_model
from src.score import score_output


def load_ground_truth() -> list[dict]:
    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_FILE}")
    return json.loads(GROUND_TRUTH_FILE.read_text())


def run_pipeline(sanity_check: bool = True, use_llm_judge: bool = False) -> None:
    clips = load_ground_truth()
    if sanity_check:
        clips = clips[:2]
        print(f"[pipeline] Sanity check mode — running {len(clips)} clip(s) × {len(MODELS)} model(s)")
    else:
        print(f"[pipeline] Full run — {len(clips)} clip(s) × {len(MODELS)} model(s)")

    if use_llm_judge:
        print("[pipeline] LLM judge ENABLED (~$0.005 per run)")
    else:
        print("[pipeline] LLM judge DISABLED (use --judge to enable)")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("video_eval_v1")

    total = len(clips) * len(MODELS)
    completed = 0
    errors = 0

    for clip in clips:
        clip_id   = clip["clip_id"]
        video_file = CLIPS_DIR / clip["video_file"]
        ground_truth = clip["ground_truth"]

        print(f"\n── Clip: {clip_id} ({clip.get('category', '?')}) ──")

        # Extract frames once, share across models
        try:
            frame_paths = extract_frames(str(video_file), clip_id)
            print(f"  Extracted {len(frame_paths)} frames")
        except Exception as e:
            print(f"  [SKIP] Frame extraction failed: {e}")
            errors += len(MODELS)
            continue

        for model_cfg in MODELS:
            model_name = model_cfg["name"]
            print(f"  → {model_name} ({model_cfg['type']}) ...", end=" ", flush=True)

            with mlflow.start_run(run_name=f"{clip_id}__{model_cfg['id'].split('/')[-1]}"):
                result = run_model(clip_id, frame_paths, model_cfg)

                if result["error"]:
                    print(f"ERROR: {result['error']}")
                    result["ground_truth"] = ground_truth
                    log_results(result)
                    errors += 1
                    completed += 1
                    continue

                scores = score_output(
                    result["output_text"],
                    ground_truth,
                    use_llm_judge=use_llm_judge,
                )
                result.update(scores)
                result["ground_truth"] = ground_truth

                log_results(result)
                completed += 1

                cost_str = f"${result['cost_usd']:.4f}" if result["cost_usd"] else "?"
                rouge_str = f"{result['rouge_l']:.3f}" if result["rouge_l"] else "?"
                bert_str  = f"{result['bert_score_f1']:.3f}" if result["bert_score_f1"] else "?"
                judge_str = str(result["llm_judge_score"]) if result["llm_judge_score"] else "—"
                print(
                    f"OK  cost={cost_str}  ROUGE-L={rouge_str}  "
                    f"BERTScore={bert_str}  judge={judge_str}/5  "
                    f"latency={result['latency_ms']}ms"
                )

    print(f"\n{'='*60}")
    print(f"Done: {completed}/{total} runs completed, {errors} error(s)")
    print(f"Results → results/runs.json")
    print(f"MLflow  → run: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"          then open http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Analysis Eval Pipeline")
    parser.add_argument("--full",  action="store_true", help="Run all clips (default: sanity check, 2 clips)")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge scoring (+$0.005/run)")
    args = parser.parse_args()

    run_pipeline(
        sanity_check=not args.full,
        use_llm_judge=args.judge,
    )
