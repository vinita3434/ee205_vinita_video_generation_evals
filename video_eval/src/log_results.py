"""
log_results.py — Log one (clip × model) result to MLflow and results/runs.json.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MLFLOW_TRACKING_URI, RESULTS_FILE


def log_results(result: dict) -> None:
    """Log a result dict to MLflow and append to results/runs.json.

    Expected keys in result (all optional except clip_id/model_id):
        clip_id, model_id, model_name, model_type, ground_truth, output_text,
        latency_ms, ttft_ms, input_tokens, output_tokens, cost_usd,
        rouge_l, bert_score_f1, llm_judge_score, llm_judge_reasoning, error
    """
    _log_to_mlflow(result)
    _append_to_json(result)


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def _log_to_mlflow(result: dict) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Params — categorical identifiers
        params = {
            "clip_id":    result.get("clip_id", ""),
            "model_id":   result.get("model_id", ""),
            "model_name": result.get("model_name", ""),
            "model_type": result.get("model_type", ""),
        }
        mlflow.log_params(params)

        # Metrics — numeric values only
        metric_keys = [
            "latency_ms", "ttft_ms", "input_tokens", "output_tokens",
            "cost_usd", "rouge_l", "bert_score_f1", "llm_judge_score",
        ]
        metrics = {k: result[k] for k in metric_keys if result.get(k) is not None}
        if metrics:
            mlflow.log_metrics(metrics)

        # Artifacts — text files
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            if result.get("output_text"):
                out_file = tmp_path / "output_text.txt"
                out_file.write_text(result["output_text"], encoding="utf-8")
                mlflow.log_artifact(str(out_file))

            if result.get("ground_truth"):
                gt_file = tmp_path / "ground_truth.txt"
                gt_file.write_text(result["ground_truth"], encoding="utf-8")
                mlflow.log_artifact(str(gt_file))

            if result.get("llm_judge_reasoning"):
                jr_file = tmp_path / "judge_reasoning.txt"
                jr_file.write_text(result["llm_judge_reasoning"], encoding="utf-8")
                mlflow.log_artifact(str(jr_file))

    except Exception as e:
        print(f"[log] MLflow logging failed: {e}")


# ---------------------------------------------------------------------------
# runs.json
# ---------------------------------------------------------------------------

def _append_to_json(result: dict) -> None:
    try:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

        existing: list = []
        if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0:
            try:
                existing = json.loads(RESULTS_FILE.read_text())
            except json.JSONDecodeError:
                existing = []

        existing.append(result)

        RESULTS_FILE.write_text(
            json.dumps(existing, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[log] runs.json write failed: {e}")
