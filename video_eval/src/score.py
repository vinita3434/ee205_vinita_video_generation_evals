"""
score.py — ROUGE-L, BERTScore, and LLM-as-judge scoring.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JUDGE_PROMPT, OPENROUTER_API_KEY


def score_output(
    output_text: str,
    ground_truth: str,
    use_llm_judge: bool = True,
) -> dict:
    """Compute all scoring metrics for one (output, ground_truth) pair.

    Returns:
        {
            rouge_l: float (0-1),
            bert_score_f1: float (0-1),
            llm_judge_score: int | None (1-5),
            llm_judge_reasoning: str | None,
        }
    """
    rouge_l       = _rouge_l(output_text, ground_truth)
    bert_f1       = _bert_score(output_text, ground_truth)
    judge_score   = None
    judge_reason  = None

    if use_llm_judge:
        judge_score, judge_reason = _llm_judge(output_text, ground_truth)

    return {
        "rouge_l":              rouge_l,
        "bert_score_f1":        bert_f1,
        "llm_judge_score":      judge_score,
        "llm_judge_reasoning":  judge_reason,
    }


# ---------------------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------------------

def _rouge_l(output_text: str, ground_truth: str) -> Optional[float]:
    try:
        import evaluate
        rouge = evaluate.load("rouge")
        result = rouge.compute(
            predictions=[output_text],
            references=[ground_truth],
        )
        return round(float(result["rougeL"]), 4)
    except Exception as e:
        print(f"[score] ROUGE-L failed: {e}")
        return None


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def _bert_score(output_text: str, ground_truth: str) -> Optional[float]:
    try:
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn(
            [output_text],
            [ground_truth],
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        return round(float(F1[0]), 4)
    except Exception as e:
        print(f"[score] BERTScore failed: {e}")
        return None


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------

def _llm_judge(
    output_text: str,
    ground_truth: str,
) -> tuple[Optional[int], Optional[str]]:
    try:
        import litellm
        litellm.suppress_debug_info = True

        prompt = JUDGE_PROMPT.format(
            ground_truth=ground_truth,
            model_output=output_text,
        )
        response = litellm.completion(
            model="openrouter/openai/gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            api_base="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            max_tokens=100,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Parse JSON — handle markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        score = int(parsed["score"])
        reasoning = str(parsed.get("reasoning", ""))
        return score, reasoning
    except Exception as e:
        print(f"[score] LLM judge failed: {e}")
        return None, None


if __name__ == "__main__":
    # Quick smoke test on hardcoded strings
    out = "LeBron drives baseline, passes to Davis who dunks. Lakers lead by 4."
    gt  = "LeBron James receives a pass, drives baseline past his defender, and finds Anthony Davis for a powerful dunk."
    result = score_output(out, gt, use_llm_judge=False)
    print("Scores (no judge):", result)
