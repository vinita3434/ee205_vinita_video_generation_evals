"""
export_qivd_samples.py — Export the first 5 samples from the already-downloaded
QIVD fiftyone dataset into the Gruve Atlas test_clips/ and ground_truth.json.

Requires the dataset to have been downloaded first:
    python3 video_analysis/scripts/download_qivd.py

Usage (from repo root):
    python3 video_analysis/scripts/export_qivd_samples.py
"""
import json
import re
import shutil
import sys
from pathlib import Path

try:
    import fiftyone as fo
except ImportError:
    print("fiftyone not found — run download_qivd.py first.")
    sys.exit(1)

DATASET_NAME = "qivd-preview"
N_SAMPLES    = 5

REPO_ROOT   = Path(__file__).resolve().parents[2]
TEST_CLIPS  = REPO_ROOT / "video_analysis" / "data" / "test_clips"
GT_FILE     = REPO_ROOT / "video_analysis" / "data" / "ground_truth.json"

TEST_CLIPS.mkdir(parents=True, exist_ok=True)


def _slug(text: str) -> str:
    """Convert a question string into a short snake_case filename stem."""
    text = text.lower().strip().rstrip("?").rstrip(".")
    text = re.sub(r"[^a-z0-9 ]", "", text)
    words = text.split()[:5]            # keep first 5 words
    return "_".join(words)


# ── 1. Load the persistent dataset ───────────────────────────────────────────
if not fo.dataset_exists(DATASET_NAME):
    print(f"Dataset '{DATASET_NAME}' not found — run download_qivd.py first.")
    sys.exit(1)

ds      = fo.load_dataset(DATASET_NAME)
samples = list(ds)[:N_SAMPLES]
print(f"[✓] Loaded '{DATASET_NAME}' — using first {len(samples)} samples")

# ── 2. Export video files ─────────────────────────────────────────────────────
ground_truth = []

for s in samples:
    src       = Path(s.filepath)
    category  = s.category.label.replace(" ", "_") if s.category else "unknown"
    stem      = f"{category}__{_slug(s.question)}"
    dst       = TEST_CLIPS / f"{stem}.mp4"

    shutil.copy2(src, dst)
    print(f"  [copy] {src.name} → {dst.name}")

    # ── 3. Collect all available fields ──────────────────────────────────────
    entry = {
        "clip_id":          stem,
        "filename":         dst.name,
        "source_file":      src.name,
        "category":         s.category.label if s.category else None,
        "question":         s.question,
        "answer":           s.answer,
        "short_answer":     s.short_answer,
        "answer_timestamp": s.answer_timestamp,
    }
    ground_truth.append(entry)

# ── 4. Write ground_truth.json ────────────────────────────────────────────────
GT_FILE.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False))
print(f"\n[✓] ground_truth.json written → {GT_FILE}")
print(f"[✓] {len(ground_truth)} clips ready in {TEST_CLIPS}\n")

for e in ground_truth:
    print(f"  {e['filename']}")
    print(f"    category:    {e['category']}")
    print(f"    question:    {e['question']}")
    print(f"    short_answer:{e['short_answer']}")
    print(f"    timestamp:   {e['answer_timestamp']}")
    print()
