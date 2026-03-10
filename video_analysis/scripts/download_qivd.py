"""
download_qivd.py — Download a sample of the Qualcomm Interactive Video Dataset
via FiftyOne and launch the app for manual clip selection.

Usage:
    python video_analysis/scripts/download_qivd.py
"""
import subprocess
import sys


# ── 1. Ensure fiftyone is installed ──────────────────────────────────────────
# NOTE: fiftyone requires _lzma (py7zr dependency). If the current interpreter
# is a pyenv build missing _lzma, pip-install will succeed but import will fail.
# In that case this script should be run with the system python3:
#   python3 video_analysis/scripts/download_qivd.py
def _ensure_fiftyone():
    try:
        import fiftyone  # noqa: F401
        print("[✓] fiftyone already installed")
    except ImportError:
        print("[…] Installing fiftyone…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fiftyone"])
        print("[✓] fiftyone installed")


_ensure_fiftyone()

import os  # noqa: E402

# Inject ffmpeg (from imageio-ffmpeg) into PATH so FiftyOne's server subprocess finds it
try:
    import imageio_ffmpeg
    _ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"[✓] ffmpeg path injected: {_ffmpeg_dir}")
except Exception as e:
    print(f"[!] Could not inject ffmpeg: {e}")

import fiftyone as fo  # noqa: E402
import fiftyone.utils.huggingface as fouh  # noqa: E402

# ── 2. Download 20 samples from QIVD ─────────────────────────────────────────
DATASET_NAME = "qivd-preview"
MAX_SAMPLES  = 20

print(f"\n[…] Loading {MAX_SAMPLES} samples from Voxel51/qualcomm-interactive-video-dataset …")

# Delete existing dataset with same name so re-runs are clean
if fo.dataset_exists(DATASET_NAME):
    fo.delete_dataset(DATASET_NAME)

dataset = fouh.load_from_hub(
    "Voxel51/qualcomm-interactive-video-dataset",
    name=DATASET_NAME,
    max_samples=MAX_SAMPLES,
    persistent=True,
)
dataset.name = DATASET_NAME

print(f"[✓] Loaded {len(dataset)} samples  (dataset: '{DATASET_NAME}')")

# ── 3. Print all field names from the first sample ───────────────────────────
print("\n── Field names on first sample ──────────────────────────────────────")
sample = dataset.first()
for field_name, value in sample.iter_fields():
    print(f"  {field_name!s:30s}  {type(value).__name__}  =  {repr(value)[:120]}")

print("\n── Dataset-level field schema ───────────────────────────────────────")
for name, field in dataset.get_field_schema().items():
    print(f"  {name!s:30s}  {field}")

# ── 4. Launch the FiftyOne app ────────────────────────────────────────────────
import time
print("\n[…] Launching FiftyOne App — pick your 5 clips, then Ctrl-C to exit.\n")
session = fo.launch_app(dataset, port=5151, auto=False)
print("    → http://localhost:5151")
print("    (server stays alive until you press Ctrl-C)\n")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[✓] Done.")
