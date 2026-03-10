"""
config.py — Gruve Atlas video analysis config.
Loads only what the analysis service needs.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Walk up to find .env (repo root is two levels up from video_analysis/backend/)
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env")


def _k(name: str) -> str | None:
    v = os.getenv(name)
    if not isinstance(v, str):
        return None
    cleaned = v.strip().replace("\r", "").replace("\n", "").strip()
    return cleaned if cleaned else None


OPENROUTER_API_KEY: str | None = _k("OPENROUTER_API_KEY")

# Runtime data dirs (relative to this file's package, not cwd)
_PKG_ROOT = Path(__file__).resolve().parent.parent  # video_analysis/
DATA_DIR       = _PKG_ROOT / "data"
TEST_CLIPS_DIR = DATA_DIR / "test_clips"
FRAMES_DIR     = DATA_DIR / "frames"

for _d in (TEST_CLIPS_DIR, FRAMES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
