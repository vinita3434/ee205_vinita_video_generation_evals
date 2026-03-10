"""
Gruve Atlas — Combined Server
Serves Video Generation at / and Video Analysis at /analysis on one port.

Local dev:   uvicorn combined:app --port 8080 --reload
Production:  uvicorn combined:app --host 0.0.0.0 --port $PORT

Routes:
  /            → Video Generation frontend + API (/generate, /judge, /video/*)
  /analysis    → Video Analysis frontend + API (/analysis/analyze, /analysis/benchmark-results, ...)
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
# Make both backend packages importable
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "video_analysis"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

app = FastAPI(title="Gruve Atlas")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount analysis app at /analysis ──────────────────────────────────────────
# Must be registered BEFORE the root mount so longer prefix wins.
from video_analysis.backend.main import app as _analysis_app   # noqa: E402
app.mount("/analysis", _analysis_app)

# ── Mount video gen app at root ───────────────────────────────────────────────
from backend.main import app as _gen_app                       # noqa: E402
app.mount("/", _gen_app)
