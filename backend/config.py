import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _k(name: str) -> Optional[str]:
    """Load and clean an env var: strip whitespace and remove any stray newlines/carriage returns."""
    v = os.getenv(name)
    if not isinstance(v, str):
        return None
    # Strip and remove any newlines/carriage returns (common when pasting keys)
    cleaned = v.strip().replace("\r", "").replace("\n", "").strip()
    return cleaned if cleaned else None


# Video generation API keys
OPENAI_API_KEY = _k("OPENAI_API_KEY")
FAL_KEY = _k("FAL_KEY")
NOVITA_API_KEY = _k("NOVITA_API_KEY")
# Set in env so libraries that read from env get the clean key
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if FAL_KEY:
    os.environ["FAL_KEY"] = FAL_KEY

# Judge (OpenRouter → Gemini)
OPENROUTER_API_KEY = _k("OPENROUTER_API_KEY")
JUDGE_MODEL = (os.getenv("JUDGE_MODEL") or "google/gemini-2.5-flash").strip()

# Output directory for generated videos (absolute so it works from any cwd)
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Published per-second generation costs (USD) — update if pricing changes
# Kling 2.6 Pro via fal.ai: $0.07/s without audio (fal.ai/models/fal-ai/kling-video/v2.6/pro/text-to-video)
# Sora 2 via OpenAI: $0.10/s at 720p (openai.com/api/pricing)
KLING_COST_PER_S: float = 0.07   # generates 5s → $0.35/run
SORA_COST_PER_S: float = 0.10    # generates 8s → $0.80/run
