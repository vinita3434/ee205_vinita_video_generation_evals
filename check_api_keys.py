#!/usr/bin/env python3
"""
Temporary script to verify API keys. Run from project root:
  python check_api_keys.py
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Load .env from project root
_root = Path(__file__).resolve().parent
_env = _root / ".env"
if _env.exists():
    from dotenv import load_dotenv
    load_dotenv(_env)
    load_dotenv()
else:
    print("No .env found at", _env)
    sys.exit(1)


def _k(name: str) -> Optional[str]:
    v = os.getenv(name)
    if not isinstance(v, str):
        return None
    # Strip whitespace, newlines, and surrounding quotes (e.g. from .env: KEY="value")
    cleaned = v.strip().replace("\r", "").replace("\n", "").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in '"\'':
        cleaned = cleaned[1:-1].strip()
    return cleaned if cleaned else None


OPENAI_API_KEY = _k("OPENAI_API_KEY")
FAL_KEY = _k("FAL_KEY")
OPENROUTER_API_KEY = _k("OPENROUTER_API_KEY")


def check_openai():
    """Verify OpenAI API key with a minimal request."""
    if not OPENAI_API_KEY:
        return "OPENAI_API_KEY not set in .env"
    try:
        import httpx
        r = httpx.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=10,
        )
        if r.status_code == 200:
            return "OK (key valid)"
        if r.status_code == 401:
            return f"FAIL: 401 - {r.text[:300]}"
        return f"FAIL: {r.status_code} - {r.text[:200]}"
    except Exception as e:
        return f"FAIL: {e}"


def check_fal():
    """Verify FAL key - just try SyncClient and a very cheap model run."""
    if not FAL_KEY:
        return "FAL_KEY not set in .env"
    try:
        from fal_client import SyncClient
        client = SyncClient(key=FAL_KEY)
        # Minimal image gen (fast and cheap) to verify key
        result = client.run("fal-ai/fast-lightning-sdxl", arguments={"prompt": "a red dot", "image_size": "square_hd"})
        if result and (isinstance(result, dict) and "images" in result or hasattr(result, "images")):
            return "OK (key valid)"
        return "OK (key valid)"
    except Exception as e:
        err = str(e).lower()
        if "auth" in err or "401" in err or "invalid" in err or "required" in err:
            return f"FAIL: Invalid API key - {e}"
        return f"FAIL: {e}"


def check_openrouter():
    """Verify OpenRouter API key with a minimal completion."""
    if not OPENROUTER_API_KEY:
        return "OPENROUTER_API_KEY not set in .env"
    try:
        import httpx
        r = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("JUDGE_MODEL", "google/gemini-2.5-flash"),
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 5,
            },
            timeout=15,
        )
        if r.status_code == 200:
            return "OK (key valid)"
        if r.status_code == 401:
            return f"FAIL: 401 - {r.text[:300]}"
        return f"FAIL: {r.status_code} - {r.text[:200]}"
    except Exception as e:
        return f"FAIL: {e}"


def _mask(k: Optional[str]) -> str:
    if not k:
        return "(not set)"
    if len(k) <= 8:
        return "(too short?)"
    return f"{k[:4]}...{k[-4:]} (len={len(k)})"


def main():
    print("Checking API keys (using .env from project root)\n")
    print("Key load check:")
    print("  OPENAI_API_KEY:", _mask(OPENAI_API_KEY))
    print("  FAL_KEY:      ", _mask(FAL_KEY))
    print("  OPENROUTER:   ", _mask(OPENROUTER_API_KEY))
    print()
    print("API check:")
    print("  OPENAI_API_KEY (Sora):", check_openai())
    print("  FAL_KEY (Kling):      ", check_fal())
    print("  OPENROUTER_API_KEY:  ", check_openrouter())
    print("\nDone.")


if __name__ == "__main__":
    main()
