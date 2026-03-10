"""
download_sports_clips.py — Download NBA + NFL highlight clips and update ground_truth.json.

Usage (from repo root):
    python3 video_analysis/scripts/download_sports_clips.py
"""
import json
import subprocess
import sys
from pathlib import Path

# ── Install yt-dlp if missing ──────────────────────────────────────────────
try:
    import yt_dlp
    print("[✓] yt-dlp already installed")
except ImportError:
    print("[…] Installing yt-dlp…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp"])
    import yt_dlp
    print("[✓] yt-dlp installed")

# Locate ffmpeg — prefer imageio-ffmpeg, fall back to ~/bin/ffmpeg symlink or system
import os as _os
_HOME_BIN = str(Path.home() / "bin" / "ffmpeg")
try:
    import imageio_ffmpeg
    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"[✓] ffmpeg (imageio): {FFMPEG_BIN}")
except Exception:
    if Path(_HOME_BIN).exists():
        FFMPEG_BIN = _HOME_BIN
        print(f"[✓] ffmpeg (~bin): {FFMPEG_BIN}")
    else:
        FFMPEG_BIN = "ffmpeg"
        print("[!] ffmpeg: using system fallback")
FFPROBE_BIN = FFMPEG_BIN.replace("ffmpeg", "ffprobe") if "ffmpeg" in FFMPEG_BIN else "ffprobe"

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT      = Path(__file__).resolve().parents[2]
TEST_CLIPS_DIR = REPO_ROOT / "video_analysis" / "data" / "test_clips"
GT_FILE        = REPO_ROOT / "video_analysis" / "data" / "ground_truth.json"
TEST_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# ── Clips to download ──────────────────────────────────────────────────────
CLIPS = [
    {
        "url":      "https://www.youtube.com/watch?v=KQkN5-uwdT0",
        "filename": "nba_highlight.mp4",
        "sport":    "NBA",  # Knicks at Lakers, March 8 2026 — official NBA channel
    },
    {
        "url":      "https://www.youtube.com/watch?v=4zShusD66jI",
        "filename": "nfl_highlight.mp4",
        "sport":    "NFL",  # Patriots vs Broncos AFC Championship 2025 — official NFL channel
    },
]

# ── Download + trim to 60s ────────────────────────────────────────────────
downloaded = {}
for clip in CLIPS:
    out_path = TEST_CLIPS_DIR / clip["filename"]
    tmp_path = TEST_CLIPS_DIR / f"_tmp_{clip['filename']}"

    print(f"\n[…] Downloading {clip['sport']} clip…")
    print(f"    URL:  {clip['url']}")
    print(f"    Dest: {out_path}")

    ydl_opts = {
        "format":  "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "outtmpl": str(tmp_path.with_suffix("")),   # yt-dlp adds extension
        "merge_output_format": "mp4",
        "quiet":   True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clip["url"]])
    except Exception as e:
        print(f"  [!] Download failed: {e}")
        downloaded[clip["filename"]] = False
        continue

    # yt-dlp may have saved as .mp4 or .webm etc — find the file
    candidates = list(TEST_CLIPS_DIR.glob(f"_tmp_{clip['filename'].replace('.mp4','')}*"))
    if not candidates:
        print(f"  [!] Downloaded file not found in {TEST_CLIPS_DIR}")
        downloaded[clip["filename"]] = False
        continue

    raw_file = candidates[0]
    print(f"  [✓] Downloaded: {raw_file.name} ({raw_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # Trim to first 60 seconds with ffmpeg
    print(f"  […] Trimming to 60s…")
    result = subprocess.run(
        [FFMPEG_BIN, "-y", "-i", str(raw_file), "-t", "60",
         "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart",
         str(out_path)],
        capture_output=True,
    )
    raw_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  [!] ffmpeg trim failed: {result.stderr.decode()[-200:]}")
        downloaded[clip["filename"]] = False
        continue

    size_mb  = out_path.stat().st_size / 1024 / 1024
    # Get duration via ffprobe
    probe = subprocess.run(
        [FFPROBE_BIN, "-v", "quiet",
         "-print_format", "json", "-show_streams", str(out_path)],
        capture_output=True,
    )
    duration_str = "~60s"
    try:
        import json as _json
        info = _json.loads(probe.stdout)
        dur  = float(info["streams"][0].get("duration", 60))
        duration_str = f"{dur:.1f}s"
    except Exception:
        pass

    print(f"  [✓] Saved: {out_path.name}  |  {size_mb:.1f} MB  |  {duration_str}")
    downloaded[clip["filename"]] = True

# ── Build ground truth entries ────────────────────────────────────────────
TBD = "TBD — watch the clip and fill this in before running benchmark"

NEW_ENTRIES = [
    # ── NBA ──────────────────────────────────────────────────────────────
    {
        "clip_id":         "nba_sport_type",
        "filename":        "nba_highlight.mp4",
        "category":        "sports",
        "domain":          "sports_archive",
        "question":        "What sport is being played?",
        "answer":          "basketball",
        "short_answer":    "basketball",
        "answer_timestamp": "0",
    },
    {
        "clip_id":         "nba_jersey_color",
        "filename":        "nba_highlight.mp4",
        "category":        "object_attributes",
        "domain":          "sports_archive",
        "question":        "What color jersey is the scoring player wearing?",
        "answer":          TBD,
        "short_answer":    TBD,
        "answer_timestamp": "5",
    },
    {
        "clip_id":         "nba_player_count",
        "filename":        "nba_highlight.mp4",
        "category":        "counting",
        "domain":          "sports_archive",
        "question":        "How many players are visible on the court?",
        "answer":          TBD,
        "short_answer":    TBD,
        "answer_timestamp": "3",
    },
    # ── NFL ──────────────────────────────────────────────────────────────
    {
        "clip_id":         "nfl_sport_type",
        "filename":        "nfl_highlight.mp4",
        "category":        "sports",
        "domain":          "sports_archive",
        "question":        "What sport is being played?",
        "answer":          "football",
        "short_answer":    "football",
        "answer_timestamp": "0",
    },
    {
        "clip_id":         "nfl_jersey_color",
        "filename":        "nfl_highlight.mp4",
        "category":        "object_attributes",
        "domain":          "sports_archive",
        "question":        "What color jersey is the team with the ball wearing?",
        "answer":          TBD,
        "short_answer":    TBD,
        "answer_timestamp": "5",
    },
    {
        "clip_id":         "nfl_player_count",
        "filename":        "nfl_highlight.mp4",
        "category":        "counting",
        "domain":          "sports_archive",
        "question":        "How many players are visible in the frame?",
        "answer":          TBD,
        "short_answer":    TBD,
        "answer_timestamp": "3",
    },
]

# Load existing ground truth and append (deduplicate by clip_id)
existing: list = []
if GT_FILE.exists():
    existing = json.loads(GT_FILE.read_text())

existing_ids = {e.get("clip_id") for e in existing}
added = 0
for entry in NEW_ENTRIES:
    if entry["clip_id"] not in existing_ids:
        existing.append(entry)
        added += 1

GT_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
print(f"\n[✓] ground_truth.json updated — added {added} new entries ({len(existing)} total)")

# ── Print warnings for TBD entries ───────────────────────────────────────
tbd_entries = [e for e in NEW_ENTRIES if TBD in str(e.get("answer", ""))]
if tbd_entries:
    print("\n" + "═" * 60)
    print("⚠  MANUAL ACTION REQUIRED — fill in ground truth answers:")
    print("═" * 60)
    for e in tbd_entries:
        print(f"\n  clip_id:  {e['clip_id']}")
        print(f"  file:     {e['filename']}")
        print(f"  question: {e['question']}")
        print(f"  → Edit video_analysis/data/ground_truth.json")
        print(f"    Set \"answer\" and \"short_answer\" for this entry")
    print("\n  Run benchmark ONLY after filling in answers marked TBD.")
    print("  Skip sports clips in run_benchmark.py until then.")
    print("═" * 60)

# ── Final summary ─────────────────────────────────────────────────────────
print("\n── Download summary ─────────────────────────────────────────")
for clip in CLIPS:
    status = "✓" if downloaded.get(clip["filename"]) else "✗ FAILED"
    print(f"  [{status}] {clip['filename']}")
