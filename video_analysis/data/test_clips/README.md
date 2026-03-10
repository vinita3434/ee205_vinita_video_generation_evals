# Test Clips

Place `.mp4` video files in this directory.

Each filename will appear as a selectable option in the **Benchmark Panel** dropdown on the frontend.

**Naming convention:** Use descriptive snake_case or hyphen-separated names — spaces and underscores are converted to title-case labels in the UI.

```
data/test_clips/
  sports_basketball_play.mp4     →  "Sports Basketball Play"
  surveillance_parking_lot.mp4   →  "Surveillance Parking Lot"
  entertainment_concert.mp4      →  "Entertainment Concert"
```

Files are served directly by their absolute path to the analysis backend — no upload required for benchmark runs.
