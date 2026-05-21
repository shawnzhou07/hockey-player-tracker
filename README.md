# HockeyPlayerTracker

Track hockey players in broadcast footage. Outputs per-player analytics: screen time, average speed, max speed, and total distance traveled.

---

## How It Works

1. **Detection** — YOLOv8 detects 15 classes per frame: `team_home`, `team_away`, and 13 rink landmarks (blue lines, face-off dots, corners, center line)
2. **Tracking** — DeepSORT assigns persistent IDs to players across frames using appearance features
3. **Calibration** — Detected rink landmarks are matched to known NHL rink coordinates (meters). `cv2.findHomography` computes a per-frame perspective transform from screen pixels → ice coordinates. Falls back to a fixed `0.03 m/px` constant when fewer than 3 landmarks are visible.
4. **Analytics** — Speed and distance are calculated from projected ice positions. Results written to CSV.

---

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, M4 Mac (MPS) or CUDA GPU recommended.

---

## Usage

### Run full analytics pipeline (screen time + speed + distance)

```bash
python src/video_inference_deepsort_analytics.py
```

Edit the `run_inference(...)` call at the bottom of the script to point to your video and desired output path.

**Output CSV columns:**

| Column | Description |
|---|---|
| `Player_ID` | DeepSORT track ID (e.g. `player_4`) |
| `Team` | `team_home` or `team_away` |
| `Screen_Time_s` | Seconds the player was visible on screen |
| `Avg_Speed_kmh` | Average skating speed (km/h) |
| `Max_Speed_kmh` | Peak skating speed (km/h) |
| `Total_Distance_m` | Total distance traveled on ice (meters) |
| `Frames_Detected` | Raw frame count (used to filter noise; < 30 excluded) |

### Run screen time only (no homography)

```bash
python src/video_inference_deepsort.py
```

Uses v2 (2-class) model. Faster — no rink feature detection overhead.

---

## Models

| Model | Classes | Frames | Use |
|---|---|---|---|
| `models/v1.pt` | 2 (players only) | 96 | Archived — used to bootstrap v2 labels |
| `models/v2.pt` | 2 (players only) | 549 | Screen time only (no homography) |
| `models/v3.pt` | 15 (players + rink features) | 82 | Full analytics pipeline |

See [models/MODEL_INFO.md](models/MODEL_INFO.md) for full training history.

---

## Scripts

| Script | Purpose |
|---|---|
| `src/frames_extract.py` | Extract frames from raw video at a set interval |
| `src/dataset_split.py` | Build `train/` and `val/` splits from `data/annotations/` |
| `src/model_train.py` | Train or fine-tune a YOLOv8 model |
| `src/model_auto_label.py` | Auto-label unlabeled frames using an existing model |
| `src/video_inference_deepsort.py` | DeepSORT tracking → screen time CSV |
| `src/video_inference_deepsort_analytics.py` | DeepSORT + homography → full analytics CSV |

---

## Known Limitations

- **Single game tested** — only validated on TBL@BUF 2026-03-08 footage (dark vs white jerseys)
- **Small rink feature dataset** — v3 trained on 82 frames; feature detection may degrade in unusual lighting or tight camera angles
- **Homography fallback** — camera pans reduce visible landmarks; speed/distance accuracy drops when fallback constant is used
- **ID switches** — DeepSORT can re-assign IDs after occlusions, causing speed spikes in the output
- **Frame skipping** — pipeline processes every other frame for speed; very fast movements may be undersampled
- **No annotated video output** — bounding box overlay not yet implemented (see [LEVELS.md](LEVELS.md))

---

## Project Status

See [LEVELS.md](LEVELS.md) for a full milestone breakdown. Levels 1–5 complete.
