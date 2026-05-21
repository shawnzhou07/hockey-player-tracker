# Project Levels — HockeyPlayerTracker

Tracks completed milestones and what's next. Each level builds on the previous.

---

## Level 1 — Player Detection ✅
**Status: Complete**

- Labeled 96 frames manually from TBL@BUF 2026-03-08 broadcast footage
- Trained YOLOv8n on 2 classes: `team_home`, `team_away`
- mAP50: 99.3%, mAP50-95: 72.9%
- Model: `models/v1.pt`

---

## Level 2 — Scaled Dataset + Better Model ✅
**Status: Complete**

- Used v1 to auto-label 453 additional frames, manually corrected
- Retrained on 549 total frames (439 train, 110 val)
- mAP50: 99.1%, mAP50-95: 81.9% — improved generalization
- Model: `models/v2.pt`

---

## Level 3 — DeepSORT Tracking + Screen Time ✅
**Status: Complete**

- Integrated DeepSORT for persistent player IDs across frames
- Screen time calculated by counting confirmed track frames / FPS
- Output CSV: `Player_ID`, `Team`, `Screen_Time_Seconds`, `Frames_Detected`
- Script: `src/video_inference_deepsort.py`

---

## Level 4 — Rink Feature Detection + Homography ✅
**Status: Complete**

- Extended YOLO model to 15 classes: 2 player + 13 rink landmarks
- Labeled 82 frames with all 15 classes, trained `models/v3.pt`
- Homography matrix computed per frame from detected rink features → NHL ice coordinates
- Falls back to fixed pixel-to-meter ratio (0.03 m/px) when < 3 features detected
- Script: `src/video_inference_deepsort_analytics.py`

---

## Level 5 — Full Movement Analytics ✅
**Status: Complete**

- Speed (avg/max km/h) and total distance (m) calculated from homography-projected positions
- Output CSV columns: `Player_ID`, `Team`, `Screen_Time_s`, `Avg_Speed_kmh`, `Max_Speed_kmh`, `Total_Distance_m`, `Frames_Detected`
- Players with < 30 detected frames filtered out as noise
- Script: `src/video_inference_deepsort_analytics.py`

---

## Level 6 — Annotated Video Output 🔲
**Status: Not started**

- Draw bounding boxes color-coded by team
- Overlay tracking ID and live ice time on each player
- Optional stats panel (top players by ice time or speed)
- Export as `.mp4`

---

## Level 7 — Multi-Game Generalization 🔲
**Status: Not started**

- Collect 3-5 additional games with diverse jersey colors
- Retrain v3 on expanded dataset for better rink feature detection
- Validate homography accuracy across different arena camera angles
- Target: homography active >80% of frames

---

## Level 8 — Polish & Demo 🔲
**Status: Not started**

- End-to-end processing of a new video without manual config changes
- Fill out README with setup, usage, and sample outputs
- Add requirements.txt with pinned versions
- Create demo video clip + stats CSV for GitHub

---

## Known Limitations (Current)

- v3 rink feature model trained on only 82 frames — may miss features in unusual lighting
- Speed spikes when DeepSORT re-assigns IDs after occlusion
- Fallback pixel-to-meter constant degrades accuracy during camera pans
- Only tested on TBL@BUF 2026-03-08 footage (one game, one camera angle)
