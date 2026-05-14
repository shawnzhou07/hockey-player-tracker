# HockeyPlayerTracker

**Goal:** Track hockey players in game footage, calculate ice time per player, and output annotated video with tracking overlays.

---

## Project Overview

An object detection and tracking system that:
1. Detects players in hockey game video
2. Tracks each player across frames with persistent IDs
3. Differentiates teams by jersey color
4. Calculates ice time statistics per player
5. Outputs video with real-time overlays showing tracking IDs

**Timeline:** 3-4 weeks

**Current Status:** Planning phase

---

## Architecture (High-Level)

Raw Video Input → Player Detection (YOLOv8) → Multi-Object Tracking (BoT-SORT or similar) → Team Classification (Color-based) → Ice Time Analytics (Frame counting) → Video Output (Annotated with overlays)

---

## Technology Stack

### Core ML/CV
- PyTorch - Deep learning framework
- YOLOv8 (Ultralytics) - Object detection model
- OpenCV - Video processing
- NumPy/Pandas - Data manipulation

### Tracking (To Be Decided)
- BoT-SORT (built into YOLOv8) - Multi-object tracking
- OR DeepSORT - Alternative tracking algorithm
- Decision: Start with BoT-SORT (simpler integration)

### Team Differentiation
- Color-based classification using jersey hue/saturation
- May use simple k-means clustering or manual thresholds

### Visualization
- Matplotlib - Charts/analytics
- OpenCV - Video annotation

---

## Hardware

- M4 MacBook (CPU/GPU)
- Note: M4 supports MPS (Metal Performance Shaders) for PyTorch acceleration

---

## Development Phases

### Phase 1: Setup & Data Collection (Week 1)
**Objective:** Environment ready, video data collected and organized

**Tasks:**
- Install dependencies (PyTorch, Ultralytics, OpenCV, etc.)
- Download 3-5 YouTube hockey clips (2-3 min each, broadcast angle, clear footage)
- Set up project structure (data/, models/, src/, outputs/)
- Verify MPS acceleration works on M4

**Success Criteria:**
- Can run sample YOLO inference on test image
- Have 3-5 video clips ready for labeling

**Time Estimate:** 2-3 days

---

### Phase 2: Data Labeling (Week 1-2)
**Objective:** Labeled dataset of 300-500 frames with player bounding boxes

**Tasks:**
- Install labeling tool (LabelImg or CVAT)
- Label players in frames (sample every 10-15 frames from clips)
- Create two classes: team_home (e.g., red jerseys), team_away (e.g., blue jerseys)
- Split data into train/val (80/20)
- Convert annotations to YOLO format

**Success Criteria:**
- 300-500 labeled frames
- Annotations in correct YOLO format
- Train/val split created

**Time Estimate:** 5-7 days (labeling is time-consuming)

**Notes:**
- Can use pre-trained model to auto-label after first 100 frames, then manually correct
- Focus on clear, unoccluded players initially

---

### Phase 3: Train Player Detector (Week 2)
**Objective:** Working YOLOv8 model that detects players and distinguishes teams

**Tasks:**
- Set up training script using Ultralytics YOLOv8
- Train model on labeled data (start with yolov8n for speed)
- Monitor training metrics (mAP, loss curves)
- Evaluate on validation set
- Tune hyperparameters if needed

**Success Criteria:**
- mAP > 0.6 on validation set (aim for 0.7+)
- Model can detect most players in test frames
- Can distinguish between teams with reasonable accuracy

**Time Estimate:** 2-3 days

**Notes:**
- Use MPS device for acceleration on M4
- Start with small model (yolov8n), can upgrade to yolov8s/m later if needed

---

### Phase 4: Multi-Object Tracking (Week 3)
**Objective:** Track players across frames with persistent IDs

**Tasks:**
- Integrate BoT-SORT tracking (built into Ultralytics)
- Run tracking on test videos
- Tune tracking parameters (max_age, n_init, etc.)
- Handle ID switches and occlusions
- Test on multiple clips

**Success Criteria:**
- Players maintain consistent IDs across most of the video
- ID switches are minimal (< 10% of tracks)
- Can track through brief occlusions

**Time Estimate:** 3-4 days

**Alternative:** If BoT-SORT doesn't work well, try ByteTrack or DeepSORT

---

### Phase 5: Ice Time Analytics (Week 3)
**Objective:** Calculate ice time statistics per player

**Tasks:**
- Count frames each tracking ID appears in
- Convert frame counts to seconds/minutes (using video FPS)
- Detect shifts (when player enters/leaves ice)
- Calculate per-player statistics: total ice time, number of shifts, avg shift length
- Output to CSV

**Success Criteria:**
- Accurate ice time calculations (within ±5% of manual count)
- Can identify individual shifts per player
- Clean CSV output with player stats

**Time Estimate:** 2-3 days

---

### Phase 6: Video Output with Overlays (Week 4)
**Objective:** Annotated video showing tracked players with overlays

**Tasks:**
- Draw bounding boxes on each player
- Add tracking ID labels above boxes
- Color-code boxes by team (red/blue)
- Add ice time counter per player (live updating)
- Optionally: Add stats panel showing top players by ice time
- Export annotated video

**Success Criteria:**
- Video clearly shows tracked players with IDs
- Overlays are readable and not cluttered
- Ice time updates correctly throughout video

**Time Estimate:** 2-3 days

---

### Phase 7: Testing & Polish (Week 4)
**Objective:** Clean code, documentation, working demo

**Tasks:**
- Test on multiple videos
- Fix edge cases and bugs
- Create README with setup instructions, usage, results
- Add requirements.txt
- Create sample outputs (video + stats CSV)
- Document known limitations

**Success Criteria:**
- Can process a new video end-to-end
- README clearly explains how to run the project
- Have demo video and stats to show

**Time Estimate:** 2-3 days

---

## Potential Extensions (Post-MVP)

These are ideas for future development, NOT required for initial version:

- Jersey number OCR (read numbers, match to roster)
- Player names overlay (if roster available)
- Advanced analytics: heat maps, positioning, line combinations
- Goal detection (track puck + goal area)
- Real-time processing optimization
- Web dashboard (Streamlit/Flask)
- Multi-game aggregation

---

## Known Limitations & Challenges

**Current assumptions:**
- Only tracking players, not puck or refs
- Teams must have clearly different jersey colors
- Requires broadcast-angle footage (not behind-net or corner views)
- No real-time processing (offline only)
- No player identity (just tracking IDs, not real names)

**Expected challenges:**
- Occlusions when players overlap
- ID switching when players cross paths
- Jersey color similarity in certain lighting
- Fast camera pans may lose tracks
- Players entering/leaving frame

---

## Success Metrics

**Minimum Viable Product (MVP):**
- Detect players with >70% mAP
- Track players with <15% ID switch rate
- Calculate ice time with ±10% accuracy
- Output watchable annotated video

**Stretch Goals:**
- >80% mAP detection
- <5% ID switch rate
- ±5% ice time accuracy
- Team classification >90% accurate

---

## File Structure

HockeyPlayerTracker/
├── data/
│   ├── raw_videos/           # Downloaded YouTube clips
│   ├── labeled_frames/       # Extracted frames for labeling
│   │   ├── train/
│   │   └── val/
│   └── annotations/          # YOLO format labels
├── models/
│   └── player_detector.pt    # Trained YOLOv8 model
├── src/
│   ├── train.py             # Model training script
│   ├── track.py             # Tracking pipeline
│   ├── analytics.py         # Ice time calculations
│   └── visualize.py         # Video annotation
├── outputs/
│   ├── tracked_videos/      # Annotated output videos
│   └── stats/               # CSV files with analytics
├── requirements.txt
├── README.md
└── PROJECT.md               # This file

---

## Dependencies (Initial List)

torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
pandas>=2.0.0
matplotlib>=3.7.0
numpy>=1.24.0
pillow>=10.0.0

Additional dependencies may be added as needed.

---

## Notes & Decisions Log

**2025-05-13:**
- Project scope defined: ice time tracking with team differentiation
- Timeline: 3-4 weeks
- Approach: Start simple, iterate
- Team classification: Color-based (not OCR)
- Tracking: BoT-SORT (built into YOLOv8)
- Hardware: M4 MacBook with MPS acceleration
- Data: 3-5 YouTube clips, 300-500 labeled frames

---

## Questions to Resolve

- Exact YouTube clips to use (need broadcast angle, clear footage)
- Labeling tool preference (LabelImg vs CVAT)
- Team color scheme for overlays (red/blue? home/away?)
- Video output resolution (720p? 1080p?)

---

## Contact & Resources

- YOLOv8 Documentation: https://docs.ultralytics.com/
- PyTorch MPS Guide: https://pytorch.org/docs/stable/notes/mps.html
- BoT-SORT Paper: https://arxiv.org/abs/2206.14651

---

**Last Updated:** 2026-05-14

---

## Project Structure & Conventions

### Directory Organization

- `data/annotations/` — YOLO-format label `.txt` files (corrected, source of truth)
- `models/` — Trained `.pt` model weights; see `models/MODEL_INFO.md` for version history
- `src/` — All Python scripts (see naming convention below)
- `runs/detect/` — Training run artifacts; keep final run for reference, delete predict* artifacts
- `outputs/` — Annotated videos and stats CSVs (gitignored)

### Script Naming Convention
- Use noun_verb pattern: `frames_extract.py`, `dataset_split.py`, `model_train.py`
- Be descriptive and consistent
- Group by domain: frames/dataset operations, model operations, video operations

---

## Dataset Versioning Strategy

### Approach
We maintain one active `data/train/` and `data/val/` directory for current work. Dataset history and composition is documented in `models/MODEL_INFO.md` rather than duplicating folders.

### Why This Approach
- No data duplication (v1 frames are subset of v2, no need to store twice)
- Clean file structure (one `train/` folder, not `train_v1/`, `train_v2/`, `train_v3/`)
- Full traceability through `MODEL_INFO.md` documentation
- Disk space efficient

### Documentation in MODEL_INFO.md
Each model entry documents exactly what data it was trained on:

Example:
```
### v2.pt
- Training Date: 2026-05-14
- Frames Used: 549 (439 train, 110 val)
- Source Game: TBL@BUF 2026-03-08
- Dataset Composition: 96 manually labeled + 453 auto-labeled (corrected)
- Notes: All frames from single game, dark vs white jersey contrast

### v3.pt (future example)
- Training Date: 2026-05-20
- Frames Used: 1500 (1200 train, 300 val)
- Source Games: TBL@BUF (549 frames), TOR@NYR (300 frames), DET@BOS (300 frames), CHI@MIN (351 frames)
- Dataset Composition: Mixed jersey colors for better generalization
- Notes: Multi-game training to handle diverse jersey combinations
```

### Workflow for Adding New Data
1. Extract and label frames from new game(s)
2. Add new annotations to `data/annotations/`
3. Run `dataset_split.py` to regenerate train/val split with ALL frames (old + new)
4. Train new model version
5. Update `models/MODEL_INFO.md` with the new dataset composition

### Reproducing Old Models
To retrain v2 exactly as it was:
1. Check `models/MODEL_INFO.md` for v2 dataset details
2. Filter `data/annotations/` to only include those 549 frames
3. Run `dataset_split.py` on filtered set
4. Train with same hyperparameters documented in `MODEL_INFO.md`

### Key Principle
The `data/train/` and `data/val/` folders represent "current working dataset" — they change as you add more data. `models/MODEL_INFO.md` is the source of truth for what each model was trained on. 