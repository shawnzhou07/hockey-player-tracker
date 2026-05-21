# Feature Roadmap: Levels of Complexity

This document outlines the feature progression for the hockey player tracking system, organized by implementation difficulty and time investment.

---

## Level 1: MVP — Basic Ice Time Tracker
**Time Required:** 5-7 hours  
**Difficulty:** ⭐⭐☆☆☆ (Beginner-Intermediate)  
**Status:** IN PROGRESS

### Features:
- ✓ Load trained model (v2.pt)
- ✓ Run detection on full game video
- ✓ Proximity-based tracking (simple distance matching)
- ✓ Calculate screen time per player
- ✓ Export CSV with Player_ID, Team, Screen_Time_Seconds
- ✓ Document limitations in README

### Known Limitations:
- Off-camera time not counted
- IDs reset on camera cuts
- Includes stoppage time
- Players crossing paths may swap IDs

### Why Build This First:
- Functional deliverable
- Core pipeline (detection → tracking → export)
- Foundation for all future features
- Validates the trained model works on full video

### Skills Learned:
- Video processing with OpenCV
- Object tracking fundamentals
- Data aggregation and export
- Production ML deployment

---

## Level 2: Advanced Tracking (DeepSORT/ByteTrack)
**Time Required:** 8-12 hours  
**Difficulty:** ⭐⭐⭐☆☆ (Intermediate)  
**Status:** NOT STARTED

### Features:
- Integrate DeepSORT or ByteTrack library
- Appearance-based feature matching (not just position)
- Motion prediction (Kalman filter)
- Handle camera cuts via visual similarity
- More robust ID persistence

### Improvements Over Level 1:
- IDs survive camera cuts (70-80% success rate)
- Better handling of occlusions
- Reduced ID swapping when players cross
- More accurate long-term tracking

### Implementation:
```python
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=30, n_init=3)
```

### Resources:
- deep-sort-realtime library (Python)
- ByteTrack paper + implementation
- Kalman filter tutorials

### Why Not Build This First:
- Adds complexity before core pipeline is proven
- Requires understanding of Level 1 tracking first
- Libraries can be finicky (dependency hell)

---

## Level 3: Jersey Number OCR
**Time Required:** 10-15 hours  
**Difficulty:** ⭐⭐⭐⭐☆ (Intermediate-Advanced)  
**Status:** NOT STARTED

### Features:
- Crop player bounding boxes from frames
- Run OCR (EasyOCR or Tesseract) on jersey back
- Extract numbers (1-99)
- Filter false positives (noise, wrong detections)
- Map numbers to permanent player IDs

### Challenges:
- Jersey numbers are small (20-30 pixels often)
- Motion blur in fast gameplay
- Occluded by other players
- Only visible from certain angles (back view)
- OCR accuracy ~60-70% on hockey footage

### Implementation Strategy:
1. Detect front vs back of jersey (pose estimation)
2. Only run OCR on clear back views
3. Track OCR results over multiple frames (consensus voting)
4. Fallback to proximity tracking when OCR fails

### Dependencies:
- EasyOCR or PaddleOCR (better for small text)
- Image preprocessing (sharpening, contrast enhancement)
- Filtering logic (validate numbers 1-99)

### Expected Outcome:
- 60-80% of players get correct jersey number ID
- Remaining 20-40% fall back to proximity tracking
- Significant improvement in ID persistence

---

## Level 4: Player Name Lookup (NHL API Integration)
**Time Required:** 3-5 hours  
**Difficulty:** ⭐⭐☆☆☆ (Easy-Intermediate)  
**Status:** NOT STARTED  
**Prerequisite:** Level 3 (Jersey OCR)

### Features:
- Call NHL Stats API to get roster for game date
- Map jersey numbers to player names
- Enrich CSV output with real names

### API Endpoint:
https://statsapi.web.nhl.com/api/v1/schedule?date=2026-03-08
https://statsapi.web.nhl.com/api/v1/teams/{teamId}/roster

### Output Enhancement:
Before:
```csv
Player_ID,Team,Screen_Time_Seconds
player_1,team_home,324.5
```

After:
```csv
Player_ID,Jersey_Number,Player_Name,Team,Screen_Time_Seconds
player_1,91,Steven Stamkos,team_home,324.5
```

### Why This is Easy:
- NHL API is well-documented and free
- Simple JSON parsing
- No computer vision complexity
- Only depends on having jersey numbers from Level 3

---

## Level 5: Game Clock OCR (Active Play Detection)
**Time Required:** 5-8 hours  
**Difficulty:** ⭐⭐⭐☆☆ (Intermediate)  
**Status:** NOT STARTED

### Features:
- Detect game clock position on screen (varies by broadcast)
- Extract time remaining via OCR (MM:SS format)
- Detect when clock is running vs stopped
- Only count ice time during active play

### Challenges:
- Clock position changes between broadcasts
- Clock sometimes hidden or small
- Need to detect clock movement (running vs stopped)
- Handle period transitions (20:00 → 0:00 → 20:00)

### Implementation:
1. Manual region-of-interest (ROI) selection per broadcast
2. Run OCR on clock region every frame
3. Compare timestamps: if changing → play is active
4. Gate ice time counting based on clock status

### Expected Outcome:
- Ice time now excludes stoppages, timeouts, intermissions
- More accurate "official ice time" estimates
- Still imperfect (OCR errors, clock display issues)

---

## Level 6: Camera Cut Detection & Re-identification
**Time Required:** 20-30 hours  
**Difficulty:** ⭐⭐⭐⭐⭐ (Advanced-Research Level)  
**Status:** NOT STARTED  
**Prerequisite:** Level 2 (DeepSORT)

### Features:
- Detect camera cuts/scene changes
- Extract visual embeddings (ResNet features) per player
- Re-identify players after cuts via appearance matching
- Handle complete perspective changes (wide shot → close-up)

### Why This is Hard:
- Player appearance changes drastically with angle
- Lighting differences between cameras
- Similar jerseys confuse appearance features
- No ground truth for re-identification training
- Requires deep learning for feature extraction

### Research-Level Techniques:
- Scene change detection (histogram differences)
- Siamese networks for re-identification
- Visual feature banks (store embeddings per player)
- Cosine similarity matching across cuts

### Expected Outcome:
- 70-80% success rate on re-identification
- Significant improvement over proximity tracking
- Still not perfect (some players will get new IDs)

### Alternatives:
- Use jersey number OCR as primary ID (Level 3)
- Accept ID resets as documented limitation
- Wait for production-grade tracking systems

---

## Level 7: Multi-Game Training & Generalization
**Time Required:** 15-20 hours  
**Difficulty:** ⭐⭐⭐☆☆ (Intermediate)  
**Status:** NOT STARTED

### Features:
- Label 300+ frames from 3-5 different games
- Diverse jersey colors (red vs white, blue vs black, etc.)
- Train v3 model on ~1500 frames
- Test generalization across NHL games

### Why This Matters:
- Current model only works on dark vs light jerseys
- Will fail on similar jersey colors (red vs blue)
- Need diverse training data for production use

### Games to Add:
1. Red vs white (e.g., Red Wings vs anyone)
2. Blue vs black (Maple Leafs vs someone)
3. Yellow vs dark (Predators game)
4. Keep existing TBL vs BUF (dark vs white)

### Time Breakdown:
- Download/extract frames: 2 hours
- Auto-label with v2: 1 hour
- Manual corrections: 8-12 hours
- Retrain model: 2 hours
- Validation testing: 2 hours

---

## Level 8: Broadcast Overlay System (Bonus Feature)
**Time Required:** 20-25 hours  
**Difficulty:** ⭐⭐⭐⭐☆ (Advanced)  
**Status:** NOT STARTED  
**Prerequisites:** Levels 3, 4 (Jersey OCR + API)

### Features:
- Real-time stats overlay on video
- Display player name, number, stats on detection
- Current shift length, ice time this game
- Speed calculation (position change per frame)
- Professional broadcast-style graphics

### Implementation:
1. Detect player → get jersey number (OCR)
2. Fetch stats from NHL API (goals, assists, points)
3. Draw overlay box with stats near player
4. Update in real-time as video plays

### Example Output:
[Player bounding box]
#91 Steven Stamkos
18:32 ice time | Current shift: 1:23
Speed: 24.5 km/h

### Why This is Cool:
- Looks extremely impressive
- Combines all previous features
- Production-quality demo
- Portfolio showpiece

---

## Recommended Implementation Order

**Phase 1 (Week 1):** Level 1 → MVP ice time tracker  
**Phase 2 (Week 2):** Level 2 → DeepSORT tracking  
**Phase 3 (Week 3):** Level 3 → Jersey OCR  
**Phase 4 (Week 4):** Level 4 → NHL API integration  

**Stop here for a solid portfolio project.**

**Phase 5+ (Optional):** Levels 5-8 if you want to go deeper