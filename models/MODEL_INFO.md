# Model Training History

This file tracks all trained models and their specifications. **Update this file every time you train a new model.**

## Model Version Guide

### v3.pt (CURRENT PRODUCTION MODEL)
- **Training Date:** 2026-05-14
- **Frames Used:** 82 (66 train, 16 val)
- **Source Game:** TBL@BUF 2026-03-08
- **Epochs Trained:** 50
- **Performance:** TBD (training metrics not yet recorded)
- **Classes (15):** team_home, team_away, corner_left_far, corner_right_far, blue_line_left_board, blue_line_right_board, center_line_board, dot_end_left_far, dot_end_left_close, dot_end_right_far, dot_end_right_close, dot_neutral_left_far, dot_neutral_left_close, dot_neutral_right_far, dot_neutral_right_close
- **Notes:** First model with rink feature detection. Enables homography-based calibration for real-world speed/distance analytics. Small training set (82 frames) — rink features may miss in unusual lighting or angles. Use with `video_inference_deepsort_analytics.py`.

---

### v2.pt
- **Training Date:** 2026-05-14
- **Frames Used:** 549 (439 train, 110 val)
- **Source Game:** TBL@BUF 2026-03-08
- **Epochs Trained:** 30 (early stopping at epoch 20)
- **Performance:**
  - mAP50: 99.1%
  - mAP50-95: 81.9%
  - Precision: 96.7%
  - Recall: 97.8%
- **Classes:** team_home (blue/dark jerseys), team_away (white jerseys)
- **Notes:** Production-ready. Auto-labeled 453 frames, manually corrected. Best model for TBL/BUF-style games (dark vs white jerseys).

---

### v1.pt
- **Training Date:** 2026-05-13
- **Frames Used:** 96 (76 train, 20 val)
- **Source Game:** TBL@BUF 2026-03-08
- **Epochs Trained:** 50
- **Performance:**
  - mAP50: 99.3%
  - mAP50-95: 72.9%
  - Precision: 98.5%
  - Recall: 95.6%
- **Classes:** team_home (blue/dark jerseys), team_away (white jerseys)
- **Notes:** Initial model. All frames manually labeled. Used to auto-label remaining frames for v2 training.

---

## Training Guidelines

When training a new model, add an entry above using this template:

```markdown
### v{N}.pt
- **Training Date:** YYYY-MM-DD
- **Frames Used:** {total} ({train} train, {val} val)
- **Source Game(s):** {game identifiers}
- **Epochs Trained:** {epochs}
- **Performance:**
  - mAP50: {value}%
  - mAP50-95: {value}%
  - Precision: {value}%
  - Recall: {value}%
- **Classes:** {class names and what they represent}
- **Notes:** {any important context, limitations, intended use}
```

## Usage Notes

- **Current production model:** Always the highest version number unless specified otherwise
- **Jersey color limitation:** Models trained only on TBL vs BUF may not generalize to games with similar jersey colors
- **Future improvements:** Train on 3-5 games with diverse jersey combinations for better generalization
