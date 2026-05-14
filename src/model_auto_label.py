from ultralytics import YOLO
import os

# load your trained model
model = YOLO('runs/detect/hockey_player_detector/weights/best.pt')

# directory with all unlabeled frames
all_frames_dir = 'data/labeled_frames/TBL@BUF_2026-03-08'
labeled_frames = set(os.listdir('data/annotations'))  # frames we already labeled

# get unlabeled frames
all_frames = [f for f in os.listdir(all_frames_dir) if f.endswith('.jpg')]
unlabeled_frames = [f for f in all_frames if f.replace('.jpg', '.txt') not in labeled_frames]

print(f"Found {len(unlabeled_frames)} unlabeled frames")

# run detection and save labels
for frame in unlabeled_frames:
    frame_path = os.path.join(all_frames_dir, frame)
    results = model.predict(frame_path, save_txt=True, save_conf=False)

print("Auto-labeling complete!")