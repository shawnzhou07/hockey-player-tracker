from ultralytics import YOLO # loads trained model
import cv2 # reads video frames
import pandas as pd # exports CSV
from deep_sort_realtime.deepsort_tracker import DeepSort # deepsort tracker

def run_inference(video_path, model_path, output_csv):
    model = YOLO(model_path) # load trained model
    
    cap = cv2.VideoCapture(video_path) # open video file
    
    fps = cap.get(cv2.CAP_PROP_FPS) # frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total frames in video
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    # tracking data
    player_detections = {} # counts frames each player appears in
    player_teams = {} # stores which team each player is on
    
    # initialize deepsort tracker
    tracker = DeepSort(max_age=30, n_init=3, embedder="mobilenet")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read() # read next frame
        if not ret: # video ended
            break
        
        frame_count += 1
        if frame_count % 2 != 0: # skip every other frame
            continue
        
        results = model.predict(frame, verbose=False, device='mps') # run detection on frame with GPU
        
        # prepare detections for deepsort (format: [[x1, y1, w, h, confidence, class], ...])
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # bounding box coordinates
                confidence = float(box.conf[0]) # detection confidence
                class_id = int(box.cls[0]) # 0 = team_home, 1 = team_away
                
                # convert to deepsort format [x, y, w, h]
                w = x2 - x1
                h = y2 - y1
                
                detections.append(([x1, y1, w, h], confidence, class_id))
        
        # update tracker with detections
        tracks = tracker.update_tracks(detections, frame=frame)
        
        # process confirmed tracks
        for track in tracks:
            if not track.is_confirmed(): # skip unconfirmed tracks
                continue
            
            player_id = f"player_{track.track_id}" # get track ID from deepsort
            class_id = track.get_det_class() # get class (0 or 1)
            team = 'team_home' if class_id == 0 else 'team_away' # convert to name
            
            # count this frame for this player
            if player_id not in player_detections:
                player_detections[player_id] = 0
            player_detections[player_id] += 1
            player_teams[player_id] = team # remember which team this player is on
        
        if frame_count % 100 == 0: # progress update every 100 frames
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release() # close video file
    
    # convert frame counts to ice time in seconds
    ice_time_data = []
    for player_id, frame_count in player_detections.items():
        ice_time_seconds = frame_count / fps # convert frames to seconds
        team = player_teams[player_id]
        
        ice_time_data.append({
            'Player_ID': player_id,
            'Team': team,
            'Screen_Time_Seconds': round(ice_time_seconds, 1),
            'Frames_Detected': frame_count
        })
    
    # create dataframe and export to CSV
    df = pd.DataFrame(ice_time_data)
    df = df[df['Frames_Detected'] >= 30] # filter out noise (players with < 1 second screen time)
    df = df.sort_values('Screen_Time_Seconds', ascending=False) # sort by most ice time first
    df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to {output_csv}")
    print(f"Total players tracked: {len(player_detections)}")

# call the function
run_inference(
    video_path='data/raw_videos/TBL@BUF_2026-03-08_clip_00-05-15_to_00-07-13.mp4',
    model_path='models/v2.pt',
    output_csv='outputs/ice_time_TBL@BUF_2026-03-08_clip_00-05-15_to_00-07-13_deepsort.csv'
)