from ultralytics import YOLO
import cv2 # reads video frames
import pandas as pd # exports CSV
from collections import defaultdict # counts frames per player

def run_inference(video_path, model_path, output_csv):
    model = YOLO(model_path) # load trained model
    
    cap = cv2.VideoCapture(video_path) # open video file
    
    fps = cap.get(cv2.CAP_PROP_FPS) # frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total frames in video
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    # tracking data
    player_detections = defaultdict(int) # counts frames each player appears in
    player_teams = {} # stores which team each player is on
    next_player_id = 1 # counter for assigning new player IDs
    previous_detections = [] # detections from last frame for proximity matching
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read() # read next frame
        if not ret: # video ended
            break
        
        results = model.predict(frame, verbose=False) # run detection on frame
        current_detections = []
        
        # process each detected player in this frame
        for result in results: # usually once runs once since its a list of one containing prediction results
            for box in result.boxes:
                # extract detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # bounding box coordinates
                confidence = float(box.conf[0]) # detection confidence
                class_id = int(box.cls[0]) # 0 = team_home, 1 = team_away
                team = 'team_home' if class_id == 0 else 'team_away' # convert to name
                
                # calculate center point for tracking
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # find closest player from previous frame (proximity tracking)
                matched_id = None
                min_distance = float('inf') # start with infinite distance
                
                for prev_detection in previous_detections:
                    prev_x, prev_y = prev_detection['center']
                    
                    # calculate distance between current and previous detection
                    distance = ((center_x - prev_x)**2 + (center_y - prev_y)**2)**0.5
                    
                    if distance < min_distance and distance < 50: # within 50 pixels
                        min_distance = distance
                        matched_id = prev_detection['id']
                
                # assign ID based on match result
                if matched_id is not None:
                    player_id = matched_id # reuse existing ID
                else:
                    player_id = f"player_{next_player_id}" # create new ID
                    next_player_id += 1
                
                # store detection for next frame's matching
                current_detections.append({
                    'id': player_id,
                    'center': (center_x, center_y),
                    'team': team
                })
                
                # count this frame for this player
                player_detections[player_id] += 1
                player_teams[player_id] = team # remember which team this player is on
        
        # update previous detections for next frame
        previous_detections = current_detections
        
        frame_count += 1
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
    output_csv='outputs/ice_time_TBL@BUF_2026-03-08_clip_00-05-15_to_00-07-13_proxmity.csv'
)