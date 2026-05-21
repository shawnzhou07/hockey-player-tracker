from ultralytics import YOLO # loads trained model
import cv2 # reads video frames
import pandas as pd # exports CSV
from deep_sort_realtime.deepsort_tracker import DeepSort # deepsort tracker
import numpy as np # for homography calculations

def run_inference(video_path, model_path, output_csv):

    model = YOLO(model_path) # load trained model
    
    cap = cv2.VideoCapture(video_path) # open video file
    
    fps = cap.get(cv2.CAP_PROP_FPS) # frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # total frames in video
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    # tracking data
    player_detections = {} # counts frames each player appears in
    player_teams = {} # stores which team each player is on
    player_position_history = {} # stores position per frame for speed/distance
    
    # rink feature coordinates in meters (NHL rink dimensions)
    rink_coords = {
        'corner_left_far': (-30.5, 13),
        'corner_right_far': (30.5, 13),
        'blue_line_left_board': (-15, 13),
        'blue_line_right_board': (15, 13),
        'center_line_board': (0, 13),
        'dot_end_left_far': (-20, 6.7),
        'dot_end_left_close': (-20, -6.7),
        'dot_end_right_far': (20, 6.7),
        'dot_end_right_close': (20, -6.7),
        'dot_neutral_left_far': (-6.7, 6.7),
        'dot_neutral_left_close': (-6.7, -6.7),
        'dot_neutral_right_far': (6.7, 6.7),
        'dot_neutral_right_close': (6.7, -6.7)
    }
    
    # class name mapping (matches dataset.yaml order)
    class_names = [
        'team_home', 'team_away', 'corner_left_far', 'corner_right_far',
        'blue_line_left_board', 'blue_line_right_board', 'center_line_board',
        'dot_end_left_far', 'dot_end_left_close', 'dot_end_right_far',
        'dot_end_right_close', 'dot_neutral_left_far', 'dot_neutral_left_close',
        'dot_neutral_right_far', 'dot_neutral_right_close'
    ]
    
    # fallback calibration constant (used when homography fails)
    PIXELS_TO_METERS_FALLBACK = 0.03
    
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
        
        # separate player detections from rink feature detections
        player_detections_frame = []
        rink_features = {}
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # bounding box coordinates
                confidence = float(box.conf[0]) # detection confidence
                class_id = int(box.cls[0]) # class index
                class_name = class_names[class_id]
                
                # check if it's a player or rink feature
                if class_name in ['team_home', 'team_away']:
                    # player detection
                    w = x2 - x1
                    h = y2 - y1
                    player_detections_frame.append(([x1, y1, w, h], confidence, class_id))
                else:
                    # rink feature detection
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    rink_features[class_name] = (center_x, center_y)
        
        # compute homography if 3+ rink features detected
        H = None
        use_homography = False
        
        if len(rink_features) >= 3:
            try:
                # prepare points for homography
                screen_points = []
                rink_points = []
                
                for feature_name, screen_pos in rink_features.items():
                    if feature_name in rink_coords:
                        screen_points.append(screen_pos)
                        rink_points.append(rink_coords[feature_name])
                
                if len(screen_points) >= 3:
                    screen_points = np.array(screen_points, dtype='float32')
                    rink_points = np.array(rink_points, dtype='float32')
                    
                    # compute homography matrix
                    H, _ = cv2.findHomography(screen_points, rink_points)
                    use_homography = True
            except:
                pass # homography failed, use fallback
        
        # update tracker with player detections
        tracks = tracker.update_tracks(player_detections_frame, frame=frame)
        
        # process confirmed tracks
        track_idx = 0
        for track in tracks:
            if not track.is_confirmed(): # skip unconfirmed tracks
                continue
            
            player_id = f"player_{track.track_id}" # get track ID from deepsort
            class_id = track.get_det_class() # get class (0 or 1)
            team = 'team_home' if class_id == 0 else 'team_away' # convert to name
            
            # get player bounding box
            if track_idx < len(player_detections_frame):
                bbox = player_detections_frame[track_idx][0] # [x, y, w, h]
                x1, y1, w, h = bbox
                x2 = x1 + w
                y2 = y1 + h
                
                # calculate player position on ice (bottom center of bbox = feet)
                player_screen_x = (x1 + x2) / 2
                player_screen_y = y2 # bottom of bbox
                
                # transform to ice coordinates
                if use_homography and H is not None:
                    # use homography transformation
                    screen_pos = np.array([[[player_screen_x, player_screen_y]]], dtype='float32')
                    ice_pos = cv2.perspectiveTransform(screen_pos, H)
                    ice_x, ice_y = ice_pos[0][0]
                else:
                    # fallback: use screen coordinates (will be converted later)
                    ice_x, ice_y = player_screen_x, player_screen_y
                
                # store position history
                timestamp = frame_count / fps
                if player_id not in player_position_history:
                    player_position_history[player_id] = []
                
                player_position_history[player_id].append({
                    'timestamp': timestamp,
                    'x': ice_x,
                    'y': ice_y,
                    'use_homography': use_homography
                })
            
            track_idx += 1
            
            # count this frame for this player
            if player_id not in player_detections:
                player_detections[player_id] = 0
            player_detections[player_id] += 1
            player_teams[player_id] = team # remember which team this player is on
        
        if frame_count % 100 == 0: # progress update every 100 frames
            print(f"Processed {frame_count}/{total_frames} frames")
    
    cap.release() # close video file
    
    # calculate speed and distance for each player
    ice_time_data = []
    
    for player_id, frame_count_val in player_detections.items():
        positions = player_position_history.get(player_id, [])
        
        if len(positions) < 2:
            # not enough data to calculate speed
            ice_time_data.append({
                'Player_ID': player_id,
                'Team': player_teams[player_id],
                'Screen_Time_s': round(frame_count_val / fps, 1),
                'Avg_Speed_kmh': 0,
                'Max_Speed_kmh': 0,
                'Total_Distance_m': 0,
                'Frames_Detected': frame_count_val
            })
            continue
        
        # calculate distances and speeds
        speeds = []
        total_distance = 0
        
        for i in range(1, len(positions)):
            prev = positions[i-1]
            curr = positions[i]
            
            # calculate distance
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            
            if curr['use_homography']:
                # already in meters
                distance_m = np.sqrt(dx**2 + dy**2)
            else:
                # convert pixels to meters using fallback
                distance_px = np.sqrt(dx**2 + dy**2)
                distance_m = distance_px * PIXELS_TO_METERS_FALLBACK
            
            total_distance += distance_m
            
            # calculate speed
            time_delta = curr['timestamp'] - prev['timestamp']
            if time_delta > 0:
                speed_ms = distance_m / time_delta
                speed_kmh = speed_ms * 3.6 # convert m/s to km/h
                speeds.append(speed_kmh)
        
        # aggregate stats
        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = np.max(speeds) if speeds else 0
        
        ice_time_data.append({
            'Player_ID': player_id,
            'Team': player_teams[player_id],
            'Screen_Time_s': round(frame_count_val / fps, 1),
            'Avg_Speed_kmh': round(avg_speed, 1),
            'Max_Speed_kmh': round(max_speed, 1),
            'Total_Distance_m': round(total_distance, 1),
            'Frames_Detected': frame_count_val
        })
    
    # create dataframe and export to CSV
    df = pd.DataFrame(ice_time_data)
    df = df[df['Frames_Detected'] >= 30] # filter out noise (players with < 1 second screen time)
    df = df.sort_values('Screen_Time_s', ascending=False) # sort by most ice time first
    df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to {output_csv}")
    print(f"Total players tracked: {len(player_detections)}")

# call the function
run_inference(
    video_path='data/raw_videos/TBL@BUF_2026-03-08_clip_00-05-15_to_00-07-13.mp4',
    model_path='models/v2.pt',
    output_csv='outputs/ice_time_TBL@BUF_2026-03-08_clip_00-05-15_to_00-07-13_analytics.csv'
)