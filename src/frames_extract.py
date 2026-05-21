import cv2
import os

def extract_frames_from_range(video_path, output_dir, start_time, end_time, sample_rate=15):
    
    cap = cv2.VideoCapture(video_path) # open the video file
    
    fps = cap.get(cv2.CAP_PROP_FPS) # get the frames per second of the video
    
    print(f"Video FPS: {fps}")
    
    start_frame = int(start_time * fps) # convert start time to frame number
    end_frame = int(end_time * fps) # convert end time to frame number
    
    os.makedirs(output_dir, exist_ok=True) # create the output directory if it doesn't exist
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame) # jump to the start frame
    
    frame_count = start_frame
    saved_count = 0
    
    while frame_count < end_frame:
        ret, frame = cap.read() # read next frame
        if not ret:
            break # video ended
        
        if (frame_count - start_frame) % sample_rate == 0: # check if we should save this frame
            filename = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame) # save the frame as an image
            saved_count += 1
        
        frame_count += 1
    
    cap.release() # close the video file
    print(f"Extracted {saved_count} frames")

# call the function
extract_frames_from_range(
    video_path="data/raw_videos/TBL@BUF_2026-03-08.mp4",
    output_dir="data/labeled_frames/TBL@BUF_2026-03-08",
    start_time=0,
    end_time=275,
    sample_rate=15
)