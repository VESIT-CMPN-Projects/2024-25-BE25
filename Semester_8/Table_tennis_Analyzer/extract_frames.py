import cv2
import os
from tqdm import tqdm

def extract_frames(video_path, output_folder, frame_interval=30, video_name=None):
    """
    Extract frames from a video file at specified intervals
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save extracted frames
        frame_interval: Extract every nth frame
        video_name: Name of the video (to use in the frame filename)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video has {total_frames} frames at {fps} FPS")
    
    frame_count = 0
    saved_count = 0
    
    # Loop through the video frames
    with tqdm(total=total_frames//frame_interval) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Save every nth frame
            if frame_count % frame_interval == 0:
                # Include video name in the filename to avoid conflicts
                frame_filename = os.path.join(output_folder, f"{video_name}_frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1
                pbar.update(1)
            
            frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

# Process all videos in a directory
def process_videos(videos_dir, output_dir, frame_interval=30):
    """Process all videos in a directory and save frames to a single folder"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    for video_file in video_files:
        video_path = os.path.join(videos_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        print(f"Processing video: {video_file}")
        # Pass the video name to include it in the frame filenames
        extract_frames(video_path, output_dir, frame_interval, video_name)

# Example usage
videos_dir = "E:/Table tennis analyser for changes/videos"
output_dir = "extracted_frames"
process_videos(videos_dir, output_dir, frame_interval=30)  # Adjust interval as needed