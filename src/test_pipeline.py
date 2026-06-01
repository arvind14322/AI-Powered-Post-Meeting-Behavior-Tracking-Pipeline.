import os
import glob
from pipeline import EngagementPipeline

def main():
    # Find any video file in the workspace
    video_files = glob.glob("*.mp4") + glob.glob("data/*.mp4")
    
    if not video_files:
        print("No .mp4 video files found in workspace.")
        print("Please place a recorded meeting video (e.g., 'meeting.mp4') in the data/ folder.")
        return

    test_video = video_files[0]
    print(f"Testing pipeline with video: {test_video}")
    
    # Initialize pipeline
    pipeline = EngagementPipeline(model_path="../models/emotion_cnn.pth", num_classes=5)
    
    # Process first 100 frames
    print("Processing video frames...")
    df = pipeline.process_video(test_video, max_frames=100)
    
    print("\nPipeline execution complete. Sample output:")
    print(df[["frame", "participant_id", "face_present", "gaze_deviation", "dominant_emotion", "engagement_score_smoothed"]].head(10))

if __name__ == "__main__":
    main()
