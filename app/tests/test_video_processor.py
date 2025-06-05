import cv2
import os
import sys
import numpy as np
import logging # For the logger in video_emotion_processor

# Add the current directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

app_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'app'))
if app_dir not in sys.path:
    sys.path.append(app_dir)
    
# Suppress DeepFace internal logging for a cleaner test output
logging.getLogger('deepface').setLevel(logging.ERROR)
# Configure a basic logger for this test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the new function from your actual video_emotion_processor.py
from app.video_processor import get_facial_emotion_from_frame

# --- Configuration for Testing ---
# Provide a path to a test image or a video file.
# If using a video file, we'll extract the first frame.
TEST_MEDIA_PATH = r"C:\Users\kingk\OneDrive\Documents\Projects\AffectLink\data\WIN_20250529_10_51_21_Pro.mp4" # <--- IMPORTANT: CHANGE THIS PATH
# Example: "test_image.jpg" or "test_video.mp4"

def run_test():
    frame_to_process = None

    if not os.path.exists(TEST_MEDIA_PATH):
        logging.error(f"Error: Test media file not found at {TEST_MEDIA_PATH}")
        logging.info("Please update TEST_MEDIA_PATH to a valid image or video file on your system.")
        return

    # Determine if it's an image or video
    if TEST_MEDIA_PATH.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        logging.info(f"Loading image for test: {TEST_MEDIA_PATH}")
        frame_to_process = cv2.imread(TEST_MEDIA_PATH)
        if frame_to_process is None:
            logging.error(f"Failed to load image from {TEST_MEDIA_PATH}")
            return
    elif TEST_MEDIA_PATH.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        logging.info(f"Loading first frame from video for test: {TEST_MEDIA_PATH}")
        cap = cv2.VideoCapture(TEST_MEDIA_PATH)
        if not cap.isOpened():
            logging.error(f"Failed to open video file {TEST_MEDIA_PATH}")
            return
        ret, frame_to_process = cap.read()
        cap.release()
        if not ret or frame_to_process is None:
            logging.error(f"Failed to read first frame from video {TEST_MEDIA_PATH}")
            return
    else:
        logging.error(f"Unsupported media type for test: {TEST_MEDIA_PATH}")
        return

    logging.info(f"Processing frame of shape: {frame_to_process.shape}")

    # Call the new function
    emotion_data, full_scores = get_facial_emotion_from_frame(frame_to_process)

    # Print results
    if emotion_data and full_scores:
        logging.info(f"\n--- Facial Emotion Analysis Results ---")
        logging.info(f"Detected Emotion: {emotion_data[0]} (Confidence: {emotion_data[1]:.2f})")
        logging.info(f"Full Scores (Normalized): {full_scores}")
        logging.info(f"Test complete. Check for valid emotion data above.")
    else:
        logging.warning("No facial emotion data returned. Check for errors or no face detected.")

if __name__ == "__main__":
    run_test()