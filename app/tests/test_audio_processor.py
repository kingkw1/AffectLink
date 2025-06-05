import os
import sys
import numpy as np
import soundfile as sf
import logging
import torch
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import whisper
import time # Added for timing model loading
import librosa # Added for loading audio from video

# Add the project root to sys.path to ensure local module imports work
# This assumes 'app' is at the same level as 'constants.py' or in a parent directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming project structure is: project_root/app/audio_emotion_processor.py
# And constants.py might be in project_root/constants.py or project_root/app/constants.py
project_root = os.path.dirname(current_dir)

# Add relevant paths if not already in sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
# If audio_emotion_processor.py is directly in 'app' and 'constants.py' is also in 'app'
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Set DeepFace logging to error to reduce noise, though not directly used here
logging.getLogger('deepface').setLevel(logging.ERROR)
# Configure a basic logger for this test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import the new function and necessary constants
try:
    from app.audio_processor import process_audio_chunk_from_file
    # Import constants from wherever they reside in your project
    # Adjust this import based on your actual constants.py location
    from constants import SER_TO_UNIFIED, TEXT_TO_UNIFIED, UNIFIED_EMOTIONS 
except ImportError as e:
    logging.error(f"Failed to import necessary modules. Ensure audio_emotion_processor.py and constants.py are accessible via sys.path. Error: {e}")
    logging.error(f"Current sys.path: {sys.path}")
    sys.exit(1)


# --- Configuration for Testing ---
# IMPORTANT: CHANGE THIS PATH to a valid audio file (e.g., .wav, .mp3) on your system.
# A clear voice speaking some words for 5-10 seconds would be ideal for testing transcription.
# AUDIO_FILE_PATH = "C:\\\\Users\\\\kingk\\\\OneDrive\\\\Documents\\\\Projects\\\\AffectLink\\\\data\\\\audio_test.wav" 
VIDEO_FILE_PATH = r"C:\Users\kingk\OneDrive\Documents\Projects\AffectLink\data\WIN_20250529_10_51_21_Pro.mp4"

# Model IDs (these should match what you use in main_processor.py)
TEXT_CLASSIFIER_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SER_MODEL_ID = "superb/hubert-large-superb-er"
WHISPER_MODEL_SIZE = "base" # Or "small", "medium", etc.

def load_models():
    """Loads all necessary AI models for audio processing."""
    logging.info("Starting model loading...")
    start_time = time.time()

    logging.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    logging.info("Whisper model loaded.")

    logging.info(f"Loading text emotion classifier: '{TEXT_CLASSIFIER_MODEL_ID}'...")
    text_emotion_classifier = pipeline("sentiment-analysis", model=TEXT_CLASSIFIER_MODEL_ID)
    logging.info("Text emotion classifier loaded.")

    logging.info(f"Loading audio feature extractor: '{SER_MODEL_ID}'...")
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL_ID)
    logging.info("Audio feature extractor loaded.")

    logging.info(f"Loading audio emotion classifier: '{SER_MODEL_ID}'...")
    audio_emotion_classifier = AutoModelForAudioClassification.from_pretrained(SER_MODEL_ID)
    logging.info("Audio emotion classifier loaded.")

    end_time = time.time()
    logging.info(f"All models loaded in {end_time - start_time:.2f} seconds.")
    return whisper_model, text_emotion_classifier, audio_feature_extractor, audio_emotion_classifier

def run_test():
    if not os.path.exists(VIDEO_FILE_PATH):
        logging.error(f"Error: Video file not found at {VIDEO_FILE_PATH}")
        logging.info("Please update VIDEO_FILE_PATH to a valid video file on your system.")
        return

    logging.info(f"Attempting to load audio from video: {VIDEO_FILE_PATH}")
    try:
        # Load audio from video file using librosa.
        # sr=None preserves the original sampling rate.
        # mono=True converts to mono.
        # dtype=np.float32 ensures the data is float32.
        audio_data, audio_sample_rate = librosa.load(VIDEO_FILE_PATH, sr=None, mono=True, dtype=np.float32)
        logging.info(f"Audio loaded from video. Shape: {audio_data.shape}, Sample Rate: {audio_sample_rate}Hz")
    except Exception as e:
        logging.error(f"Error loading audio from video file {VIDEO_FILE_PATH}: {e}", exc_info=True)
        return

    # Load models (this will download models on first run)
    whisper_model, text_emotion_classifier, audio_feature_extractor, audio_emotion_classifier = load_models()

    # Define a chunk size (e.g., 5 seconds of audio)
    chunk_duration_seconds = 5
    chunk_size_samples = int(chunk_duration_seconds * audio_sample_rate)

    # Take the first chunk for testing, or the whole audio if it's shorter
    test_chunk = audio_data[:min(len(audio_data), chunk_size_samples)]

    if len(test_chunk) == 0:
        logging.error("Test audio chunk is empty after loading. Please use a longer video file or check the video content.")
        return

    logging.info(f"\\nProcessing audio chunk of {len(test_chunk) / audio_sample_rate:.2f} seconds...")

    # Call the new function
    transcribed_text, text_emotion_data, audio_emotion_data = \
        process_audio_chunk_from_file(
            test_chunk,
            audio_sample_rate,
            whisper_model,
            text_emotion_classifier,
            audio_feature_extractor,
            audio_emotion_classifier
        )

    # Print results
    logging.info(f"\n--- Audio Emotion Analysis Results ---")
    logging.info(f"Transcribed Text: \"{transcribed_text}\"")
    logging.info(f"Text Emotion: {text_emotion_data[0]} (Confidence: {text_emotion_data[1]:.2f})")
    logging.info(f"Audio Emotion: {audio_emotion_data[0]} (Confidence: {audio_emotion_data[1]:.2f})")
    logging.info(f"Test complete. Check for valid transcription and emotion data above.")

if __name__ == "__main__":
    run_test()