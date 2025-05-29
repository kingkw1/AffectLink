import os
import sys
import time
import mlflow
import cv2
import librosa
import numpy as np
import soundfile as sf
import torch
import whisper
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import logging

# Add the project root to sys.path to ensure local modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import local modules
from constants import (
    FACIAL_TO_UNIFIED, SER_TO_UNIFIED, UNIFIED_EMOTIONS, TEXT_TO_UNIFIED,
    AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE,
    TEXT_CLASSIFIER_MODEL_ID, SER_MODEL_ID
)
from audio_emotion_processor import process_audio_chunk_from_file
from video_emotion_processor import get_facial_emotion_from_frame
from main_processor import calculate_cosine_similarity, get_consistency_level, calculate_average_multimodal_similarity

# Set up environment for DeepFace model caching
deepface_cache_dir = os.path.join(project_root, "models", "deepface_cache")
os.environ['DEEPFACE_HOME'] = deepface_cache_dir
os.makedirs(deepface_cache_dir, exist_ok=True)

# Configure logging for batch_analyzer.py
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress DeepFace logging for cleaner console output during analysis
logging.getLogger('deepface').setLevel(logging.ERROR)

# --- Model Loading (will be done once in process_media_file) ---
whisper_model = None
text_emotion_classifier = None
audio_feature_extractor = None
audio_emotion_classifier = None

def load_models():
    """Loads all necessary AI/ML models."""
    global whisper_model, text_emotion_classifier, audio_feature_extractor, audio_emotion_classifier
    logger.info("Starting model loading...")
    
    if whisper_model is None:
        logger.info("Loading Whisper model 'base'...")
        whisper_model = whisper.load_model("base")
        logger.info("Whisper model loaded.")

    if text_emotion_classifier is None:
        logger.info(f"Loading text emotion classifier: '{TEXT_CLASSIFIER_MODEL_ID}'...")
        # Suppress TensorFlow warnings if using TF backend for pipeline
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TF info, warning, error messages
        try:
            text_emotion_classifier = pipeline(
                "sentiment-analysis", 
                model=TEXT_CLASSIFIER_MODEL_ID, 
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.error(f"Error loading text emotion classifier: {e}")
            text_emotion_classifier = None # Ensure it's None if loading fails
        logger.info("Text emotion classifier loaded.")

    if audio_feature_extractor is None:
        logger.info(f"Loading audio feature extractor: '{SER_MODEL_ID}'...")
        audio_feature_extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL_ID)
        logger.info("Audio feature extractor loaded.")

    if audio_emotion_classifier is None:
        logger.info(f"Loading audio emotion classifier: '{SER_MODEL_ID}'...")
        audio_emotion_classifier = AutoModelForAudioClassification.from_pretrained(SER_MODEL_ID)
        if torch.cuda.is_available():
            audio_emotion_classifier.to("cuda")
        logger.info("Audio emotion classifier loaded.")

    logger.info("All models loaded.")

def process_audio(input_file_path):
    logger.info(f"Processing audio from file: {input_file_path}")
    try:
        # Load entire audio file
        audio_data, current_audio_sample_rate = librosa.load(input_file_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
        
        total_samples = len(audio_data)
        samples_per_chunk = int(AUDIO_CHUNK_SIZE * current_audio_sample_rate)
        
        for i in range(0, total_samples, samples_per_chunk):
            audio_chunk = audio_data[i : i + samples_per_chunk]
            
            # If the chunk is too small at the end, pad it or skip
            if len(audio_chunk) < samples_per_chunk and i + samples_per_chunk < total_samples:
                # Pad the last chunk if it's too short for the model to process effectively
                padding_needed = samples_per_chunk - len(audio_chunk)
                audio_chunk = np.pad(audio_chunk, (0, padding_needed), mode='constant')
            elif len(audio_chunk) == 0:
                continue # Skip empty chunks

            # Process the audio chunk using the dedicated processor
            transcribed_text, text_emotion_data, audio_emotion_data = \
                process_audio_chunk_from_file(
                    audio_chunk, current_audio_sample_rate, 
                    whisper_model, text_emotion_classifier, 
                    audio_feature_extractor, audio_emotion_classifier
                )

            start_time_chunk_sec = i / current_audio_sample_rate
            end_time_chunk_sec = (i + len(audio_chunk)) / current_audio_sample_rate

            logger.info(f"[{start_time_chunk_sec:.2f}s - {end_time_chunk_sec:.2f}s] Audio Chunk Results:")
            logger.info(f"  Transcribed Text: \"{transcribed_text}\"")
            logger.info(f"  Text Emotion: {text_emotion_data[0]} (Conf: {text_emotion_data[1]:.2f})")
            logger.info(f"  Audio Emotion: {audio_emotion_data[0]} (Conf: {audio_emotion_data[1]:.2f})")
        
        logger.info(f"Finished processing audio file: {input_file_path}")
    except Exception as e:
        logger.error(f"Error processing audio file {input_file_path}: {e}", exc_info=True)

def process_video(input_file_path):
    logger.info(f"Processing video from file: {input_file_path}")
    cap = cv2.VideoCapture(input_file_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {input_file_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 # Time in seconds

        facial_emotion = ("unknown", 0.0)
        
        try:                  
            facial_emotion_data, raw_emotion_scores = get_facial_emotion_from_frame(frame)
                    
            if facial_emotion_data and facial_emotion_data[0] != "unknown" and facial_emotion_data[0] != "error":
                dominant_emotion, confidence = facial_emotion_data
                confidence = max(raw_emotion_scores.values(), default=0.0) # Get max confidence from scores

                # Map to unified emotion
                unified_emotion = FACIAL_TO_UNIFIED.get(dominant_emotion)
                if unified_emotion:
                    facial_emotion = (unified_emotion, float(confidence))
                else:
                    logger.warning(f"Facial emotion label '{dominant_emotion}' not found in FACIAL_TO_UNIFIED map.")                

        except Exception as e:
            logger.debug(f"No face detected or error in facial analysis for frame {frame_idx}: {e}")
            facial_emotion = ("unknown", 0.0)

        logger.info(f"[{timestamp_sec:.2f}s] Frame {frame_idx}: Facial Emotion: {facial_emotion[0]} (Conf: {facial_emotion[1]:.2f})")
        frame_idx += 1
    
    cap.release()
    logger.info(f"Finished processing video file: {input_file_path}")


def process_media_file(input_file_path):
    """
    Processes a video or audio file for emotion analysis and prints results.
    """
    file_extension = os.path.splitext(input_file_path)[1].lower()
    
    load_models() # Ensure models are loaded
    
    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(input_file_path)

        # Check if the video has audio
        try:
            audio_data, sample_rate = librosa.load(input_file_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
            if len(audio_data) > 0:
                logger.info(f"Audio track found in video {input_file_path}. Processing audio...")
                process_audio(input_file_path)
            else:
                logger.warning(f"No audio track found in video {input_file_path}.")
        except Exception as e:
            logger.error(f"Error loading audio from video file {input_file_path}: {e}", exc_info=True)

    elif file_extension in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        process_audio(input_file_path)

    else:
        logger.error(f"Unsupported file type: {file_extension} for {input_file_path}")

def main(video_file_path):
    start_time = time.time()
    
    # Check if the hardcoded path exists
    if not os.path.exists(video_file_path):
        logger.error(f"Error: VIDEO_FILE_PATH '{video_file_path}' does not exist. Please update it to a valid path.")
        return

    with mlflow.start_run(run_name=f"Batch_Analysis_{os.path.basename(video_file_path)}_{time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_param("input_file", video_file_path)
        mlflow.log_param("analysis_type", "Offline Batch")

        process_media_file(video_file_path)

        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("processing_duration_seconds", duration)
        logger.info(f"Batch analysis completed in {duration:.2f} seconds.")
        logger.info(f"MLflow run logged. View with: 'mlflow ui'")

if __name__ == '__main__':

    # --- Hardcoded File Path (UPDATE THIS!) ---
    video_file_path = "C:\\Users\\kingk\\OneDrive\\Documents\\Projects\\AffectLink\\data\\WIN_20250529_10_51_21_Pro.mp4"

    main(video_file_path)