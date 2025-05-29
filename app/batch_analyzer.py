# filepath: c:\Users\kingk\OneDrive\Documents\Projects\AffectLink\app\batch_analyzer.py
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
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
        try:
            text_emotion_classifier = pipeline(
                "sentiment-analysis", 
                model=TEXT_CLASSIFIER_MODEL_ID, 
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True # Ensure pipeline returns all scores for text emotion
            )
        except Exception as e:
            logger.error(f"Error loading text emotion classifier: {e}")
            text_emotion_classifier = None
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
    audio_results = []
    try:
        # Load entire audio file
        audio_data, current_audio_sample_rate = librosa.load(input_file_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
        
        total_samples = len(audio_data)
        samples_per_chunk = int(AUDIO_CHUNK_SIZE * current_audio_sample_rate)
        
        for i in range(0, total_samples, samples_per_chunk):
            audio_chunk = audio_data[i : i + samples_per_chunk]
            
            if len(audio_chunk) < samples_per_chunk and i + samples_per_chunk < total_samples:
                padding_needed = samples_per_chunk - len(audio_chunk)
                audio_chunk = np.pad(audio_chunk, (0, padding_needed), mode='constant')
            elif len(audio_chunk) == 0:
                continue

            transcribed_text, text_emotion_data, audio_emotion_data = \
                process_audio_chunk_from_file(
                    audio_chunk, current_audio_sample_rate, 
                    whisper_model, text_emotion_classifier, 
                    audio_feature_extractor, audio_emotion_classifier
                )

            start_time_chunk_sec = i / current_audio_sample_rate
            end_time_chunk_sec = (i + len(audio_chunk)) / current_audio_sample_rate

            # NOTE: process_audio_chunk_from_file currently does not return full score dictionaries.
            # Placeholder empty dicts are used for 'text_emotion_full_scores' and 'audio_emotion_full_scores'.
            # This may need to be updated if process_audio_chunk_from_file is modified to provide them.
            audio_results.append({
                'start_time_sec': start_time_chunk_sec,
                'end_time_sec': end_time_chunk_sec,
                'transcribed_text': transcribed_text,
                'text_emotion': text_emotion_data,
                'audio_emotion': audio_emotion_data,
                'text_emotion_full_scores': {}, # Placeholder
                'audio_emotion_full_scores': {}  # Placeholder
            })
        
        logger.info(f"Finished processing audio file: {input_file_path}. Collected {len(audio_results)} audio segments.")
    except Exception as e:
        logger.error(f"Error processing audio file {input_file_path}: {e}", exc_info=True)
    return audio_results

def process_video(input_file_path):
    logger.info(f"Processing video from file: {input_file_path}")
    video_results = []
    cap = cv2.VideoCapture(input_file_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {input_file_path}")
        return video_results

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        current_facial_emotion = ("unknown", 0.0)
        current_facial_scores = {}
        
        try:                  
            facial_emotion_data, raw_emotion_scores = get_facial_emotion_from_frame(frame)

            if facial_emotion_data and facial_emotion_data[0] != "unknown" and facial_emotion_data[0] != "error":
                # facial_emotion_data is already (unified_emotion_label, confidence)
                # raw_emotion_scores is the dictionary of full scores
                current_facial_emotion = facial_emotion_data
                current_facial_scores = raw_emotion_scores
                logger.info(f"[{timestamp_sec:.2f}s] Frame {frame_idx}:  Facial emotion (filtered): {facial_emotion_data[0]} ({facial_emotion_data[1]:.2f})")
            elif facial_emotion_data: # Handle cases like ("unknown", 0.0) or ("error", 0.0) from get_facial_emotion_from_frame
                current_facial_emotion = facial_emotion_data
                current_facial_scores = raw_emotion_scores if raw_emotion_scores else {}
            # If facial_emotion_data is None, defaults remain ("unknown", 0.0) and {}

        except Exception as e:
            logger.debug(f"Error in facial analysis for frame {frame_idx} at {timestamp_sec:.2f}s: {e}")
            # Defaults current_facial_emotion = ("unknown", 0.0), current_facial_scores = {} are kept

        video_results.append({
            'timestamp_sec': timestamp_sec,
            'facial_emotion': current_facial_emotion,
            'facial_emotion_full_scores': current_facial_scores
        })
        frame_idx += 1
    
    cap.release()
    logger.info(f"Finished processing video file: {input_file_path}. Collected {len(video_results)} video frames.")
    return video_results


def process_media_file(input_file_path):
    """
    Processes a video or audio file for emotion analysis and prints results.
    """
    file_extension = os.path.splitext(input_file_path)[1].lower()
    
    load_models() # Ensure models are loaded
    
    video_results = []
    audio_results = []

    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        video_results = process_video(input_file_path)

        try:
            # Attempt to load audio from the video file to check if it exists
            # We don't need to store the data here if process_audio will load it again,
            # but it's a quick check. Alternatively, process_audio could take raw data.
            temp_audio_data, _ = librosa.load(input_file_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
            if len(temp_audio_data) > 0:
                logger.info(f"Video file {input_file_path} contains audio. Processing audio component.")
                audio_results = process_audio(input_file_path) # Process the same file for audio
            else:
                logger.info(f"Video file {input_file_path} does not contain a significant audio component.")
        except Exception as e:
            logger.warning(f"Could not load or process audio from video file {input_file_path}: {e}", exc_info=True)

    elif file_extension in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        audio_results = process_audio(input_file_path)

    else:
        logger.error(f"Unsupported file type: {file_extension} for {input_file_path}")
        return [], []

    logger.info(f"Media processing complete for {input_file_path}.")
    if video_results:
        logger.info(f"Collected {len(video_results)} video analysis results.")
    if audio_results:
        logger.info(f"Collected {len(audio_results)} audio analysis results.")
    
    # For now, just returning them. Further processing (like consistency analysis) would happen here or be passed on.
    return video_results, audio_results

def main(video_file_path): # Renamed parameter for clarity
    start_time = time.time()
    
    if not os.path.exists(video_file_path):
        logger.error(f"Error: Input media file '{video_file_path}' does not exist. Please update the path.")
        return

    with mlflow.start_run(run_name=f"Batch_Analysis_{os.path.basename(video_file_path)}_{time.strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_param("input_file", video_file_path)
        mlflow.log_param("analysis_type", "Offline Batch")

        video_results, audio_results = process_media_file(video_file_path)

        # TODO: Implement consistency analysis using video_results and audio_results
        # For now, we just log that the data is available.
        if video_results:
            logger.info(f"Main: Received {len(video_results)} video results for further analysis.")
        if audio_results:
            logger.info(f"Main: Received {len(audio_results)} audio results for further analysis.")

        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("processing_duration_seconds", duration)
        logger.info(f"Batch analysis completed in {duration:.2f} seconds.")
        logger.info(f"MLflow run logged. View with: 'mlflow ui'")

if __name__ == '__main__':

    media_file_path = "C:\\Users\\kingk\\OneDrive\\Documents\\Projects\\AffectLink\\data\\WIN_20250529_10_51_21_Pro.mp4"

    main(media_file_path)
