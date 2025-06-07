#!/usr/bin/env python3
"""
run_app.py - Main script for AffectLink with improved frame sharing
between the emotion detection and dashboard processes.
This script sets up the environment and starts the dashboard and emotion detection processes.
It handles device detection and initialization.
Usage:
    python run_app.py
"""
import time # Added for timestamps
from collections import deque
import os
import tempfile
import math
import json
import shutil
import traceback # Moved from various functions
import torch
import whisper
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import logging
import threading
import numpy as np

from constants import FACIAL_TO_UNIFIED, SER_TO_UNIFIED, UNIFIED_EMOTIONS, TEXT_TO_UNIFIED # Ensure TEXT_TO_UNIFIED is imported if used by convert_to_serializable or related logic
from audio_processor import audio_processing_loop, record_audio
from video_processor import process_video

# Suppress DeepFace logging for cleaner console output
logging.getLogger('deepface').setLevel(logging.ERROR)

# Logger configuration
# Ensure logger is configured only once if not already configured by root.
if not logging.getLogger().handlers: # Check root logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

def convert_to_serializable(obj):
    """
    Recursively converts an object to ensure it's JSON serializable.
    Handles numpy types, deques, and common collections.
    """
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    if isinstance(obj, tuple):
        # Convert tuples to lists as JSON doesn't have a tuple type,
        # and lists are generally more common for sequences in JSON.
        return [convert_to_serializable(i) for i in obj]
    if isinstance(obj, deque):
        return [convert_to_serializable(i) for i in list(obj)] # Convert deque to list
    # Pass through types that are already JSON serializable
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
    # For any other unhandled types, log a warning and attempt to convert to string as a fallback.
    # This might not be ideal for all types but can prevent outright crashes.
    logger.warning(f"Unhandled type for serialization: {type(obj)}. Converting to string.")
    return str(obj)

def get_consistency_level(cosine_sim):
    """Convert cosine similarity to consistency level label"""
    if cosine_sim >= 0.8:
        return "High"
    elif cosine_sim >= 0.6:
        return "Medium"
    elif cosine_sim >= 0.3:
        return "Low"
    elif cosine_sim <= 0.01: # Match dashboard's "Unknown" condition
        return "Unknown" 
    else: # Covers 0.01 < cosine_sim < 0.3
        return "Very Low"

def calculate_cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors
    """
    if len(vector_a) != len(vector_b):
        return 0.0
        
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
        
    return dot_product / (magnitude_a * magnitude_b)


def create_unified_emotion_vector(emotion_scores, mapping_dict):
    """
    Create a vector of scores in the unified emotion space
    """
    unified_vector = [0.0] * len(UNIFIED_EMOTIONS)
    
    for emotion, score in emotion_scores.items():
        # Map to unified emotion if possible
        if emotion in mapping_dict and mapping_dict[emotion] is not None:
            unified_emotion = mapping_dict[emotion]
            unified_index = UNIFIED_EMOTIONS.index(unified_emotion)
            unified_vector[unified_index] += score
            
    # Normalize
    total = sum(unified_vector)
    if total > 0:
        unified_vector = [score/total for score in unified_vector]
        
    return unified_vector

def create_unified_emotion_dict(emotion_scores, mapping_dict):
    """
    Create a dictionary of unified emotion scores
    """
    unified_scores = {emotion: 0.0 for emotion in UNIFIED_EMOTIONS}
    
    for emotion, score in emotion_scores.items():
        if emotion in mapping_dict and mapping_dict[emotion] is not None:
            unified_emotion = mapping_dict[emotion]
            if unified_emotion in unified_scores:
                unified_scores[unified_emotion] += score
                
    # Normalize
    total = sum(unified_scores.values())
    if total > 0:
        for emotion in unified_scores:
            unified_scores[emotion] /= total
            
    return unified_scores

# Suppress DeepFace logging for cleaner console output
logging.getLogger('deepface').setLevel(logging.ERROR) # Changed to target 'deepface' specifically

# Add logging for debug information
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('detect_emotion')

# Global variables
shared_state = {
    'emotion_queue': None,
    'stop_event': None,
    'latest_audio': None,
    'latest_frame': None,
    'transcribed_text': "Waiting for audio transcription...",
    'facial_emotion': ("neutral", 0.0),
    'audio_emotion': ("neutral", 0.0),
    'text_emotion': ("neutral", 0.0),
    'overall_emotion': "neutral",
    'facial_emotion_full_scores': {},
    'audio_emotion_full_scores': [],
    'facial_emotion_history': deque(maxlen=60), # Added for historical tracking
    'text_emotion_history': deque(maxlen=60),   # Added for historical tracking
    'ser_emotion_history': deque(maxlen=60)     # Added for historical tracking
}

# Reusable whisper model instance
whisper_model = None

# Text emotion classification pipeline
text_classifier = None

# Audio emotion classification model
audio_feature_extractor = None
audio_classifier = None

last_audio_analysis = time.time() - 10  # Force initial analysis

# Face detection and emotion recognition
face_cascade = None

# Logger configuration
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Clear any existing frame and emotion files at startup to prevent the dashboard
# from loading stale data from previous sessions
def clear_stale_files():
    """Delete any existing frame and emotion files to ensure a fresh start"""
    try:
        # Define paths
        frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")
        emotion_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
        
        # Delete frame file if it exists
        if os.path.exists(frame_path):
            os.remove(frame_path)
            logger.info(f"Deleted old frame file: {frame_path}")
        
        # Delete emotion file if it exists
        if os.path.exists(emotion_path):
            os.remove(emotion_path)
            logger.info(f"Deleted old emotion file: {emotion_path}")
    except Exception as e:
        logger.warning(f"Error clearing stale files: {e}")

# Clear stale files at module import time
clear_stale_files()

def calculate_average_multimodal_similarity(facial_vector, audio_vector, text_vector):
    """
    Calculate average cosine similarity across three modalities:
    facial, audio, and text.
    Returns the average similarity score.
    
    Args:
        facial_vector (list): Vector of facial emotion scores.
        audio_vector (list): Vector of audio emotion scores.
        text_vector (list): Vector of text emotion scores.
    Returns:
        float: Overall cosine similarity score across modalities.
        """
    
    # Calculate pairwise cosine similarities
    similarity_fa = calculate_cosine_similarity(facial_vector, audio_vector)
    similarity_ft = calculate_cosine_similarity(facial_vector, text_vector)
    similarity_at = calculate_cosine_similarity(audio_vector, text_vector)
    
    # Average similarity (or other combination logic)
    # Consider only valid similarities (e.g. if a modality is not present, its vector might be all zeros)
    valid_similarities = []
    if facial_vector != [0.0] * len(UNIFIED_EMOTIONS) and audio_vector != [0.0] * len(UNIFIED_EMOTIONS):
        valid_similarities.append(similarity_fa)
    if facial_vector != [0.0] * len(UNIFIED_EMOTIONS) and text_vector != [0.0] * len(UNIFIED_EMOTIONS):
        valid_similarities.append(similarity_ft)
    if audio_vector != [0.0] * len(UNIFIED_EMOTIONS) and text_vector != [0.0] * len(UNIFIED_EMOTIONS):
        valid_similarities.append(similarity_at)

    if valid_similarities:
        overall_cosine_similarity = sum(valid_similarities) / len(valid_similarities)
    else:
        overall_cosine_similarity = 0.0 # Default if no valid pairs
        
    return overall_cosine_similarity

def load_models():
    
    # Define project root and model cache directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # This assumes 'app' is one level down from project root
    
    whisper_cache_dir = os.path.join(project_root, "models", "whisper_cache")
    transformers_cache_dir = os.path.join(project_root, "models", "transformers_cache")

    # Create cache directories if they don't exist
    os.makedirs(whisper_cache_dir, exist_ok=True)
    os.makedirs(transformers_cache_dir, exist_ok=True)
    logger.info(f"Whisper models will be cached in: {whisper_cache_dir}")
    logger.info(f"Transformers models will be cached in: {transformers_cache_dir}")
    
    # Initialize the Whisper model for transcription
    print("Initializing Whisper model...")
    try:
        # Use base model instead of tiny for better accuracy
        whisper_model = whisper.load_model("base", download_root=whisper_cache_dir)
        # Verify whisper model loaded correctly
        if whisper_model is None:
            logger.error("Failed to initialize Whisper model")
            return
        logger.info(f"Whisper model successfully loaded: {type(whisper_model).__name__} (base)")
        
        # Move model to CUDA if available
        if hasattr(whisper_model, 'to') and torch.cuda.is_available():
            whisper_model = whisper_model.to("cuda")
            logger.info("Whisper model moved to CUDA")
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        return  # Exit if we can't load the model
    
    # Initialize text emotion classifier
    print("Initializing text emotion classifier...")
    try:
        text_classifier = pipeline("text-classification", 
                                model="j-hartmann/emotion-english-distilroberta-base", 
                                top_k=None,
                                # cache_dir=transformers_cache_dir
                                )
    except Exception as e:
        logger.error(f"Error loading text classifier: {e}")
        return  # Exit if we can't load the classifier
    
    # Initialize audio emotion classifier - using a more accessible model
    print("Initializing audio emotion classifier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to use {device}")
        
    ser_model_name = "superb/hubert-large-superb-er" # Correct SER model
    try:      
        audio_feature_extractor = AutoFeatureExtractor.from_pretrained(ser_model_name, cache_dir=transformers_cache_dir)
        audio_classifier = AutoModelForAudioClassification.from_pretrained(ser_model_name, cache_dir=transformers_cache_dir)
        
        # Make sure both are on the same device
        audio_classifier = audio_classifier.to(device)
        logger.info(f"Audio classifier ({ser_model_name}) successfully loaded and moved to {device}")

        # Log the labels from the loaded SER model config
        if hasattr(audio_classifier, 'config') and hasattr(audio_classifier.config, 'id2label'):
            model_labels_dict = audio_classifier.config.id2label
            ser_actual_labels_from_config = [model_labels_dict[i] for i in sorted(model_labels_dict.keys())]
            logger.info(f"SER_MODEL_LABELS_AT_INIT: {ser_actual_labels_from_config}")
        else:
            logger.warning("SER_MODEL_LABELS_AT_INIT: Could not retrieve id2label from SER model config.")

    except Exception as e:
        logger.error(f"Error loading audio classifier: {e}")
    
    return whisper_model, text_classifier, audio_feature_extractor, audio_classifier, device

def main(emotion_queue=None, stop_event=None, camera_index=0):
    """Main function for emotion detection."""
    global shared_state, whisper_model, text_classifier, audio_feature_extractor, audio_classifier, face_cascade
    # Ensure ser_model and ser_processor are correctly scoped or passed if needed by audio_processing_loop
    # Depending on their initialization, they might need to be global or passed differently.
    
    logger.info(f"Detect_emotion main started with camera_index: {camera_index}")
        
    # Store the queue and stop event
    shared_state['emotion_queue'] = emotion_queue
    shared_state['stop_event'] = stop_event
    
    # Convert int/str camera_index to int
    if isinstance(camera_index, str):
        camera_index = int(camera_index)
    
    # Create a dictionary to hold the stop flag if not provided
    stop_flag = {'stop': False}
    if stop_event is None:
        shared_state['stop_event'] = stop_flag
    
    # Log if we have access to a frame queue
    if isinstance(stop_event, dict) and 'shared_frame_data' in stop_event:
        print(f"Detector received shared frame queue: {stop_event['shared_frame_data']}")
    
    # Load models
    whisper_model, text_classifier, audio_feature_extractor, audio_classifier, device = load_models()

    # Event for synchronization between video and audio threads
    video_started_event = threading.Event()
    
    # Start audio recording thread first (collects audio data)
    print("Starting audio recording thread...")
    audio_thread = threading.Thread(target=record_audio, args=(shared_state,))
    audio_thread.daemon = True
    audio_thread.start()
    
    # Wait a moment for the audio thread to initialize
    time.sleep(1)
    
    # Create audio analysis data structures
    # audio_emotion_log = [] # Removed, now part of shared_state
    audio_lock = threading.Lock()

    use_whisper_api_toggle = True # Set to True to use API, False for local
    use_ser_api_toggle = True # Set to True to use API, False for local
    use_text_classifier_api_toggle = True # <--- NEW TOGGLE: Set to True to use API, False for local

    # Start audio processing thread with error handling
    print("Starting audio processing thread...")
    try:
        audio_processing_thread = threading.Thread(
            target=audio_processing_loop,
            args=(shared_state, audio_lock, 
                  whisper_model, text_classifier, # text_emotion_classifier is still passed for local use
                  audio_classifier, audio_feature_extractor, 
                  device, video_started_event,
                  use_whisper_api_toggle,
                  use_ser_api_toggle,
                  use_text_classifier_api_toggle, 
                  )
        )
        audio_processing_thread.daemon = True
        audio_processing_thread.start()
        print("Audio processing thread started successfully")
    except Exception as e:
        logger.error(f"Failed to start audio processing thread: {e}")
        logger.error(traceback.format_exc())
    
    # --- END AUDIO PROCESSING COMPONENTS ---

    # Start video processing thread
    print("Starting video processing thread...")
    video_lock = threading.Lock() # Create a lock for video processing
    video_thread = threading.Thread(target=process_video, args=(shared_state, video_lock, video_started_event))
    video_thread.daemon = True
    video_thread.start()
    
    # Set the event to signal video has started
    video_started_event.set()
    
    # Main loop for emotion analysis
    print("Starting main emotion analysis loop...")
    try:
        while True:
            # Check if we need to stop
            stop_requested = False
            if isinstance(shared_state.get('stop_event'), dict):
                if shared_state['stop_event'].get('stop', False):
                    stop_requested = True
            elif hasattr(shared_state.get('stop_event'), 'is_set') and shared_state['stop_event'].is_set():
                stop_requested = True
            
            if stop_requested:
                logger.info("Stop signal received, exiting main emotion analysis loop.")
                break
            
            # Prepare data for JSON
            # Ensure all relevant shared_state items are included
            result_data = {
                "timestamp": time.time(),
                "facial_emotion": shared_state.get('facial_emotion', ("neutral", 0.0)),
                "text_emotion": shared_state.get('text_emotion', ("neutral", 0.0)),
                "audio_emotion": shared_state.get('audio_emotion', ("neutral", 0.0)),
                "overall_emotion": shared_state.get('overall_emotion', "neutral"),
                "transcribed_text": shared_state.get('transcribed_text', "Waiting for audio transcription..."),
                "facial_emotion_full_scores": shared_state.get('facial_emotion_full_scores', {}),
                "audio_emotion_full_scores": shared_state.get('audio_emotion_full_scores', []),
                "text_emotion_smoothed": shared_state.get('text_emotion_smoothed', ("unknown", 0.0)),
                "audio_emotion_smoothed": shared_state.get('audio_emotion_smoothed', ("unknown", 0.0)),
                "facial_emotion_history": list(shared_state.get('facial_emotion_history', [])), # deque to list
                "text_emotion_history": list(shared_state.get('text_emotion_history', [])),     # deque to list
                "ser_emotion_history": list(shared_state.get('ser_emotion_history', [])),       # deque to list
                "text_emotion_unified_scores": shared_state.get('text_emotion_unified_scores', {}), # Add this line
                "cosine_similarity": 0.0, # Placeholder, will be calculated next
                "consistency_level": "Unknown" # Placeholder
            }
            
            # Calculate overall emotion and cosine similarity
            # Ensure all emotion data is available and in the correct format
            facial_emotion_data = shared_state.get('facial_emotion_full_scores', {})
            audio_emotion_data = shared_state.get('audio_emotion_full_scores', []) # This is a list of dicts
            text_emotion_data = shared_state.get('text_emotion_unified_scores', {}) # This is a dict

            # Create unified vectors
            facial_vector = create_unified_emotion_vector(facial_emotion_data, FACIAL_TO_UNIFIED)
            
            # For audio, convert list of dicts to dict for create_unified_emotion_vector
            audio_scores_dict = {item['emotion']: item['score'] for item in audio_emotion_data if isinstance(item, dict) and 'emotion' in item and 'score' in item}
            audio_vector = create_unified_emotion_vector(audio_scores_dict, SER_TO_UNIFIED)
            
            # Text data is already in unified format from audio_emotion_processor
            # but create_unified_emotion_vector expects a mapping, so we pass an identity-like mapping or ensure it's already a vector
            # For simplicity, if text_emotion_data is already a unified score dict, we can use it directly if it matches UNIFIED_EMOTIONS order
            # Or, ensure create_unified_emotion_vector can handle it or pre-process it.
            # Assuming text_emotion_data is {unified_emotion: score, ...}
            text_vector = [text_emotion_data.get(emotion, 0.0) for emotion in UNIFIED_EMOTIONS]
            # Normalize text_vector if not already normalized
            text_total = sum(text_vector)
            if text_total > 0:
                text_vector = [score/text_total for score in text_vector]

            # Get overall cosine similarity
            overall_cosine_similarity = calculate_average_multimodal_similarity(facial_vector, audio_vector, text_vector)

            result_data["cosine_similarity"] = overall_cosine_similarity
            result_data["consistency_level"] = get_consistency_level(overall_cosine_similarity)
            
            # Determine overall dominant emotion (simple majority or weighted average of vectors)
            # For now, let's average the vectors and find the max component
            if facial_vector and audio_vector and text_vector: # Ensure all vectors are non-empty
                avg_vector = [
                    (f + a + t) / 3 
                    for f, a, t in zip(facial_vector, audio_vector, text_vector)
                ]
                if any(avg_vector): # Check if avg_vector is not all zeros
                    overall_emotion_index = avg_vector.index(max(avg_vector))
                    result_data["overall_emotion"] = UNIFIED_EMOTIONS[overall_emotion_index]
                    shared_state['overall_emotion'] = UNIFIED_EMOTIONS[overall_emotion_index] # Update shared_state as well
                else:
                    result_data["overall_emotion"] = "neutral" # Default if vectors are zero
                    shared_state['overall_emotion'] = "neutral"
            else:
                 result_data["overall_emotion"] = "neutral" # Default if any vector is missing
                 shared_state['overall_emotion'] = "neutral"

            # Convert the entire result_data structure to be JSON serializable
            serializable_result_data = convert_to_serializable(result_data)

            # Save to JSON file
            emotion_file_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
            # Use a unique temporary file name to prevent race conditions or partial writes
            temp_emotion_file_path = os.path.join(tempfile.gettempdir(), f"affectlink_emotion_tmp_{os.getpid()}_{time.time_ns()}.json")

            try:
                with open(temp_emotion_file_path, 'w') as f:
                    json.dump(serializable_result_data, f, indent=4)
                
                # Atomically move/rename the temporary file to the final destination
                shutil.move(temp_emotion_file_path, emotion_file_path)
                # logger.debug(f"Emotion data saved to {emotion_file_path}") # Debug level might be more appropriate
            except TypeError as e:
                logger.error(f"Error saving emotion data to file (TypeError): {e}")
                # To aid debugging, log the problematic data structure if the error persists.
                # Be cautious as this could be very verbose.
                # logger.error(f"Data causing error (first level): {{k: type(v) for k,v in serializable_result_data.items()}}")
            except Exception as e:
                logger.error(f"Unexpected error saving emotion data: {e}")
                # If temp_emotion_file_path was created, try to clean it up
                if os.path.exists(temp_emotion_file_path):
                    try:
                        os.remove(temp_emotion_file_path)
                    except Exception as cleanup_err:
                        logger.error(f"Error cleaning up temporary emotion file: {cleanup_err}")
            
            # Send data to queue if it exists
            if shared_state.get('emotion_queue') is not None:
                try:
                    # Send the serializable data to the queue as well
                    shared_state['emotion_queue'].put(serializable_result_data)
                    # logger.debug("Sent emotion data to queue.") # Debug level
                except Exception as q_err:
                    logger.warning(f"Could not put emotion data to queue: {q_err}")

            time.sleep(0.1) # Main loop processing interval
            
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main_processor. Exiting.")
    except Exception as e:
        logger.error(f"Critical error in main_processor main loop: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Main_processor cleaning up and stopping threads...")
        # Signal threads to stop
        if isinstance(shared_state.get('stop_event'), dict):
            shared_state['stop_event']['stop'] = True
        elif hasattr(shared_state.get('stop_event'), 'set'): # Check if it's an Event-like object
            shared_state['stop_event'].set()
        
        # Wait for threads to finish (optional, as they are daemons)
        # Giving a brief moment for daemon threads to attempt cleanup if they have try/finally blocks
        time.sleep(1) 
        logger.info("Main_processor finished.")

if __name__ == '__main__':
    # This part is typically called by start_realtime.py or run_app.py as a subprocess
    # or if the script is run directly for testing.
    
    # Example of direct invocation (for testing purposes)
    # Create a dummy stop event (threading.Event) if running standalone
    stop_event_main = threading.Event()
    
    # For direct testing, you might not have a queue from another process.
    # main(stop_event=stop_event_main, camera_index=0)
    
    # The actual entry point when run via multiprocessing in run_app.py doesn't need this __main__ block
    # to call main() again, as the target function for the process is main().
    # However, if this script is intended to be runnable standalone for some reason:
    logger.info("main_processor.py executed directly (likely for testing or unintended use).")
    # Consider what should happen if run directly. For now, just log.
    # If you need to parse arguments for camera_index when run directly:
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--camera_index", type=int, default=0)
    # args = parser.parse_args()
    # main(stop_event=stop_event_main, camera_index=args.camera_index)
