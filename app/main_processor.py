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
import random
import json
import shutil
import traceback # Moved from various functions
import torch
import whisper
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import logging
import threading
import numpy as np

from app.constants import FACIAL_TO_UNIFIED, SER_TO_UNIFIED, UNIFIED_EMOTIONS
from audio_emotion_processor import audio_processing_loop, record_audio
from video_emotion_processor import process_video

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
model = None

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

def main(emotion_queue=None, stop_event=None, camera_index=0):
    """Main function for emotion detection."""
    global shared_state, model, text_classifier, audio_feature_extractor, audio_classifier, face_cascade
    global ser_model, ser_processor # Ensure these are global if accessed directly
    
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
    
    # Initialize the Whisper model for transcription
    print("Initializing Whisper model...")
    try:
        # Use base model instead of tiny for better accuracy
        model = whisper.load_model("base")
        # Verify whisper model loaded correctly
        if model is None:
            logger.error("Failed to initialize Whisper model")
            return
        logger.info(f"Whisper model successfully loaded: {type(model).__name__} (base)")
        
        # Move model to CUDA if available
        if hasattr(model, 'to') and torch.cuda.is_available():
            model = model.to("cuda")
            logger.info("Whisper model moved to CUDA")
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        return  # Exit if we can't load the model
    
    # Initialize text emotion classifier
    print("Initializing text emotion classifier...")
    try:
        text_classifier = pipeline("text-classification", 
                                model="j-hartmann/emotion-english-distilroberta-base", 
                                top_k=None)
    except Exception as e:
        logger.error(f"Error loading text classifier: {e}")
        return  # Exit if we can't load the classifier
    
    # Initialize audio emotion classifier - using a more accessible model
    print("Initializing audio emotion classifier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device set to use {device}")
        
    ser_model_name = "superb/hubert-large-superb-er" # Correct SER model
    try:
        # audio_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name) # Original
        # audio_classifier = AutoModelForAudioClassification.from_pretrained(model_name) # Original
        
        audio_feature_extractor = AutoFeatureExtractor.from_pretrained(ser_model_name)
        audio_classifier = AutoModelForAudioClassification.from_pretrained(ser_model_name)
        
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
    
    # Event for synchronization between video and audio threads
    video_started_event = threading.Event()

    # --- START AUDIO PROCESSING COMPONENTS ---
    
    # Start audio recording thread first (collects audio data)
    print("Starting audio recording thread...")
    audio_thread = threading.Thread(target=record_audio)
    audio_thread.daemon = True
    audio_thread.start()
    
    # Wait a moment for the audio thread to initialize
    time.sleep(1)
    
    # Create audio analysis data structures
    audio_emotion_log = []
    audio_lock = threading.Lock()
    
    # Start audio processing thread with error handling
    print("Starting audio processing thread...")
    try:
        audio_processing_thread = threading.Thread(
            target=audio_processing_loop,
            args=(audio_emotion_log, audio_lock, shared_state['stop_event'], 
                  model, text_classifier, 
                  audio_classifier, audio_feature_extractor, 
                  device, video_started_event) # REMOVED list(SER_TO_UNIFIED.keys())
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
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
    # Set the event to signal video has started
    video_started_event.set()
    
    # Main loop for emotion analysis
    print("Starting main emotion analysis loop...")
    try:
        while True:
            # Check if we need to stop
            if isinstance(shared_state['stop_event'], dict):
                if shared_state['stop_event'].get('stop', False):
                    print("Stop signal received in main loop")
                    break
            elif shared_state['stop_event'] and shared_state['stop_event']:
                print("Stop event detected in main loop")
                break
            
            # Store results for consistency checking
            latest = {}
            
            # Process the latest frame for facial emotion if available
            if shared_state['latest_frame'] is not None:
                # We already processed facial emotions in the video thread
                facial_emotion, confidence = shared_state['facial_emotion']
                latest['facial_emotion'] = facial_emotion
                latest['facial_confidence'] = confidence
                
                print(f"Facial emotion: {facial_emotion} ({confidence:.2f})")
            
            # Print audio emotion if available for debugging
            audio_emotion, audio_score = shared_state['audio_emotion']
            if audio_emotion and audio_emotion != "neutral":
                print(f"Audio emotion: {audio_emotion} ({audio_score:.2f})")
                
            # Print text emotion if available for debugging
            text_emotion, text_score = shared_state['text_emotion']
            if shared_state['transcribed_text'] and text_emotion != "neutral":
                print(f"Text emotion: {text_emotion} ({text_score:.2f}) - '{shared_state['transcribed_text'][:30]}...'")
            
            # --- Calculate Facial/Audio Consistency ---
            logger.info("HARDCODED_TEST_LOG: Inside consistency block") # ADDED FOR DEBUGGING
            facial_full_scores_dict = shared_state.get('facial_emotion_full_scores', {})
            audio_full_scores_list = shared_state.get('audio_emotion_full_scores', [])

            # Convert audio scores list to dict format if it's a list of dicts
            # Expected format for audio_full_scores_list: [{'emotion': 'neu', 'score': 0.7}, ...]
            audio_scores_dict = {e['emotion']: e['score'] for e in audio_full_scores_list if isinstance(e, dict) and 'emotion' in e and 'score' in e}

            logger.info(f"CONSISTENCY_DEBUG: Facial Full Scores: {facial_full_scores_dict}")
            logger.info(f"CONSISTENCY_DEBUG: Audio Full Scores (List from shared_state): {audio_full_scores_list}")
            logger.info(f"CONSISTENCY_DEBUG: Audio Scores (Dict for vectorization): {audio_scores_dict}")

            facial_unified_vector = create_unified_emotion_vector(facial_full_scores_dict, FACIAL_TO_UNIFIED)
            audio_unified_vector = create_unified_emotion_vector(audio_scores_dict, SER_TO_UNIFIED)
            
            logger.info(f"CONSISTENCY_DEBUG: Facial Unified Vector: {facial_unified_vector}")
            logger.info(f"CONSISTENCY_DEBUG: Audio Unified Vector: {audio_unified_vector}")
            
            calculated_cosine_similarity = calculate_cosine_similarity(facial_unified_vector, audio_unified_vector)
            
            logger.info(f"CONSISTENCY_DEBUG: Calculated Cosine Similarity: {calculated_cosine_similarity}")

            # Combine all current emotional data
            PLACEHOLDER_TEXT = "Waiting for audio transcription..."
            transcribed_text_to_send = shared_state['transcribed_text']
            if transcribed_text_to_send == PLACEHOLDER_TEXT:
                transcribed_text_to_send = ""  # Do not send placeholder to UI/queue

            result_data = {
                "facial_emotion": shared_state['facial_emotion'],
                "text_emotion": shared_state['text_emotion'],
                "audio_emotion": shared_state['audio_emotion'],
                "transcribed_text": transcribed_text_to_send,
                "overall_emotion": shared_state['overall_emotion'],
                "cosine_similarity": calculated_cosine_similarity,
                "facial_emotion_full_scores": shared_state['facial_emotion_full_scores'],
                "audio_emotion_full_scores": shared_state['audio_emotion_full_scores'],
                "facial_emotion_history": list(shared_state['facial_emotion_history']),
                "text_emotion_history": list(shared_state['text_emotion_history']),
                "ser_emotion_history": list(shared_state['ser_emotion_history']),
                "update_id": f"update_{time.time()}" 
            }
            logger.info(f"CONSISTENCY_DEBUG: Data for JSON/Queue (cosine_similarity): {result_data.get('cosine_similarity')}")
            
            # Add timestamps to transcriptions to ensure they update in UI
            if transcribed_text_to_send:
                result_data["transcribed_text"] = f"{transcribed_text_to_send} [{time.time():.3f}]"
            else:
                result_data["transcribed_text"] = ""
            
            # First try sending via queue if available
            if shared_state['emotion_queue'] is not None:
                try:
                    # Send without blocking
                    shared_state['emotion_queue'].put(result_data, block=False)
                    logger.info(f"Sent emotion data to queue with text: '{result_data['transcribed_text'][:30]}...'")
                except Exception as e:
                    logger.error(f"Error sending emotion data to queue: {e}")
                
                # Always save to file for dashboard to access - we want to ensure updates
                try:
                    # Add a unique identifier to each save to ensure file changes are detected
                    result_data["update_id"] = f"{time.time():.6f}-{random.randint(1000, 9999)}"
                    
                    # Create a JSON-serializable version of the data
                    serializable_data = {}
                    
                    # Convert any NumPy values to standard Python types
                    for key, value in result_data.items():
                        if key in ["facial_emotion", "text_emotion", "audio_emotion"]:
                            # These are tuples with emotion name and confidence
                            emotion_name, confidence = value
                            # Convert any NumPy float to Python float
                            if isinstance(confidence, np.number):
                                confidence = float(confidence)
                            # Store as a dictionary to preserve structure in JSON
                            serializable_data[key] = {"emotion": emotion_name, "confidence": confidence}
                        elif key == "facial_emotion_full_scores": # ADDED THIS BLOCK
                            # This is a dict, convert its values (scores) to float
                            serializable_data[key] = {e_name: float(e_score) if isinstance(e_score, np.number) else e_score 
                                                      for e_name, e_score in value.items()}
                        elif isinstance(value, np.number): # Handles other top-level np.number types
                            serializable_data[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            serializable_data[key] = value.tolist()
                        else:
                            serializable_data[key] = value
                    
                    # Add timestamp to the data
                    serializable_data["timestamp"] = float(time.time())
                    
                    # Also directly copy transcribed_text to ensure it's included
                    serializable_data["transcribed_text"] = result_data["transcribed_text"]
                    serializable_data["update_id"] = result_data["update_id"]
                    
                    # Log the data being saved
                    logger.debug(f"Saving emotion data to file with text: '{serializable_data['transcribed_text'][:30]}...'")
                    
                    # Save to temp file using a temporary file approach to ensure atomic writes
                    emotion_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
                    temp_path = f"{emotion_path}.tmp"
                    
                    # First write to a temp file, then rename to the final path
                    with open(temp_path, 'w') as f:
                        json.dump(serializable_data, f)
                        f.flush()
                        os.fsync(f.fileno())  # Ensure data is written to disk
                        
                    # Rename for atomic replacement
                    shutil.move(temp_path, emotion_path)

                    # Log success
                    logger.debug(f"Successfully wrote emotion data to {emotion_path}")
                except Exception as e:
                    logger.error(f"Error saving emotion data to file: {e}")
                    
            # Sleep to avoid high CPU usage
            time.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        # Signal threads to stop
        if isinstance(shared_state['stop_event'], dict):
            shared_state['stop_event']['stop'] = True
        print("Waiting for threads to finish...")
        # Wait for threads to finish gracefully
        time.sleep(2)
