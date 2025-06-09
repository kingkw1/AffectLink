import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Standard Library Imports
import json
import logging
import math
import shutil
import tempfile
import threading
import time
import traceback
from collections import deque

# Third-Party Imports
import numpy as np
import torch
import whisper
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor

# Local Application/Library Specific Imports
from constants import FACIAL_TO_UNIFIED, SER_TO_UNIFIED, UNIFIED_EMOTIONS, TEXT_TO_UNIFIED
from audio_processor import audio_processing_loop, record_audio
from video_processor import process_video

# Suppress DeepFace logging
logging.getLogger('deepface').setLevel(logging.ERROR)

# Logger Configuration
# Configure root logger if no handlers are present, to avoid duplicate logging from libraries.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

# Global Variables
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
    'facial_emotion_history': deque(maxlen=60),
    'text_emotion_history': deque(maxlen=60),
    'ser_emotion_history': deque(maxlen=60)
}

whisper_model = None
text_classifier = None
audio_feature_extractor = None
audio_classifier = None
face_cascade = None # Typically loaded in process_video if needed there

last_audio_analysis = time.time() - 10  # Force initial analysis on first run

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
        return [convert_to_serializable(i) for i in obj]
    if isinstance(obj, deque):
        return [convert_to_serializable(i) for i in list(obj)]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    
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
    elif cosine_sim <= 0.01:
        return "Unknown" 
    else:
        return "Very Low"


def create_unified_emotion_vector(emotion_scores, mapping_dict):
    """
    Create a vector of scores in the unified emotion space
    """
    unified_vector = [0.0] * len(UNIFIED_EMOTIONS)
    
    for emotion, score in emotion_scores.items():
        if emotion in mapping_dict and mapping_dict[emotion] is not None:
            unified_emotion = mapping_dict[emotion]
            unified_index = UNIFIED_EMOTIONS.index(unified_emotion)
            unified_vector[unified_index] += score
            
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
                
    total = sum(unified_scores.values())
    if total > 0:
        for emotion in unified_scores:
            unified_scores[emotion] /= total
            
    return unified_scores

# Clear any existing frame and emotion files at startup
def clear_stale_files():
    """Delete any existing frame and emotion files to ensure a fresh start"""
    try:
        # Define paths
        frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")
        emotion_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
        
        if os.path.exists(frame_path):
            os.remove(frame_path)
            logger.debug(f"Deleted old frame file: {frame_path}") 

        if os.path.exists(emotion_path):
            os.remove(emotion_path)
            logger.debug(f"Deleted old emotion file: {emotion_path}")
            
    except Exception as e:
        logger.warning(f"Error clearing stale files: {e}")

# Clear stale files at module import time
clear_stale_files()

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
    
    similarity_fa = calculate_cosine_similarity(facial_vector, audio_vector)
    similarity_ft = calculate_cosine_similarity(facial_vector, text_vector)
    similarity_at = calculate_cosine_similarity(audio_vector, text_vector)
    
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
        overall_cosine_similarity = 0.0
    
    logger.debug(f"Cosine similarities: facial-audio={similarity_fa:.3f}, facial-text={similarity_ft:.3f}, audio-text={similarity_at:.3f}")
    logger.debug(f"Valid similarities count: {len(valid_similarities)}; Overall cosine similarity: {overall_cosine_similarity:.3f}")

    return overall_cosine_similarity

def load_models():
    global whisper_model, text_classifier, audio_feature_extractor, audio_classifier
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    whisper_cache_dir = os.path.join(project_root, "models", "whisper_cache")
    transformers_cache_dir = os.path.join(project_root, "models", "transformers_cache")

    os.makedirs(whisper_cache_dir, exist_ok=True)
    os.makedirs(transformers_cache_dir, exist_ok=True)
    logger.info(f"Whisper models will be cached in: {whisper_cache_dir}")
    logger.info(f"Transformers models will be cached in: {transformers_cache_dir}")
    
    logger.info("Initializing Whisper model...")
    try:
        whisper_model = whisper.load_model("base", download_root=whisper_cache_dir)
        if whisper_model is None:
            logger.error("Failed to initialize Whisper model")
            return None, None, None, None, None # Ensure consistent return on failure
        logger.info(f"Whisper model successfully loaded: {type(whisper_model).__name__} (base)")
        
        if hasattr(whisper_model, 'to') and torch.cuda.is_available():
            whisper_model = whisper_model.to("cuda")
            logger.info("Whisper model moved to CUDA")
    except Exception as e:
        logger.error(f"Error loading Whisper model: {e}")
        return None, None, None, None, None
    
    logger.info("Initializing text emotion classifier...")
    try:
        text_classifier = pipeline("text-classification", 
                                model="j-hartmann/emotion-english-distilroberta-base", 
                                top_k=None,
                                cache_dir=transformers_cache_dir
                                )
    except Exception as e:
        logger.error(f"Error loading text classifier: {e}")
        return None, None, None, None, None
    
    logger.info("Initializing audio emotion classifier...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device set to use {device}")
        
    ser_model_name = "superb/hubert-large-superb-er"
    try:      
        audio_feature_extractor = AutoFeatureExtractor.from_pretrained(ser_model_name, cache_dir=transformers_cache_dir)
        audio_classifier = AutoModelForAudioClassification.from_pretrained(ser_model_name, cache_dir=transformers_cache_dir)
        
        audio_classifier = audio_classifier.to(device)
        logger.info(f"Audio classifier ({ser_model_name}) successfully loaded and moved to {device}")

        if hasattr(audio_classifier, 'config') and hasattr(audio_classifier.config, 'id2label'):
            model_labels_dict = audio_classifier.config.id2label
            ser_actual_labels_from_config = [model_labels_dict[i] for i in sorted(model_labels_dict.keys())]
            logger.info(f"SER_MODEL_LABELS_AT_INIT: {ser_actual_labels_from_config}")
        else:
            logger.warning("SER_MODEL_LABELS_AT_INIT: Could not retrieve id2label from SER model config.")

    except Exception as e:
        logger.error(f"Error loading audio classifier: {e}")
        return None, None, None, None, None # Return None for all on failure
    
    return whisper_model, text_classifier, audio_feature_extractor, audio_classifier, device

def main(
        emotion_queue=None, 
        stop_event_param=None,
        camera_index=0, 
        use_whisper_api_toggle = True,
        use_ser_api_toggle = True,
        use_text_classifier_api_toggle = True
    ):

    """Main function for emotion detection."""
    global shared_state, whisper_model, text_classifier, audio_feature_extractor, audio_classifier, face_cascade
    
    logger.info(f"Detect_emotion main started with camera_index: {camera_index}")
        
    shared_state['emotion_queue'] = emotion_queue
    shared_state['stop_event'] = stop_event_param # Use renamed parameter
    
    if isinstance(camera_index, str):
        try:
            camera_index = int(camera_index)
        except ValueError:
            logger.error(f"Invalid camera_index: {camera_index}. Defaulting to 0.")
            camera_index = 0
    
    if shared_state['stop_event'] is None: # Check shared_state for stop_event
        shared_state['stop_event'] = threading.Event() # Use threading.Event if None
    
    if isinstance(shared_state['stop_event'], dict) and 'shared_frame_data' in shared_state['stop_event']:
        logger.info(f"Detector received shared frame queue: {shared_state['stop_event']['shared_frame_data']}")
    
    # Load models if not using all API toggles
    device = None # Initialize device
    if not (use_whisper_api_toggle and use_ser_api_toggle and use_text_classifier_api_toggle):
        # Models are loaded into global scope by load_models()
        _, _, _, _, device = load_models() # Assign return values, device is the last one
        if device is None: # Check if model loading failed
            logger.error("Model loading failed. Exiting main_processor.")
            return # Exit if models didn't load
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("All API toggles are True. Models will be managed by API calls.")

    video_started_event = threading.Event()
    
    logger.info("Starting audio recording thread...")
    audio_thread = threading.Thread(target=record_audio, args=(shared_state,))
    audio_thread.daemon = True
    audio_thread.start()
    
    time.sleep(1)
    
    audio_lock = threading.Lock()

    logger.info("Starting audio processing thread...")
    try:
        audio_processing_thread = threading.Thread(
            target=audio_processing_loop,
            args=(shared_state, audio_lock, 
                  whisper_model, text_classifier,
                  audio_classifier, audio_feature_extractor, 
                  device, video_started_event,
                  use_whisper_api_toggle,
                  use_ser_api_toggle,
                  use_text_classifier_api_toggle, 
                  )
        )
        audio_processing_thread.daemon = True
        audio_processing_thread.start()
        logger.info("Audio processing thread started successfully")
    except Exception as e:
        logger.error(f"Failed to start audio processing thread: {e}")
        logger.error(traceback.format_exc())
        return # Exit if audio processing thread fails to start

    logger.info("Starting video processing thread...")
    video_lock = threading.Lock()
    video_thread = threading.Thread(target=process_video, args=(shared_state, video_lock, video_started_event))
    video_thread.daemon = True
    video_thread.start()
    
    video_started_event.set() # Signal video has started
    
    logger.info("Starting main emotion analysis loop...")
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
            result_data = {
                "timestamp": time.time(),
                "facial_emotion": shared_state.get('facial_emotion', ("neutral", 0.0)),
                "text_emotion": shared_state.get('text_emotion', ("neutral", 0.0)),
                "audio_emotion": shared_state.get('audio_emotion', ("neutral", 0.0)),
                "overall_emotion": shared_state.get('overall_emotion', "neutral"),
                "transcribed_text": shared_state.get('transcribed_text', "Waiting for audio transcription..."),
                "facial_emotion_full_scores": shared_state.get('facial_emotion_full_scores', {}),
                "audio_emotion_full_scores": shared_state.get('audio_emotion_full_scores', []),
                "text_emotion_full_scores": shared_state.get('text_emotion_full_scores', []), # <-- ADD THIS LINE
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
            facial_emotion_data = shared_state.get('facial_emotion_full_scores', {})
            audio_emotion_data = shared_state.get('audio_emotion_full_scores', []) # This is a list of dicts
            text_emotion_data = shared_state.get('text_emotion_unified_scores', {}) # This is a dict

            # Create unified vectors
            facial_vector = create_unified_emotion_vector(facial_emotion_data, FACIAL_TO_UNIFIED)
            
            # For audio, convert list of dicts to dict for create_unified_emotion_vector
            audio_scores_dict = {item['emotion']: item['score'] for item in audio_emotion_data if isinstance(item, dict) and 'emotion' in item and 'score' in item}
            audio_vector = create_unified_emotion_vector(audio_scores_dict, SER_TO_UNIFIED)
            
            # For text, retrieve the full emotion scores list from shared_state
            text_emotion_data_for_vector = shared_state.get('text_emotion_full_scores', [])
            text_scores_dict = {
                (item.get('emotion') or item.get('label')): item.get('score', 0.0) 
                for item in text_emotion_data_for_vector 
                if isinstance(item, dict) and ((item.get('emotion') or item.get('label')) is not None) and item.get('score') is not None
            }
            text_vector = create_unified_emotion_vector(text_scores_dict, TEXT_TO_UNIFIED)

            # Get overall cosine similarity
            overall_cosine_similarity = calculate_average_multimodal_similarity(facial_vector, audio_vector, text_vector)

            result_data["cosine_similarity"] = overall_cosine_similarity
            result_data["consistency_level"] = get_consistency_level(overall_cosine_similarity)
            
            # Determine overall dominant emotion (simple majority or weighted average of vectors)
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
            temp_emotion_file_path = os.path.join(tempfile.gettempdir(), f"affectlink_emotion_tmp_{os.getpid()}_{time.time_ns()}.json")

            try:
                with open(temp_emotion_file_path, 'w') as f:
                    json.dump(serializable_result_data, f, indent=4)
                
                shutil.move(temp_emotion_file_path, emotion_file_path)
            except TypeError as e:
                logger.error(f"Error saving emotion data to file (TypeError): {e}")
            except Exception as e:
                logger.error(f"Unexpected error saving emotion data: {e}")
                if os.path.exists(temp_emotion_file_path):
                    try:
                        os.remove(temp_emotion_file_path)
                    except Exception as cleanup_err:
                        logger.error(f"Error cleaning up temporary emotion file: {cleanup_err}")
            
            # Send data to queue if it exists
            if shared_state.get('emotion_queue') is not None:
                try:
                    shared_state['emotion_queue'].put(serializable_result_data)
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
    # This part is typically called by start_realtime.py or run_affectlink.py as a subprocess
    # or if the script is run directly for testing.
    
    # Example of direct invocation (for testing purposes)
    # Create a dummy stop event (threading.Event) if running standalone
    stop_event_main = threading.Event()
    
    # For direct testing, you might not have a queue from another process.
    # main(stop_event=stop_event_main, camera_index=0)
    
    # The actual entry point when run via multiprocessing in run_affectlink.py doesn't need this __main__ block
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
