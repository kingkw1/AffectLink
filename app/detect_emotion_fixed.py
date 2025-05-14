import os
import tempfile
import wave
from collections import deque
import time
import math
import soundfile as sf
import torch
import whisper
from transformers import pipeline, AutoModelForAudioClassification
import sounddevice as sd
import librosa
from transformers import AutoFeatureExtractor
import cv2
from deepface import DeepFace
import logging
import threading
from moviepy import VideoFileClip
import numpy as np

# Import our video processing helper
from video_module_loader import get_video_processing_function

# Output verbosity control
VERBOSE_OUTPUT = False

# Constants
VIDEO_WINDOW_DURATION = 5  # seconds
AUDIO_WINDOW_DURATION = 5  # seconds
CAMERA_INDEX = 0  # Default camera index, can be overridden

# Emotion categories for unified mapping
UNIFIED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry']

# Import the original emotion detection module for everything else
from detect_emotion import (
    transcribe_audio_whisper, classify_emotion, classify_emotion_full,
    analyze_audio_emotion, analyze_audio_emotion_full, record_audio_chunk,
    moving_average, match_multimodal_emotions, audio_processing_loop,
    calculate_cosine_similarity, create_unified_emotion_vector,
    TEXT_TO_UNIFIED, SER_TO_UNIFIED, FACIAL_TO_UNIFIED,
    TEXT_CLASSIFIER_MODEL_ID, SER_MODEL_ID
)

# Try to get the video processing function from our module
external_video_processing_loop = get_video_processing_function()

def video_processing_loop(video_emotions, video_lock, stop_flag, video_started_event):
    """Wrapper around the video processing loop"""
    if external_video_processing_loop:
        # Use our external implementation
        return external_video_processing_loop(video_emotions, video_lock, stop_flag, video_started_event)
    else:
        # Fall back to the original implementation 
        from detect_emotion import video_processing_loop as original_video_processing_loop
        print("Using original video_processing_loop")
        return original_video_processing_loop(video_emotions, video_lock, stop_flag, video_started_event)

# Suppress DeepFace logging for cleaner console output
logging.getLogger().setLevel(logging.ERROR)

def main(live=True, emotion_queue=None, stop_event=None, camera_index=0):
    """Main function for emotion detection (same as original)"""
    # Log environment and parameters
    print(f"[detect_emotion_fixed] ENV WEBCAM_INDEX={os.environ.get('WEBCAM_INDEX')}, WEBCAM_BACKEND={os.environ.get('WEBCAM_BACKEND')}, SHARED_FRAME_QUEUE={os.environ.get('SHARED_FRAME_QUEUE')}")
    # Set detection mode based on argument
    print(f"[detect_emotion_fixed] Detection mode: {'live' if live else 'video_file'}")
    
    # Get camera index from environment
    camera_idx = os.environ.get('WEBCAM_INDEX')
    if camera_idx is not None:
        try:
            camera_index = int(camera_idx)
            print(f"Using camera index from environment: {camera_index}")
        except ValueError:
            print(f"Invalid camera index in environment: {camera_idx}")
    
    # Store camera index for video processing
    global CAMERA_INDEX
    CAMERA_INDEX = camera_index
    
    # Check if stop_event is a dictionary with shared frame queue
    if isinstance(stop_event, dict) and 'shared_frame_data' not in stop_event:
        # If not provided, try to get the shared frame queue from environment
        queue_id_str = os.environ.get('SHARED_FRAME_QUEUE')
        if queue_id_str:
            try:
                import ctypes
                # This is a somewhat hacky way to get the queue from its ID
                # Not ideal for production but works for our demo
                queue_id = int(queue_id_str)
                shared_queue = ctypes.cast(queue_id, ctypes.py_object).value
                stop_event['shared_frame_data'] = shared_queue
                print("Successfully retrieved shared frame queue")
            except Exception as e:
                print(f"Could not retrieve shared frame queue: {e}")
    
    # Load models
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("Loading text-based emotion classification model...")
    classifier = pipeline(
        "text-classification",
        model=TEXT_CLASSIFIER_MODEL_ID,
        top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )
    print("Loading audio-based SER model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ser_model = AutoModelForAudioClassification.from_pretrained(SER_MODEL_ID).to(device)
    ser_processor = AutoFeatureExtractor.from_pretrained(SER_MODEL_ID)
    ser_label_mapping = ser_model.config.id2label

    if not live:
        # Use original implementation for file mode
        from detect_emotion import main as original_main
        return original_main(live, emotion_queue, stop_event, camera_index)
    else:
        print("[detect_emotion_fixed] Starting live microphone and video emotion detection (threaded).")
        video_emotions = []
        audio_emotion_log = []
        video_lock = threading.Lock()
        audio_lock = threading.Lock()
        
        # Use either the external stop_event or a local stop_flag
        if stop_event:
            # Convert multiprocessing.Event to a dict for compatibility with existing code
            stop_flag = {'stop': False}
            
            # Create a thread to monitor the stop_event and update stop_flag
            def monitor_stop_event():
                while not stop_event.is_set():
                    time.sleep(0.1)
                stop_flag['stop'] = True
                print("Stop event detected, stopping emotion detection...")
            
            monitor_thread = threading.Thread(target=monitor_stop_event)
            monitor_thread.daemon = True
            monitor_thread.start()
        else:
            # If no external stop_event, use internal stop_flag
            stop_flag = {'stop': False}
        
        video_started_event = threading.Event()
        
        # Start threads
        print("[detect_emotion_fixed] Starting video processing thread...")
        video_thread = threading.Thread(target=video_processing_loop, args=(video_emotions, video_lock, stop_flag, video_started_event))
        print("[detect_emotion_fixed] Starting audio processing thread...")
        audio_thread = threading.Thread(target=audio_processing_loop, args=(audio_emotion_log, audio_lock, stop_flag, whisper_model, classifier, ser_model, ser_processor, ser_label_mapping, device, video_started_event))
        video_thread.start()
        audio_thread.start()
        
        try:
            while not stop_flag['stop']:
                time.sleep(2)  # Poll every 2 seconds
                with video_lock:
                    current_time = time.time()
                    video_window = [v for v in video_emotions if current_time - v['timestamp'] <= VIDEO_WINDOW_DURATION]
                with audio_lock:
                    current_time_audio = time.time() # Use a separate timestamp if needed for strict independence
                    audio_window = [a for a in audio_emotion_log if current_time_audio - a['timestamp'] <= AUDIO_WINDOW_DURATION]

                matches = match_multimodal_emotions(video_window, audio_window)
                if matches:
                    # Get the most recent match for the dashboard
                    latest = matches[-1]
                    cosine_sim = latest['cosine_similarity']
                    
                    # Determine consistency level
                    if cosine_sim >= 0.8:
                        consistency_level = "High Consistency ✅✅"
                    elif cosine_sim >= 0.6:
                        consistency_level = "Moderate Consistency ✅"
                    elif cosine_sim >= 0.3:
                        consistency_level = "Low Consistency ⚠️"
                    else:
                        consistency_level = "Inconsistent ❌"
                    
                    # If we have a queue, send the latest match data to the dashboard
                    if emotion_queue is not None:
                        # Extract and format the data for the dashboard
                        dashboard_data = {
                            "facial_emotion": (
                                latest.get('facial_emotion', 'unknown'),
                                latest.get('facial_confidence', 0.0)
                            ),
                            "text_emotion": (
                                latest.get('text_emotion', 'unknown') if 'text_emotion' in latest else 
                                (latest.get('audio_emotion', 'unknown') if latest.get('audio_modality') == 'text' else 'unknown'),
                                latest.get('text_confidence', 0.0) if 'text_confidence' in latest else 
                                (latest.get('audio_confidence', 0.0) if latest.get('audio_modality') == 'text' else 0.0)
                            ),
                            "audio_emotion": (
                                latest.get('audio_emotion', 'unknown') if latest.get('audio_modality', '') == 'audio' else 'unknown',
                                latest.get('audio_confidence', 0.0) if latest.get('audio_modality', '') == 'audio' else 0.0
                            ),
                            "transcribed_text": latest.get('transcribed_text', ""),
                            "cosine_similarity": cosine_sim,
                            "consistency_level": consistency_level
                        }
                        
                        try:
                            # Send data to the queue without blocking (timeout=1)
                            emotion_queue.put(dashboard_data, timeout=1)
                        except Exception as e:
                            print(f"Error sending data to dashboard: {e}")
                            
        except KeyboardInterrupt:
            print("Exiting microphone and video emotion detection.")
            stop_flag['stop'] = True
            
        print("Waiting for threads to finish...")
        video_thread.join(timeout=5)
        audio_thread.join(timeout=5)

if __name__ == "__main__":
    main()
