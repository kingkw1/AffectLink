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

def record_audio():
    """Record audio continuously and store in shared state"""
    global shared_state
    
    def audio_callback(indata, frames, time, status):
        """Callback for sounddevice to continuously capture audio"""
        if status:
            print(f"Audio recording status: {status}")
        
        # Add the new audio data to our buffer
        audio_data = indata[:, 0]  # Use first channel
        for sample in audio_data:
            audio_buffer.append(sample)
        
        # Store the latest audio for processing
        shared_state['latest_audio'] = np.array(audio_buffer)
    
    try:
        # Set up the audio stream
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=audio_sample_rate,
            blocksize=int(audio_sample_rate * 0.1)  # 100ms blocks
        )
        
        # Start recording
        with stream:
            logger.info("Audio recording started")
            
            # Keep recording until stop flag is set
            while True:
                # Check for stop signal
                if isinstance(shared_state['stop_event'], dict):
                    if shared_state['stop_event'].get('stop', False):
                        logger.info("Stop signal received in audio recording")
                        break
                elif shared_state['stop_event'] and shared_state['stop_event']:
                    logger.info("Stop event detected in audio recording")
                    break
                
                # Sleep briefly to avoid burning CPU
                time.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Error in audio recording: {e}")
    finally:
        logger.info("Audio recording stopped")

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

# Add logging for debug information
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('detect_emotion')

# Global variables
shared_state = {
    'emotion_queue': None,     # Queue for sending emotion data
    'stop_event': None,        # Event for stopping the process
    'latest_audio': None,      # Latest audio data
    'latest_frame': None,      # Latest video frame
    'transcribed_text': "",    # Latest transcribed text
    'facial_emotion': ("neutral", 1.0),  # Latest facial emotion
    'audio_emotion': ("neutral", 1.0),   # Latest audio emotion
    'text_emotion': ("neutral", 1.0),    # Latest text emotion
    'overall_emotion': "neutral",         # Combined overall emotion
}

# Reusable whisper model instance
model = None

# Text emotion classification pipeline
text_classifier = None

# Audio emotion classification model
audio_feature_extractor = None
audio_classifier = None

# Audio recording settings
audio_sample_rate = 44100
audio_duration = 5   # Record 5 seconds at a time
audio_buffer = deque(maxlen=audio_duration * audio_sample_rate)
last_audio_analysis = time.time() - 10  # Force initial analysis

# Face detection and emotion recognition
face_cascade = None

# Logger configuration
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def init_webcam(preferred_index=0, try_fallbacks=True):
    """Initialize webcam with fallback options"""
    # Try to set up camera with preferred index first
    cap = cv2.VideoCapture(preferred_index) 
    
    # If that didn't work, try alternate indices
    if not cap.isOpened() and try_fallbacks:
        logger.info(f"Camera index {preferred_index} failed, trying alternates")
        
        # Try indices 0 through 2
        for idx in range(3):
            if idx == preferred_index:
                continue  # Skip the one we already tried
                
            logger.info(f"Trying camera index {idx}")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                logger.info(f"Successfully opened camera with index {idx}")
                break
                
    # If still not open, try different backend APIs
    if not cap.isOpened() and try_fallbacks:
        # Try different backend APIs (DirectShow, etc)
        for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2]:
            logger.info(f"Trying camera index {preferred_index} with backend {backend}")
            cap = cv2.VideoCapture(preferred_index + backend)
            if cap.isOpened():
                logger.info(f"Successfully opened camera with backend {backend}")
                break
                
    if not cap.isOpened():
        logger.warning("Failed to open any webcam - video analysis disabled")
        return None
        
    # Configure for reasonable defaults
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 20)
    
    return cap
    
def process_video():
    """Process video frames from webcam"""
    global shared_state, face_cascade
    
    # Initialize OpenCV face detector
    if face_cascade is None:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Try to initialize webcam
    cap = init_webcam(preferred_index=int(os.environ.get('WEBCAM_INDEX', '0')))
    
    if cap is None:
        logger.error("Failed to initialize webcam - skipping video analysis")
        return
    
    # Get shared frame queue from stop_event dict if available
    frame_queue = None
    if isinstance(shared_state['stop_event'], dict) and 'shared_frame_data' in shared_state['stop_event']:
        frame_queue = shared_state['stop_event']['shared_frame_data']
        logger.info(f"Found shared frame queue: {frame_queue}")
    
    # Process frames in a loop  
    while cap.isOpened():
        # Check if we need to stop
        if isinstance(shared_state['stop_event'], dict):
            if shared_state['stop_event'].get('stop', False):
                logger.info("Stop signal received in video processing")
                break
        elif shared_state['stop_event'] and shared_state['stop_event'].is_set():
            logger.info("Stop event set in video processing")
            break
            
        # Capture frame
        success, frame = cap.read()
        if not success:
            logger.warning("Failed to read from webcam")
            # Try to re-initialize camera
            cap.release()
            time.sleep(1)
            cap = init_webcam()
            if cap is None:
                logger.error("Failed to reinitialize webcam - exiting video processing")
                break
            continue
            
        # Store the latest frame globally
        shared_state['latest_frame'] = frame
          # Share frame with dashboard if we have a queue
        if frame_queue is not None:
            try:
                # Try to add the frame to the queue without blocking
                if hasattr(frame_queue, 'put'):
                    # Note: need to copy the frame as it might be modified elsewhere
                    frame_queue.put(frame.copy(), block=False)
                    logger.debug("Added frame to queue")
            except Exception as e:
                # Queue might be full, that's okay
                logger.debug(f"Couldn't add frame to queue: {e}")
                
        # Also save the latest frame to a shared file location for dashboard to access
        try:
            # Save frame to temp location (every ~5 frames to avoid disk I/O overhead)
            if hasattr(process_video, 'frame_count'):
                process_video.frame_count += 1
            else:
                process_video.frame_count = 0
                
            # Only save every 5th frame to reduce disk activity
            if process_video.frame_count % 5 == 0:
                import tempfile
                frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")
                cv2.imwrite(frame_path, frame)
        except Exception as e:
            # Saving to file is optional, so don't stop on errors
            logger.debug(f"Error saving frame to file: {e}")
        
        # Run facial emotion analysis periodically
        try:
            # Skip face detection if frame is None
            if frame is None:
                continue
                
            # Convert to RGB (DeepFace expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face using DeepFace
            analysis = DeepFace.analyze(rgb_frame, 
                                      actions=['emotion'],
                                      enforce_detection=False,
                                      silent=True)
            
            if analysis and len(analysis) > 0:
                emotions = analysis[0]['emotion']
                dominant_emotion = analysis[0]['dominant_emotion']
                confidence = emotions[dominant_emotion] / 100
                
                # Store the emotion
                shared_state['facial_emotion'] = (dominant_emotion, confidence)
        except Exception as e:
            logger.error(f"Error in facial emotion detection: {e}")
            # Continue processing even if facial detection fails
            
        # Sleep briefly to avoid maxing out CPU
        time.sleep(0.05)
        
    # Clean up
    if cap is not None:
        cap.release()
    logger.info("Video processing thread exited")

def main(emotion_queue=None, stop_event=None, camera_index=0):
    """Main function to run the emotion detection system"""
    global shared_state, model, text_classifier, audio_feature_extractor, audio_classifier, face_cascade
    
    print("Starting emotion detection system...")
    
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
    model = whisper.load_model("tiny")
    
    # Initialize text emotion classifier
    print("Initializing text emotion classifier...")
    text_classifier = pipeline("text-classification", 
                            model="j-hartmann/emotion-english-distilroberta-base", 
                            top_k=None)
    
    # Initialize audio emotion classifier
    print("Initializing audio emotion classifier...")
    model_name = "MIT/ast-finetuned-speech-commands-v2"
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    audio_classifier = AutoModelForAudioClassification.from_pretrained(model_name)
    
    # Start audio recording thread
    print("Starting audio recording thread...")
    audio_thread = threading.Thread(target=record_audio)
    audio_thread.daemon = True
    audio_thread.start()
    
    # Start video processing thread
    print("Starting video processing thread...")
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
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
                  # Use the latest audio data every few seconds
            global last_audio_analysis
            if (time.time() - last_audio_analysis) > 3:
                # Process audio data for emotion detection here
                # Code to analyze audio would go here
                
                # Add timestamp to check for processing frequency
                last_audio_analysis = time.time()
                    
                # Log the update
                logger.debug(f"Audio analysis performed at {time.time()}")
                  # Share the current emotional state via the queue and file
            if latest:
                # Combine all current emotional data
                result_data = {
                    "facial_emotion": shared_state['facial_emotion'],
                    "text_emotion": shared_state['text_emotion'],
                    "audio_emotion": shared_state['audio_emotion'],
                    "transcribed_text": shared_state['transcribed_text'],
                    "consistency_level": "Moderate",
                    "cosine_similarity": 0.5  # Placeholder
                }
                
                # First try sending via queue if available
                if shared_state['emotion_queue'] is not None:
                    try:
                        # Send without blocking
                        shared_state['emotion_queue'].put(result_data, block=False)
                    except Exception as e:
                        logger.error(f"Error sending emotion data to queue: {e}")
                  # Also save to file for dashboard to access
                try:
                    # Save every few updates to reduce disk I/O
                    import json
                    import tempfile
                    import os
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
                            serializable_data[key] = (emotion_name, confidence)
                        elif isinstance(value, np.number):
                            serializable_data[key] = float(value)
                        elif isinstance(value, np.ndarray):
                            serializable_data[key] = value.tolist()
                        else:
                            serializable_data[key] = value
                    
                    # Add timestamp to the data
                    serializable_data["timestamp"] = float(time.time())
                    
                    # Save to temp file using a temporary file approach to ensure atomic writes
                    emotion_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
                    temp_path = f"{emotion_path}.tmp"
                    
                    # First write to a temp file, then rename to the final path
                    with open(temp_path, 'w') as f:
                        json.dump(serializable_data, f)
                        f.flush()
                        os.fsync(f.fileno())  # Ensure data is written to disk
                        
                    # Rename for atomic replacement
                    import shutil
                    shutil.move(temp_path, emotion_path)
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

if __name__ == "__main__":
    main()
