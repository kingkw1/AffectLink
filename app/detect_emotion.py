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

# Model IDs
TEXT_CLASSIFIER_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SER_MODEL_ID = "superb/hubert-large-superb-er"

# Emotion mapping dictionaries
TEXT_TO_UNIFIED = {
    'neutral': 'neutral',
    'joy': 'happy',
    'sadness': 'sad',
    'anger': 'angry',
    'fear': None,
    'surprise': None,
    'disgust': None
}

SER_TO_UNIFIED = {
    'neu': 'neutral',
    'hap': 'happy',
    'sad': 'sad',
    'ang': 'angry'
}

FACIAL_TO_UNIFIED = {
    'neutral': 'neutral',
    'happy': 'happy',
    'sad': 'sad', 
    'angry': 'angry',
    'fear': None,
    'surprise': None,
    'disgust': None
}

# ---------------------------
# Helper functions
# ---------------------------
def transcribe_audio_whisper(audio_path, whisper_model):
    """
    Transcribe audio file using Whisper.
    """
    try:
        result = whisper_model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        print(f"Transcription error: {e}")
        return None

def classify_emotion(text, classifier):
    """
    Classify emotion from text using Hugging Face pipeline.
    """
    try:
        result = classifier(text, top_k=1)[0]
        return result['label'], result['score']
    except Exception as e:
        print(f"Emotion classification error: {e}")
        return None, None

def classify_emotion_full(text, classifier):
    """
    Get full emotion classification results.
    """
    try:
        result = classifier(text, top_k=None)[0]  # returns list of dicts
        # Sort by score descending
        result_sorted = sorted(result, key=lambda x: x['score'], reverse=True)
        return result_sorted
    except Exception as e:
        print(f"Emotion classification error: {e}")
        return []

def analyze_audio_emotion(audio_path, ser_model, ser_processor, ser_label_mapping, device):
    """
    Analyze emotion directly from audio using a pre-trained SER model.
    Loads the audio, processes it, and predicts emotion and confidence.
    """
    try:
        # Load audio (mono, 16kHz)
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Pass numpy array directly to feature extractor
        inputs = ser_processor(waveform, sampling_rate=16000, return_tensors="pt")
        # Move to device
        for k in inputs:
            inputs[k] = inputs[k].to(device)
            
        # Get logits
        with torch.no_grad():
            logits = ser_model(**inputs).logits
            
        # Get scores
        scores = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Get the most likely emotion
        top_idx = scores.argmax()
        emotion = ser_label_mapping[top_idx]
        confidence = scores[top_idx]
        
        return emotion, confidence
    except Exception as e:
        print(f"Audio emotion analysis error: {e}")
        return None, None

def analyze_audio_emotion_full(audio_path, ser_model, ser_processor, ser_label_mapping, device):
    """
    Get full detailed audio emotion analysis
    """
    try:
        # Load audio (mono, 16kHz)
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Process audio
        inputs = ser_processor(waveform, sampling_rate=16000, return_tensors="pt")
        # Move to device
        for k in inputs:
            inputs[k] = inputs[k].to(device)
            
        # Get logits
        with torch.no_grad():
            logits = ser_model(**inputs).logits
            
        # Get scores
        scores = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Create emotion-score pairs
        result = []
        for i, score in enumerate(scores):
            emotion = ser_label_mapping[i] if i < len(ser_label_mapping) else f"unknown-{i}"
            result.append({
                "emotion": emotion,
                "score": float(score)
            })
        
        # Sort by score
        result_sorted = sorted(result, key=lambda x: x["score"], reverse=True)
        return result_sorted
    except Exception as e:
        print(f"Audio emotion full analysis error: {e}")
        return []

def record_audio_chunk(duration=5, fs=16000):
    """
    Record audio for specified duration and return as numpy array
    """
    try:
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True)
        return audio_data.flatten()
    except Exception as e:
        print(f"Audio recording error: {e}")
        return np.zeros(int(duration * fs))  # Return silence on error

def moving_average(scores):
    """
    Calculate moving average for a list of scores
    """
    if not scores:
        return 0
    return sum(scores) / len(scores)

def match_multimodal_emotions(video_emotions, audio_emotions, time_threshold=1.0):
    """
    Match video and audio emotions based on timestamps
    """
    matches = []
    
    for video_entry in video_emotions:
        video_time = video_entry["timestamp"]
        
        # Find audio entries close to this video entry
        close_audio = [
            audio for audio in audio_emotions
            if abs(audio["timestamp"] - video_time) <= time_threshold
        ]
        
        if close_audio:
            # Add matches
            for audio_entry in close_audio:
                matches.append({
                    "video_emotion": video_entry["emotion"],
                    "audio_emotion": audio_entry["emotion"],
                    "timestamp": video_time,
                    "time_diff": abs(audio_entry["timestamp"] - video_time)
                })
    
    return matches

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

def audio_processing_loop(audio_emotion_log, audio_lock, stop_flag, whisper_model, classifier, ser_model, ser_processor, ser_label_mapping, device, video_started_event):
    """Process audio for speech-to-text and emotion analysis"""
    # Wait for video processing to start
    video_started_event.wait()
    
    chunk_duration = 5
    smoothing_window = 3
    emotion_window = deque(maxlen=smoothing_window)
    score_window = deque(maxlen=smoothing_window)
    audio_emotion_window = deque(maxlen=smoothing_window)
    audio_score_window = deque(maxlen=smoothing_window)
    
    while not (isinstance(stop_flag, dict) and stop_flag.get('stop', False)) and not (hasattr(stop_flag, 'is_set') and stop_flag.is_set()):
        # Record audio chunk
        temp_wav = record_audio_chunk(duration=chunk_duration)
        
        # Transcribe audio to text
        text = transcribe_audio_whisper(temp_wav, whisper_model)

        # Get all text emotion scores
        text_emotion_scores_raw = classifier(text, top_k=None) if text else [] 

        text_emotion_scores = []
        if text and text_emotion_scores_raw and isinstance(text_emotion_scores_raw, list) and len(text_emotion_scores_raw) > 0:
            # The classifier for "j-hartmann/emotion-english-distilroberta-base" returns a list containing a list of dicts
            if isinstance(text_emotion_scores_raw[0], list):
                 text_emotion_scores = sorted(text_emotion_scores_raw[0], key=lambda x: x['score'], reverse=True)
            elif isinstance(text_emotion_scores_raw[0], dict): # Fallback if the structure is flatter
                 text_emotion_scores = sorted(text_emotion_scores_raw, key=lambda x: x['score'], reverse=True)

        # Get top text emotion
        if text_emotion_scores:
            top_text = text_emotion_scores[0]
            emotion = top_text['label']
            score = top_text['score']
        else:
            emotion, score = None, None
            
        text_timestamp = time.time()
        
        # Get audio emotion scores
        audio_emotion, audio_score, audio_emotion_scores = analyze_audio_emotion_full(temp_wav, ser_model, ser_processor, ser_label_mapping, device)
        audio_timestamp = time.time()
        
        # Clean up temp file
        if isinstance(temp_wav, str) and os.path.exists(temp_wav):
            os.unlink(temp_wav)
            
        # Smoothing text emotions
        if emotion:
            emotion_window.append(emotion)
            score_window.append(score)
            smoothed_emotion = max(set(emotion_window), key=emotion_window.count)
            smoothed_score = moving_average([s for e, s in zip(emotion_window, score_window) if e == smoothed_emotion])
            log_entry_text = {
                'timestamp': text_timestamp,
                'modality': 'text',
                'emotion': smoothed_emotion,
                'confidence': smoothed_score,
                'emotion_scores': text_emotion_scores
            }
            with audio_lock:
                audio_emotion_log.append(log_entry_text)
                
        # Smoothing audio emotions
        if audio_emotion:
            audio_emotion_window.append(audio_emotion)
            audio_score_window.append(audio_score)
            smoothed_audio_emotion = max(set(audio_emotion_window), key=audio_emotion_window.count)
            smoothed_audio_score = moving_average([s for e, s in zip(audio_emotion_window, audio_score_window) if e == smoothed_audio_emotion])
            with audio_lock:
                audio_emotion_log.append({
                    'timestamp': audio_timestamp,
                    'modality': 'audio',
                    'emotion': smoothed_audio_emotion,
                    'confidence': smoothed_audio_score,
                    'emotion_scores': audio_emotion_scores
                })

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
