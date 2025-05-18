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
        # First check if we actually have valid audio data
        import soundfile as sf
        import os
        import hashlib
        import random
        
        # Generate a truly unique ID for this transcription attempt
        # Include more entropy sources to ensure we don't get cached results
        current_time_ms = int(time.time() * 1000)
        random_salt = random.randint(0, 1000000)
        process_id = os.getpid()
        audio_file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        unique_string = f"{audio_path}_{current_time_ms}_{random_salt}_{process_id}_{audio_file_size}"
        transcription_id = hashlib.md5(unique_string.encode()).hexdigest()[:8]
        
        logger.info(f"[{transcription_id}] Starting transcription of {audio_path} at {current_time_ms}")
        
        if not os.path.exists(audio_path):
            logger.warning(f"[{transcription_id}] Audio file missing: {audio_path}")
            return None
            
        file_size = os.path.getsize(audio_path)
        if file_size < 2000:  # Increased minimum size threshold
            logger.warning(f"[{transcription_id}] Audio file too small: {file_size} bytes")
            return None
            
        # Check audio content
        audio_data, sample_rate = sf.read(audio_path)
        audio_duration = len(audio_data) / sample_rate
        audio_rms = np.sqrt(np.mean(audio_data**2))
        audio_peak = np.max(np.abs(audio_data))
        audio_std = np.std(audio_data)
        
        logger.info(f"Audio file stats: path={audio_path}, duration={audio_duration:.2f}s, RMS={audio_rms:.6f}, peak={audio_peak:.6f}, std={audio_std:.6f}, samples={len(audio_data)}")
        
        # Save detailed audio stats to help diagnose issues
        segments = 5
        segment_size = len(audio_data) // segments
        if segment_size > 0:
            segment_stats = []
            for i in range(segments):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size
                segment = audio_data[start_idx:end_idx]
                segment_rms = np.sqrt(np.mean(segment**2))
                segment_stats.append(f"{segment_rms:.6f}")
            logger.info(f"Audio segment RMS values: {', '.join(segment_stats)}")
        
        # If audio is essentially silence, don't bother transcribing
        if audio_rms < 0.005:  # Increased RMS threshold for silence detection
            logger.warning(f"[{transcription_id}] Audio appears to be silence (RMS={audio_rms:.6f}), skipping transcription")
            return None
            
        # Log whisper model device
        if hasattr(whisper_model, 'device'):
            device_info = f"on device {whisper_model.device}"
        else:
            device_info = "(device unknown)"
            
        logger.info(f"[{transcription_id}] Running Whisper transcription {device_info} on {audio_duration:.2f}s audio")
            
        # Proceed with transcription for valid audio
        start_time = time.time()
        
        # Use more aggressive options for better transcription
        try:
            # Use more parameters to improve accuracy and ensure actual speech is detected
            result = whisper_model.transcribe(
                audio_path,
                language="en",  # Force English language
                temperature=0.0,  # More deterministic output
                no_speech_threshold=0.5,  # Slightly less aggressive filtering
                logprob_threshold=-1.0,  # Default, less permissive
                condition_on_previous_text=False,  # Don't condition on previous text to avoid repetition
                initial_prompt="This is a transcription of speech."  # Help the model understand context
            )
        except Exception as e:
            logger.error(f"[{transcription_id}] Error during whisper transcription: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
        transcription_time = time.time() - start_time
        
        # Log the result details
        transcribed_text = result['text'].strip() if result and 'text' in result else ""
        
        # Extra validation to ensure we're not repeating the same text due to caching
        if hasattr(transcribe_audio_whisper, 'last_transcription') and (
            transcribe_audio_whisper.last_transcription == transcribed_text or 
            (not transcribed_text and transcribe_audio_whisper.last_transcription == "")
        ):
            logger.warning(f"[{transcription_id}] Detected repeated transcription: '{transcribed_text or 'empty'}'")
            
            # Try forcing different text by adding noise parameter to next call
            if not hasattr(transcribe_audio_whisper, 'repeat_count'):
                transcribe_audio_whisper.repeat_count = 0
            transcribe_audio_whisper.repeat_count += 1
            
            # If empty transcription or same text repeatedly, try forcing buffer reset sooner
            if not transcribed_text or transcribe_audio_whisper.repeat_count > 1:
                logger.warning(f"[{transcription_id}] {'Empty' if not transcribed_text else 'Same'} text repeated, forcing audio buffer reset")
                
                # Reset the counter and clear the last transcription
                transcribe_audio_whisper.repeat_count = 0
                transcribe_audio_whisper.last_transcription = None
                
                # Signal to the caller that they should reset the buffer
                return "RESET_BUFFER"
        else:
            # Reset repeat counter when we get different text
            transcribe_audio_whisper.repeat_count = 0                # Store for comparison next time
            transcribe_audio_whisper.last_transcription = transcribed_text
            
            # Log detailed results
            if transcribed_text:
                logger.info(f"[{transcription_id}] Transcribed in {transcription_time:.2f}s: '{transcribed_text}' (length: {len(transcribed_text)})")
            
            # Log segments if available for more detailed analysis
            if 'segments' in result and len(result['segments']) > 0:
                segments_info = []
                for i, segment in enumerate(result['segments']):
                    if 'text' in segment:
                        segments_info.append(f"[{i}] {segment.get('text', '')[:30]}...")
                logger.info(f"[{transcription_id}] Segments: {' | '.join(segments_info)}")
            else:
                logger.warning(f"[{transcription_id}] Transcription returned empty result after {transcription_time:.2f}s despite valid audio")
        
        return transcribed_text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        import traceback
        logger.error(traceback.format_exc())
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

def analyze_audio_emotion(audio_path, ser_model, ser_processor, device): # REMOVED ser_label_mapping
    """
    Analyze emotion directly from audio using a pre-trained SER model.
    Loads the audio, processes it, and predicts emotion and confidence.
    """
    try:
        # Load audio (mono, 16kHz)
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Pass numpy array directly to feature extractor
        inputs = ser_processor(waveform, sampling_rate=16000, return_tensors="pt")

        # Move inputs to the same device as the model
        model_device = next(ser_model.parameters()).device
        
        for k in inputs:
            inputs[k] = inputs[k].to(model_device)
            
        # Get logits
        with torch.no_grad():
            logits = ser_model(**inputs).logits
            
        # Get scores
        scores = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Get the actual labels from the model's config
        model_labels_dict = ser_model.config.id2label
        # Create a list of labels, ensuring correct order by index
        ser_actual_labels = [model_labels_dict[i] for i in sorted(model_labels_dict.keys())]

        logger.info(f"SER_DEBUG: ser_actual_labels from model: {ser_actual_labels}")
        logger.info(f"SER_DEBUG: raw_scores: {scores.tolist()}")
        top_idx = scores.argmax()
        logger.info(f"SER_DEBUG: top_idx: {top_idx}")
        logger.info(f"SER_DEBUG: len(ser_actual_labels): {len(ser_actual_labels)}")
        logger.info(f"SER_DEBUG: condition (top_idx < len(ser_actual_labels)): {top_idx < len(ser_actual_labels)}")

        emotion = ser_actual_labels[top_idx] if top_idx < len(ser_actual_labels) else "unknown"
        confidence = float(scores[top_idx])
        
        logger.info(f"SER_DEBUG: determined_emotion: {emotion}, determined_confidence: {confidence:.4f}")

        return emotion, confidence
    except Exception as e:
        logger.error(f"Audio emotion analysis error: {e}")
        import traceback
        logger.error(traceback.format_exc()) 
        return None, None

def analyze_audio_emotion_full(audio_path, ser_model, ser_processor, device): # REMOVED ser_label_mapping
    """
    Get full detailed audio emotion analysis
    """
    try:
        # Load audio (mono, 16kHz)
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Process audio
        inputs = ser_processor(waveform, sampling_rate=16000, return_tensors="pt")
        
        model_device = next(ser_model.parameters()).device
        logger.debug(f"Model is on device: {model_device}")
        
        for k in inputs:
            inputs[k] = inputs[k].to(model_device)
            
        with torch.no_grad():
            logits = ser_model(**inputs).logits
            
        scores = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Get the actual labels from the model's config
        model_labels_dict = ser_model.config.id2label
        ser_actual_labels = [model_labels_dict[i] for i in sorted(model_labels_dict.keys())]
        
        all_results = []
        for i, score in enumerate(scores):
            emotion = ser_actual_labels[i] if i < len(ser_actual_labels) else f"unknown-{i}"
            all_results.append({
                "emotion": emotion,
                "score": float(score)
            })
        
        result_sorted = sorted(all_results, key=lambda x: x["score"], reverse=True)
        
        if result_sorted:
            top_emotion = result_sorted[0]["emotion"]
            top_score = result_sorted[0]["score"]
            return top_emotion, top_score, result_sorted
        else:
            return "neutral", 0.0, []
            
    except Exception as e:
        logger.error(f"Audio emotion full analysis error: {e}")
        return "neutral", 0.0, []

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

def audio_processing_loop(audio_emotion_log, audio_lock, stop_flag, whisper_model, classifier, ser_model, ser_processor, device, video_started_event): # REMOVED ser_label_mapping
    """Process audio for speech-to-text and emotion analysis"""
    # Add debug logging to track thread execution
    global audio_buffer  # Declare global at top of function
    logger.info("Audio processing thread started")
    
    # Add a timestamp to track how long we've had the same transcription
    last_transcription_change_time = time.time()
    last_transcription = ""
    
    try:
        # Wait for video processing to start
        logger.info("Waiting for video processing to start...")
        video_started_event.wait()
        logger.info("Video processing started, beginning audio processing")
        
        chunk_duration = 5
        smoothing_window = 3
        emotion_window = deque(maxlen=smoothing_window)
        score_window = deque(maxlen=smoothing_window)
        audio_emotion_window = deque(maxlen=smoothing_window)
        audio_score_window = deque(maxlen=smoothing_window)
        
        while not (isinstance(stop_flag, dict) and stop_flag.get('stop', False)) and not (hasattr(stop_flag, 'is_set') and stop_flag.is_set()):
            try:
                # Use the audio buffer collected in record_audio
                if shared_state['latest_audio'] is None or len(shared_state['latest_audio']) < 1000:
                    logger.info("No audio data available yet, waiting...")
                    time.sleep(1)
                    continue
                
                # Save the latest audio to a temporary file for processing
                temp_wav = os.path.join(tempfile.gettempdir(), f"affectlink_temp_audio_{int(time.time()*1000)}.wav")
                audio_data = shared_state['latest_audio'].copy()  # Create a copy to avoid race conditions
                logger.debug(f"Saving audio chunk with {len(audio_data)} samples for processing")
                
                # Calculate some basic audio stats for diagnostics
                audio_mean = np.mean(np.abs(audio_data))
                audio_max = np.max(np.abs(audio_data))
                logger.info(f"Audio data stats: mean={audio_mean:.6f}, max={audio_max:.6f}, samples={len(audio_data)}")
                
                # Resample to 16kHz for Whisper - audio_data is already 16kHz if audio_sample_rate is 16000
                sf.write(temp_wav, audio_data, 16000)

                # ADDED: Detailed logging of audio data before transcription
                if audio_data is not None and audio_data.size > 0:
                    logger.info(f"APL_PRE_TRANSCRIPTION_AUDIO: samples={len(audio_data)}, duration={len(audio_data)/16000.0:.2f}s, dtype={audio_data.dtype}, min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, mean_abs={np.mean(np.abs(audio_data)):.4f}, sum_abs={np.sum(np.abs(audio_data)):.4f}")
                else:
                    logger.warning("APL_PRE_TRANSCRIPTION_AUDIO: No audio data or empty audio data before transcription attempt.")
                
                # Transcribe audio to text
                logger.debug("Transcribing audio...")
                text = transcribe_audio_whisper(temp_wav, whisper_model)
                
                if text == "RESET_BUFFER":
                    logger.warning("Whisper requested buffer reset. Clearing audio buffer.")
                    with audio_lock:
                        audio_buffer.clear()
                        shared_state['latest_audio'] = None
                        shared_state['transcribed_text'] = "" # Clear current transcription
                        shared_state['audio_reset_time'] = time.time() # Signal update
                    # Reset periodic reset counter
                    if hasattr(audio_processing_loop, 'buffer_reset_count'):
                        audio_processing_loop.buffer_reset_count = 0
                    # Clear last texts history to prevent other resets from misfiring
                    if hasattr(audio_processing_loop, 'last_texts'):
                        audio_processing_loop.last_texts = []
                    time.sleep(0.5) # Brief pause after reset
                    continue # Skip the rest of this iteration

                # Use a simple counter for buffer management
                # Initialize buffer reset counter if it doesn't exist yet
                if not hasattr(audio_processing_loop, 'buffer_reset_count'):
                    audio_processing_loop.buffer_reset_count = 0
                
                # Increment the counter
                audio_processing_loop.buffer_reset_count += 1
                
                # More aggressive buffer management - reset more frequently
                if audio_processing_loop.buffer_reset_count >= 5:  # Changed from 2 to 5
                    logger.info("Periodically clearing audio buffer to ensure fresh audio")
                    # audio_buffer already declared as global at top of function
                    with audio_lock:
                        audio_buffer.clear()
                        # Also clear the shared state cache to force new processing
                        shared_state['latest_audio'] = None
                        shared_state['transcribed_text'] = ""
                        # Add a timestamp to force dashboard recognition of a new update
                        shared_state['audio_reset_time'] = time.time()
                    
                    # Force more variation in reset timing to break potential loops
                    import random
                    random_sleep = random.uniform(0.2, 0.7)
                    time.sleep(random_sleep)
                    
                    # Reset counter
                    audio_processing_loop.buffer_reset_count = 0
                    
                    # Log with more detail
                    logger.warning(f"Audio buffer reset performed at {time.time():.3f} with {random_sleep:.3f}s sleep")
                
                # Track the last few transcriptions to detect if we're stuck in a loop
                if not hasattr(audio_processing_loop, 'last_texts'):
                    audio_processing_loop.last_texts = []
                    audio_processing_loop.no_new_text_counter = 0
                
                if text:
                    logger.info(f"Transcribed text: {text[:50]}...")
                    
                    # Add timestamp to the text to help identify if it's actually new
                    text_with_timestamp = f"{text} [ts:{time.time():.2f}]"
                    
                    # Store this transcription for repeat detection
                    audio_processing_loop.last_texts.append(text)
                    if len(audio_processing_loop.last_texts) > 5:
                        audio_processing_loop.last_texts.pop(0)
                    
                    # If we have the same text at all, be much more aggressive about resetting
                    # We only need 2 repeat detections to trigger a reset now
                    if len(audio_processing_loop.last_texts) >= 2:
                        # Only compare the actual text without timestamps
                        if audio_processing_loop.last_texts[-1] == audio_processing_loop.last_texts[-2]:
                            logger.warning(f"Detected same transcription twice in a row - force clearing buffer: '{text}'")
                            # Clear the buffer (audio_buffer already declared as global at top of function)
                            with audio_lock:
                                audio_buffer.clear()
                                # Also clear shared state to force new processing
                                shared_state['latest_audio'] = None
                                # Update with timestamped text to force a change
                                shared_state['transcribed_text'] = text_with_timestamp
                            
                            # Force random sleep to introduce variability 
                            import random
                            random_sleep = random.uniform(0.1, 0.5)
                            time.sleep(random_sleep)
                                
                            # Also clear the tracking history and reset counter to trigger faster reset
                            audio_processing_loop.last_texts = []
                            audio_processing_loop.buffer_reset_count = 0
                else:
                    logger.warning("No transcription generated - will try again with fresh audio")

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
                text_emotion = None
                text_score = 0.0
                if text_emotion_scores:
                    top_text = text_emotion_scores[0]
                    text_emotion = top_text['label']
                    text_score = top_text['score']
                    logger.info(f"Text emotion: {text_emotion} ({text_score:.2f})")
                    # Update shared state with transcript and text emotion
                    shared_state['transcribed_text'] = text
                    shared_state['text_emotion'] = (text_emotion, text_score)
                    logger.info(f"SHARED_STATE UPDATE: text_emotion set to ({text_emotion}, {text_score:.2f})") # <--- ADDED LOG
                    
                    # Check if transcription has changed
                    if text != last_transcription:
                        # Log that we got a new transcription 
                        logger.info(f"New transcription detected: '{text[:30]}...' (previous: '{last_transcription[:20]}...')")
                        last_transcription = text
                        last_transcription_change_time = time.time()
                        
                        # Initialize or reset a stuck counter
                        if not hasattr(audio_processing_loop, 'stuck_counter'):
                            audio_processing_loop.stuck_counter = 0
                        else:
                            audio_processing_loop.stuck_counter = 0
                    else:
                        # Check if we've been stuck with the same transcription
                        current_time = time.time()
                        time_since_change = current_time - last_transcription_change_time
                        
                        # Track how long we've been stuck
                        if not hasattr(audio_processing_loop, 'stuck_counter'):
                            audio_processing_loop.stuck_counter = 0
                        audio_processing_loop.stuck_counter += 1
                        
                        # More aggressive: reduce the stuck time threshold to 10 seconds
                        if time_since_change > 10 or audio_processing_loop.stuck_counter >= 3:
                            logger.warning(f"Stuck on transcription for {time_since_change:.1f}s (count: {audio_processing_loop.stuck_counter}): '{text[:30]}...' - forcing COMPLETE reset")
                            
                            # Complete buffer reset
                            with audio_lock:
                                # Full reset of all audio-related state
                                audio_buffer.clear()
                                shared_state['latest_audio'] = None
                                shared_state['transcribed_text'] = ""
                                
                                # Reset any whisper caches by adding unique flag
                                if hasattr(transcribe_audio_whisper, 'last_transcription'):
                                    delattr(transcribe_audio_whisper, 'last_transcription')
                                if hasattr(transcribe_audio_whisper, 'repeat_count'):
                                    delattr(transcribe_audio_whisper, 'repeat_count')
                                    
                            # Reset tracking variables
                            last_transcription = ""
                            last_transcription_change_time = current_time
                            audio_processing_loop.stuck_counter = 0
                            
                            # Add random delay to help break any patterns
                            import random
                            random_sleep = random.uniform(0.3, 0.8)
                            time.sleep(random_sleep)
                            logger.info(f"Inserted random delay of {random_sleep:.2f}s after reset")
                            
                            continue  # Skip rest of processing loop
                else:
                    logger.debug("No text emotions detected")
                
                # Get audio emotion scores - using improved function
                logger.debug("Analyzing audio emotion...")
                try:
                    # Use direct analyze_audio_emotion for simplicity
                    audio_emotion, audio_score = analyze_audio_emotion(temp_wav, ser_model, ser_processor, device) # UPDATED call
                    if audio_emotion and audio_score is not None: # Check audio_score is not None explicitly
                        logger.info(f"Audio emotion: {audio_emotion} ({audio_score:.2f})")
                        # Update the shared state directly for dashboard access
                        shared_state['audio_emotion'] = (audio_emotion, audio_score)
                        logger.info(f"SHARED_STATE UPDATE: audio_emotion set to ({audio_emotion}, {audio_score:.2f})") # <--- ADDED LOG
                    else:
                        logger.debug("Audio emotion analysis returned no valid results or score was None")
                        # Ensure shared_state reflects no valid audio emotion if analysis fails
                        if shared_state.get('audio_emotion') != ("unknown", 0.0):
                             shared_state['audio_emotion'] = ("unknown", 0.0) # Default to unknown
                             logger.info(f"SHARED_STATE UPDATE: audio_emotion reset to ('unknown', 0.0) due to failed analysis") # <--- ADDED LOG

                except Exception as audio_err:
                    logger.error(f"Error analyzing audio emotion: {audio_err}")
                    if shared_state.get('audio_emotion') != ("unknown", 0.0):
                        shared_state['audio_emotion'] = ("unknown", 0.0) # Default to unknown on error
                        logger.info(f"SHARED_STATE UPDATE: audio_emotion reset to ('unknown', 0.0) due to exception") # <--- ADDED LOG
                
                # Clean up temp file
                if isinstance(temp_wav, str) and os.path.exists(temp_wav):
                    try:
                        os.unlink(temp_wav)
                    except:
                        pass  # Ignore cleanup errors
                        
                # Clear part of the audio buffer more aggressively after successful processing
                # REMOVED: This section was too aggressive and might lead to fragmented audio.
                # if text and len(audio_buffer) > audio_sample_rate:
                #     logger.info("Nearly completely resetting audio buffer to avoid repetitive transcriptions")
                #     with audio_lock:
                #         keep_samples = int(audio_sample_rate * 0.25) 
                #         new_buffer = list(audio_buffer)[-keep_samples:]
                #         audio_buffer.clear()
                #         for sample in new_buffer:
                #             audio_buffer.append(sample)
                #         current_time = time.time()
                #         shared_state['transcribed_text'] = f"" 
                #         shared_state['buffer_reset_timestamp'] = current_time
                #         logger.info(f"Audio buffer trimmed at {current_time:.3f}, kept {keep_samples} samples")
                
                # Smoothing text emotions - similar to original but with better error handling
                if text_emotion:
                    emotion_window.append(text_emotion)
                    score_window.append(text_score)
                    try:
                        smoothed_emotion = max(set(emotion_window), key=emotion_window.count)
                        smoothed_score = moving_average([s for e, s in zip(emotion_window, score_window) if e == smoothed_emotion])
                        
                        log_entry_text = {
                            'timestamp': time.time(),
                            'modality': 'text',
                            'emotion': smoothed_emotion,
                            'confidence': smoothed_score,
                            'emotion_scores': text_emotion_scores
                        }
                        with audio_lock:
                            audio_emotion_log.append(log_entry_text)
                    except Exception as e:
                        logger.error(f"Error in text emotion smoothing: {e}")
                    
                # Smoothing audio emotions
                if audio_emotion:
                    audio_emotion_window.append(audio_emotion)
                    audio_score_window.append(audio_score)
                    try:
                        smoothed_audio_emotion = max(set(audio_emotion_window), key=audio_emotion_window.count)
                        smoothed_audio_score = moving_average([s for e, s in zip(audio_emotion_window, audio_score_window) if e == smoothed_audio_emotion])
                        
                        with audio_lock:
                            audio_emotion_log.append({
                                'timestamp': time.time(),
                                'modality': 'audio',
                                'emotion': smoothed_audio_emotion,
                                'confidence': smoothed_audio_score,
                                'emotion_scores': None  # We're not returning full scores here
                            })
                    except Exception as e:
                        logger.error(f"Error in audio emotion smoothing: {e}")
                
                # Sleep to avoid high CPU usage
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
                import traceback
                logger.error(traceback.format_exc())
                time.sleep(1)  # Avoid tight loop on errors
            
    except Exception as e:
        logger.error(f"Fatal error in audio processing thread: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("Audio processing thread stopped")

def record_audio():
    """Record audio continuously and store in shared state"""
    global shared_state, audio_buffer
    
    # Print available devices for debugging
    try:
        logger.info("Available audio devices:")
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        logger.info(f"Default input device: {default_input.get('name', 'Unknown')}")
        for i, device in enumerate(devices):
            if device.get('max_input_channels', 0) > 0:  # Only input devices
                logger.info(f"  {i}: {device.get('name', 'Unknown')} (Channels: {device.get('max_input_channels')})")
    except Exception as e:
        logger.error(f"Error querying audio devices: {e}")
    
    # Track audio stats
    audio_level_tracker = deque(maxlen=20)  # Track last 20 audio chunks
    last_stats_time = time.time()
    
    def audio_callback(indata, frames, time_, status):
        """Callback for sounddevice to continuously capture audio"""
        nonlocal audio_level_tracker, last_stats_time
        
        if status:
            logger.info(f"Audio recording status: {status}")
        
        # Add the new audio data to our buffer
        audio_data = indata[:, 0]  # Use first channel
        
        # Track audio levels for diagnostics
        audio_rms = np.sqrt(np.mean(audio_data**2))
        audio_level_tracker.append(audio_rms)
        
        # Periodic diagnostics about audio levels
        current_time = time.time()
        if current_time - last_stats_time > 5:  # Every 5 seconds
            avg_level = sum(audio_level_tracker) / len(audio_level_tracker) if audio_level_tracker else 0
            peak_level = max(audio_level_tracker) if audio_level_tracker else 0
            logger.info(f"Audio levels: current={audio_rms:.6f}, avg={avg_level:.6f}, peak={peak_level:.6f}, buffer_size={len(audio_buffer)}")
            
            # Detailed buffer diagnostics
            if audio_buffer:
                buffer_array = np.array(list(audio_buffer)) # Convert deque to list then numpy array for stats
                if buffer_array.size > 0: # Ensure buffer_array is not empty
                    buffer_min = np.min(buffer_array)
                    buffer_max = np.max(buffer_array)
                    buffer_std = np.std(buffer_array)
                    logger.info(f"Buffer stats: min={buffer_min:.6f}, max={buffer_max:.6f}, std={buffer_std:.6f}")
                else:
                    logger.info("Buffer stats: audio_buffer is currently empty for stats calculation.")

            if avg_level < 0.001: # This threshold is for warning, not for stopping transcription
                logger.warning("Audio levels are very low. Is your microphone working and unmuted?")
            last_stats_time = current_time
        
        # REMOVED: Redundant buffer size check. Deque's maxlen handles this.
        # max_buffer_size = audio_sample_rate * 10 
        # if len(audio_buffer) > max_buffer_size:
        #     logger.warning(f"Audio buffer exceeded max size ({len(audio_buffer)} > {max_buffer_size}), trimming")
        #     new_buffer = list(audio_buffer)[-max_buffer_size:]
        #     audio_buffer.clear()
        #     for sample in new_buffer:
        #         audio_buffer.append(sample)
            
        # Add new audio data to buffer
        for sample in audio_data:
            audio_buffer.append(sample)
        
        # Store the latest audio for processing
        try:
            # Ensure audio_buffer is not empty before converting to numpy array
            if audio_buffer:
                audio_array = np.array(list(audio_buffer)) # Convert deque to list then numpy array
                shared_state['latest_audio'] = audio_array
            else:
                # If buffer is empty, ensure latest_audio reflects that
                shared_state['latest_audio'] = np.array([])

        except Exception as e:
            logger.error(f"Error updating latest audio: {e}")
        
        # Track silence periods
        if not hasattr(audio_callback, 'silence_tracker'):
            audio_callback.silence_tracker = 0
        
        # Check if this is essentially silence - be more aggressive with buffer clearing
        if audio_rms < 0.001:
            audio_callback.silence_tracker += 1
            # Reset buffer after 15 seconds of silence to avoid "I'm glad I found you" syndrome
            # Much more aggressive than before (15 seconds instead of 30)
            if audio_callback.silence_tracker > 150:  # ~15 seconds if called every 100ms
                logger.warning("Detected extended silence - resetting audio buffer")
                audio_buffer.clear()
                shared_state['latest_audio'] = np.array([])  # Clear latest audio as well
                audio_callback.silence_tracker = 0
        else:
            audio_callback.silence_tracker = 0  # Reset on non-silent audio
            
        # Add a timestamp to track when we last got audio data
        if not hasattr(audio_callback, 'last_audio_time'):
            audio_callback.last_audio_time = time.time()
        else:
            audio_callback.last_audio_time = time.time()
    
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
            logger.info(f"Audio recording started with sample rate {audio_sample_rate}Hz")
            
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
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info("Audio recording stopped")

def video_processing_loop(video_emotions, video_lock, stop_flag, video_started_event):
    """Wrapper around the video processing loop"""
    # Get the external video processing function
    external_function = get_video_processing_function()
    
    if external_function:
        # Use our external implementation
        return external_function(video_emotions, video_lock, stop_flag, video_started_event)
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
audio_sample_rate = 16000 # Changed from 44100 to 16000
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

# Clear any existing frame and emotion files at startup to prevent the dashboard
# from loading stale data from previous sessions
def clear_stale_files():
    """Delete any existing frame and emotion files to ensure a fresh start"""
    try:
        import tempfile
        import os
        
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
        # This is a fallback/complement to the queue-based sharing
        try:
            # Increment frame counter
            if hasattr(process_video, 'frame_count'):
                process_video.frame_count += 1
            else:
                process_video.frame_count = 0
                
            # Increase the update frequency - save every 2nd frame instead of every 3rd
            # This provides more frequent updates to the dashboard
            if process_video.frame_count % 2 == 0:  # Changed from 3 to 2
                import tempfile
                import shutil
                import random
                
                # Add a small random component to filenames to avoid any caching issues
                random_suffix = random.randint(1000, 9999)
                
                # First write to a temp file to avoid partial reads by dashboard
                tmp_frame_path = os.path.join(tempfile.gettempdir(), f"affectlink_frame_tmp_{random_suffix}.jpg")
                frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")
                
                # Save with higher quality (95) to temp file
                # Convert to RGB before saving to ensure proper color format
                if frame is not None:
                    # Make a copy to avoid modifying the original
                    frame_to_save = frame.copy()
                    cv2.imwrite(tmp_frame_path, frame_to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    
                    # Then move the temp file to the final location (atomic operation)
                    shutil.move(tmp_frame_path, frame_path)
                    logger.debug(f"Saved frame to {frame_path} (frame #{process_video.frame_count})")
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
    
    # Use simpler model instead
    model_name = "MIT/ast-finetuned-speech-commands-v2" # THIS IS LIKELY THE WRONG MODEL FOR SER
    # The SER model used in analyze_audio_emotion is superb/hubert-large-superb-er
    # Let's ensure we are loading the correct model here for consistency if this `audio_classifier` is indeed the SER model.
    # Based on the call structure, `audio_classifier` and `audio_feature_extractor` are passed as `ser_model` and `ser_processor`
    
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
        import traceback
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
            
            # Share the current emotional state via the queue and file
            # Always share data, regardless of latest, to ensure transcriptions are sent
            
            # Combine all current emotional data
            result_data = {
                "facial_emotion": shared_state['facial_emotion'],
                "text_emotion": shared_state['text_emotion'],
                "audio_emotion": shared_state['audio_emotion'],
                "transcribed_text": shared_state['transcribed_text'],
                "consistency_level": "Moderate",
                "cosine_similarity": 0.5  # Placeholder
            }
            
            # Add timestamps to transcriptions to ensure they update in UI
            if shared_state['transcribed_text']:
                result_data["transcribed_text"] = f"{shared_state['transcribed_text']} [{time.time():.3f}]"
            
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
                    import json
                    import tempfile
                    import os
                    import random
                    
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
                        elif isinstance(value, np.number):
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
                    import shutil
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

def test_single_audio_transcription(duration=7, model_name="base.en"):
    """
    Records a single audio clip, saves it, transcribes it, and prints the result.
    Helps to test the audio recording and Whisper transcription in isolation.
    """
    # Ensure the module-level logger is used (it should be defined in the global scope of detect_emotion.py)
    logger.info("=== Starting Single Audio Transcription Test ===")
    
    # 1. Load Whisper model
    whisper_model = None
    try:
        logger.info(f"Loading Whisper model: {model_name}...")
        # Ensure whisper is imported if not already at the top of the file
        import whisper 
        whisper_model = whisper.load_model(model_name)
        logger.info(f"Whisper model '{model_name}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        logger.error("Please ensure Whisper is installed correctly (e.g., pip install openai-whisper).")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

    # 2. Record audio chunk
    fs = 16000  # Whisper prefers 16kHz
    logger.info(f"Recording {duration} seconds of audio at {fs}Hz. Please speak clearly.")
    try:
        # Ensure sounddevice (sd) and numpy (np) are imported
        import sounddevice as sd
        import numpy as np
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', blocking=True)
        sd.wait() # Wait for recording to complete
        logger.info("Audio recording complete.")
    except Exception as e:
        logger.error(f"Audio recording failed: {e}")
        logger.error("Please ensure your microphone is connected and sounddevice is installed correctly.")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

    if audio_data is None or audio_data.size == 0:
        logger.error("No audio data recorded. The recording might have failed silently.")
        return None, None
    
    logger.info(f"Recorded audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}, min: {np.min(audio_data):.4f}, max: {np.max(audio_data):.4f}, mean_abs: {np.mean(np.abs(audio_data)):.4f}")


    # 3. Save to temporary WAV file
    temp_wav_path = None
    try:
        # Ensure tempfile, os, time, soundfile (sf) are imported
        import tempfile
        import os
        import time
        import soundfile as sf

        temp_dir = tempfile.gettempdir()
        temp_wav_filename = f"affectlink_test_audio_{int(time.time())}.wav"
        temp_wav_path = os.path.join(temp_dir, temp_wav_filename)
        
        logger.info(f"Saving audio to temporary file: {temp_wav_path}")
        sf.write(temp_wav_path, audio_data, fs)
        logger.info("Audio saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save audio to WAV file: {e}")
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path) # Clean up partial file
            except Exception as cleanup_e:
                logger.error(f"Error cleaning up partial WAV file: {cleanup_e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

    # 4. Transcribe
    logger.info(f"Starting transcription of {temp_wav_path}...")
    transcription = transcribe_audio_whisper(temp_wav_path, whisper_model) # transcribe_audio_whisper is already in detect_emotion.py

    if transcription == "RESET_BUFFER": 
        logger.warning("Transcription function requested a buffer reset, which is not expected in this isolated test.")
        transcription = None 

    if transcription:
        logger.info("--- Transcription Result ---")
        logger.info(f"'{transcription}'")
        logger.info("----------------------------")
    else:
        logger.info("--- No transcription obtained or transcription was empty ---")

    logger.info(f"The recorded audio was saved to: {temp_wav_path}")
    logger.info("Please listen to this file to verify the audio quality and content.")
    
    # We will not delete the file automatically so you can inspect it.
    # If you want to delete it, uncomment the lines below:
    # try:
    #     logger.info(f"Deleting temporary audio file: {temp_wav_path}")
    #     if os.path.exists(temp_wav_path):
    #         os.remove(temp_wav_path)
    # except Exception as e:
    #     logger.error(f"Could not delete temporary file {temp_wav_path}: {e}")
    
    logger.info("=== Single Audio Transcription Test Finished ===")
    return temp_wav_path, transcription

# Add this at the very end of the file:
if __name__ == '__main__':
    # Ensure logging is configured if running this file directly
    # Use the logger instance that is defined globally in the module, e.g., `logger`
    # If your global logger is `logger = logging.getLogger('detect_emotion')`
    
    # Check if the root logger (which basicConfig configures) has handlers.
    # Or check a specific logger if you know it (e.g., logger defined in the module global scope)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # If your module uses a specific logger like `logger = logging.getLogger('detect_emotion')`,
    # ensure it's also set up.
    module_logger = logging.getLogger('detect_emotion') # Or __name__ if that's what you used
    if not module_logger.hasHandlers():
        # If basicConfig didn't set up for this specific logger, add a handler.
        # This can happen if another part of the code (or an imported library)
        # already called basicConfig.
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        module_logger.addHandler(console_handler)
        module_logger.setLevel(logging.INFO) # Ensure the level is set for this logger
        module_logger.propagate = False # Avoid duplicate messages if root logger also has a handler

    module_logger.info("Running detect_emotion.py directly for testing audio transcription...")
    
    # Call the test function
    # Common model names: "tiny.en", "base.en", "small.en", "medium.en"
    audio_file_path, result = test_single_audio_transcription(duration=10, model_name="base.en")
    
    if result:
        module_logger.info(f"Transcription Result: '{result}'")
    else:
        module_logger.warning("Transcription failed or was empty.")
    
    if audio_file_path:
        module_logger.info(f"Test audio was saved to: {audio_file_path}")
        module_logger.info("Please listen to this file to check the audio quality.")
