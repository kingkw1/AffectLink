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

# Constants
VIDEO_WINDOW_DURATION = 5  # seconds
AUDIO_WINDOW_DURATION = 5  # seconds

# Model IDs
TEXT_CLASSIFIER_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SER_MODEL_ID = "superb/hubert-large-superb-er"

# Emotion categories for unified mapping
UNIFIED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry']

# Emotion mapping dictionaries
TEXT_TO_UNIFIED = {
    'neutral': 'neutral',
    'joy': 'happy',
    'sadness': 'sad',
    'anger': 'angry',
    'fear': None,  # These emotions don't map to our unified set
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
        with torch.no_grad():
            outputs = ser_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            pred_idx = scores.argmax().item()
            pred_label = ser_label_mapping[pred_idx]
            pred_score = scores[pred_idx].item()
        return pred_label, pred_score
    except Exception as e:
        print(f"Audio SER error: {e}")
        return None, None

def analyze_audio_emotion_full(audio_path, ser_model, ser_processor, ser_label_mapping, device):
    try:
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        inputs = ser_processor(waveform, sampling_rate=16000, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        with torch.no_grad():
            outputs = ser_model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            all_scores = {ser_label_mapping[i]: float(scores[i].item()) for i in range(len(scores))}
            pred_idx = scores.argmax().item()
            pred_label = ser_label_mapping[pred_idx]
            pred_score = scores[pred_idx].item()
        return pred_label, pred_score, all_scores
    except Exception as e:
        print(f"Audio SER error: {e}")
        return None, None, {}

def record_audio_chunk(duration=5, fs=16000):
    """
    Record audio from the microphone for a given duration (in seconds).
    Returns the path to a temporary WAV file.
    """
    print(f"Recording {duration} seconds of audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    with wave.open(temp_wav.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    return temp_wav.name

def moving_average(scores):
    """
    Compute the moving average of a list of scores.
    """
    if len(scores) == 0:
        return 0
    return sum(scores) / len(scores)

def match_multimodal_emotions(video_emotions, audio_emotions, time_threshold=1.0):
    """
    Match detected facial emotions with detected audio emotions based on timestamp proximity.
    Calculates cosine similarity between emotion vectors for better consistency measurement.
    
    Args:
        video_emotions: list of dicts with 'timestamp', 'emotion', 'confidence'
        audio_emotions: list of dicts with 'timestamp', 'modality', 'emotion', 'confidence'
        time_threshold: max allowed time difference (seconds) for a match
    Returns:
        List of dicts with matched emotion data from both modalities.
    """
    matches = []
    
    # Group audio modalities by timestamp to find text and audio pairs
    audio_by_timestamp = {}
    for a in audio_emotions:
        timestamp = a['timestamp']
        if timestamp not in audio_by_timestamp:
            audio_by_timestamp[timestamp] = []
        audio_by_timestamp[timestamp].append(a)
        
    for v in video_emotions:
        v_timestamp = v['timestamp']
        
        # Find matching audio entries (both text and SER) near this video timestamp
        matching_timestamps = [ts for ts in audio_by_timestamp.keys() if abs(v_timestamp - ts) <= time_threshold]
        
        for ts in matching_timestamps:
            entries = audio_by_timestamp[ts]
            
            # Extract text and audio modality entries
            text_entry = next((e for e in entries if e['modality'] == 'text'), None)
            audio_entry = next((e for e in entries if e['modality'] == 'audio'), None)
            
            # Calculate video vector once
            video_vector = create_unified_emotion_vector(v.get('emotion_scores', {}), FACIAL_TO_UNIFIED)
            
            # Calculate text and audio vectors if available
            text_vector = None
            if text_entry:
                text_vector = create_unified_emotion_vector(text_entry.get('emotion_scores', []), TEXT_TO_UNIFIED)
                
            audio_vector = None
            if audio_entry:
                audio_vector = create_unified_emotion_vector(audio_entry.get('emotion_scores', {}), SER_TO_UNIFIED)
            
            # Calculate pairwise similarities
            video_text_sim = calculate_cosine_similarity(video_vector, text_vector) if text_vector is not None else None
            video_audio_sim = calculate_cosine_similarity(video_vector, audio_vector) if audio_vector is not None else None
            text_audio_sim = calculate_cosine_similarity(text_vector, audio_vector) if text_vector is not None and audio_vector is not None else None
            
            # Calculate aggregate similarity
            cosine_similarities = [s for s in [video_text_sim, video_audio_sim, text_audio_sim] if s is not None]
            aggregate_similarity = sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0
            
            # Create match entries for each modality pair
            match_data = {
                'video_timestamp': v_timestamp,
                'facial_emotion': v['emotion'],
                'facial_confidence': v['confidence'],
                'video_emotion_scores': v.get('emotion_scores', {}),
                'video_emotion_vector': video_vector,
                'cosine_similarity': aggregate_similarity,
                'pairwise_similarities': {
                    'video_text': video_text_sim,
                    'video_audio': video_audio_sim, 
                    'text_audio': text_audio_sim
                }
            }
            
            # Add text data if available
            if text_entry:
                match_data.update({
                    'text_timestamp': text_entry['timestamp'],
                    'text_emotion': text_entry['emotion'],
                    'text_confidence': text_entry['confidence'],
                    'text_emotion_scores': text_entry.get('emotion_scores', []),
                    'text_emotion_vector': text_vector
                })
            
            # Add audio data if available
            if audio_entry:
                match_data.update({
                    'audio_timestamp': audio_entry['timestamp'],
                    'audio_emotion': audio_entry['emotion'],
                    'audio_confidence': audio_entry['confidence'],
                    'audio_emotion_scores': audio_entry.get('emotion_scores', {}),
                    'audio_emotion_vector': audio_vector
                })
                
            # For compatibility with existing code, keep the original format
            if text_entry:
                match_data.update({
                    'audio_timestamp': text_entry['timestamp'],
                    'audio_modality': 'text',
                    'audio_emotion': text_entry['emotion'],
                    'audio_confidence': text_entry['confidence'],
                    'audio_emotion_scores': text_entry.get('emotion_scores', []),
                })
            elif audio_entry:
                match_data.update({
                    'audio_timestamp': audio_entry['timestamp'],
                    'audio_modality': 'audio',
                    'audio_emotion': audio_entry['emotion'],
                    'audio_confidence': audio_entry['confidence'],
                    'audio_emotion_scores': audio_entry.get('emotion_scores', {}),
                })
                
            matches.append(match_data)
    
    return matches

def video_processing_loop(video_emotions, video_lock, stop_flag, video_started_event):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        stop_flag['stop'] = True
        return
    # Signal that video processing has started
    video_started_event.set()
    
    # For 5-second downsampling
    window_start_time = time.time()
    frame_emotions = []
    
    while not stop_flag['stop']:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read video frame.")
            break
        
        try:
            # Process this frame with DeepFace
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            faces = results if isinstance(results, list) else [results]
            
            for face in faces:
                if 'dominant_emotion' in face:
                    current_time = time.time()
                    emo = face['dominant_emotion']
                    confidence = face.get('emotion', {}).get(emo, None)
                    emotion_scores = face.get('emotion', {})
                    
                    # Store this frame's emotion data for the current window
                    frame_emotions.append({
                        'timestamp': current_time,
                        'emotion': emo,
                        'confidence': confidence,
                        'emotion_scores': emotion_scores
                    })
                    
                    # Draw rectangle and overlay text
                    region = face.get('region', {})
                    x, y, w, h = region.get('x',0), region.get('y',0), region.get('w',0), region.get('h',0)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    text_y = y-10 if y-10>10 else y+h+20
                    cv2.putText(frame, f"{emo}", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                else:
                    print("No face detected or emotion data unavailable.")
                    
            # Check if we've reached the end of a 5-second window
            current_time = time.time()
            if current_time - window_start_time >= VIDEO_WINDOW_DURATION and frame_emotions:
                # Calculate average scores for each emotion category
                unified_emotion_scores = {emotion: 0.0 for emotion in UNIFIED_EMOTIONS}
                count = 0
                
                for frame_data in frame_emotions:
                    raw_scores = frame_data.get('emotion_scores', {})
                    # Map DeepFace emotions to our unified set and accumulate scores
                    if 'neutral' in raw_scores:
                        unified_emotion_scores['neutral'] += raw_scores.get('neutral', 0)
                    if 'happy' in raw_scores:
                        unified_emotion_scores['happy'] += raw_scores.get('happy', 0)
                    if 'sad' in raw_scores:
                        unified_emotion_scores['sad'] += raw_scores.get('sad', 0)
                    if 'angry' in raw_scores:
                        unified_emotion_scores['angry'] += raw_scores.get('angry', 0)
                    count += 1
                
                if count > 0:
                    # Average the scores
                    for emotion in unified_emotion_scores:
                        unified_emotion_scores[emotion] /= count
                    
                    # Find the dominant emotion from the averaged scores
                    dominant_emotion = max(unified_emotion_scores.items(), key=lambda x: x[1])
                    dominant_label = dominant_emotion[0]
                    dominant_score = dominant_emotion[1]
                    
                    # Create the aggregated video emotion entry
                    aggregated_entry = {
                        'timestamp': current_time,  # End of the 5-second window
                        'emotion': dominant_label,
                        'confidence': dominant_score,
                        'emotion_scores': unified_emotion_scores
                    }
                    
                    # Add to video emotions log with lock
                    with video_lock:
                        video_emotions.append(aggregated_entry)
                
                # Reset for next window
                window_start_time = current_time
                frame_emotions = []
                
        except Exception as e:
            print(f"Video analysis error: {e}")
            
        cv2.imshow('Real-time Video Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag['stop'] = True
            break
            
    cap.release()
    cv2.destroyAllWindows()

def audio_processing_loop(audio_emotion_log, audio_lock, stop_flag, whisper_model, classifier, ser_model, ser_processor, ser_label_mapping, device, video_started_event):
    # Wait for video processing to start
    video_started_event.wait()
    chunk_duration = 5
    smoothing_window = 3
    emotion_window = deque(maxlen=smoothing_window)
    score_window = deque(maxlen=smoothing_window)
    audio_emotion_window = deque(maxlen=smoothing_window)
    audio_score_window = deque(maxlen=smoothing_window)
    while not stop_flag['stop']:
        temp_wav = record_audio_chunk(duration=chunk_duration)
        text = transcribe_audio_whisper(temp_wav, whisper_model)

        # Get all text emotion scores
        text_emotion_scores_raw = classifier(text, top_k=None) if text else [] # Get raw output

        text_emotion_scores = []
        if text and text_emotion_scores_raw and isinstance(text_emotion_scores_raw, list) and len(text_emotion_scores_raw) > 0:
            # The classifier for "j-hartmann/emotion-english-distilroberta-base" returns a list containing a list of dicts
            if isinstance(text_emotion_scores_raw[0], list):
                 text_emotion_scores = sorted(text_emotion_scores_raw[0], key=lambda x: x['score'], reverse=True)
            elif isinstance(text_emotion_scores_raw[0], dict): # Fallback if the structure is flatter
                 text_emotion_scores = sorted(text_emotion_scores_raw, key=lambda x: x['score'], reverse=True)

        if text_emotion_scores:
            top_text = text_emotion_scores[0]
            emotion = top_text['label']
            score = top_text['score']
        else:
            emotion, score = None, None
        text_timestamp = time.time()
        # SER all scores
        audio_emotion, audio_score, audio_emotion_scores = analyze_audio_emotion_full(temp_wav, ser_model, ser_processor, ser_label_mapping, device)
        audio_timestamp = time.time()
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

def calculate_cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vector_a: First vector as a numpy array
        vector_b: Second vector as a numpy array
        
    Returns:
        Cosine similarity value between -1 and 1
    """
    # Ensure vectors are numpy arrays
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    
    # Calculate dot product
    dot_product = np.dot(vector_a, vector_b)
    
    # Calculate magnitudes
    magnitude_a = np.sqrt(np.sum(vector_a**2))
    magnitude_b = np.sqrt(np.sum(vector_b**2))
    
    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    # Calculate cosine similarity
    cosine_similarity = dot_product / (magnitude_a * magnitude_b)
    return cosine_similarity

def create_unified_emotion_vector(emotion_scores, mapping_dict):
    """
    Create a vector representing emotion scores in the unified emotion space.
    
    Args:
        emotion_scores: Dictionary or list of dicts with emotion scores
        mapping_dict: Dictionary mapping source emotions to unified emotions
        
    Returns:
        List with scores for each unified emotion in the order of UNIFIED_EMOTIONS
    """
    # Initialize the unified vector with zeros
    unified_vector = [0.0] * len(UNIFIED_EMOTIONS)
    
    # Handle different input formats
    if isinstance(emotion_scores, dict):
        # Direct dictionary of emotion scores
        for emotion, score in emotion_scores.items():
            if emotion in mapping_dict and mapping_dict[emotion] is not None:
                unified_emotion = mapping_dict[emotion]
                if unified_emotion in UNIFIED_EMOTIONS:
                    idx = UNIFIED_EMOTIONS.index(unified_emotion)
                    unified_vector[idx] = score
    elif isinstance(emotion_scores, list):
        # List of emotion score dictionaries (from text classifier)
        for item in emotion_scores:
            if isinstance(item, dict) and 'label' in item and 'score' in item:
                emotion = item['label']
                score = item['score']
                if emotion in mapping_dict and mapping_dict[emotion] is not None:
                    unified_emotion = mapping_dict[emotion]
                    if unified_emotion in UNIFIED_EMOTIONS:
                        idx = UNIFIED_EMOTIONS.index(unified_emotion)
                        unified_vector[idx] = score
    
    return unified_vector

# ---------------------------
# Main script
# ---------------------------
# Suppress DeepFace logging for cleaner console output
logging.getLogger().setLevel(logging.ERROR)

def main(live=True):
    # Set detection mode based on argument
    print(f"Detection mode: {'live' if live else 'video_file'}")
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
        print("Video File Emotion Detection Mode")
        video_path = input("Enter path to video file: ").strip()
        if not os.path.isfile(video_path):
            print("File not found.")
            return
        # Extract audio from video file
        print("Extracting audio from video file...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            audio_path = temp_audio.name
        try:
            videoclip = VideoFileClip(video_path)
            videoclip.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le')
        except Exception as e:
            print(f"Audio extraction failed: {e}")
            return
        # --- Video frame analysis ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Cannot open video file.")
            return
        
        # Initialize variables for 5-second downsampling
        video_emotions = []
        frame_emotions = []
        window_start_time = time.time()
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        print("Processing video file for facial emotions with 5-second downsampling...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Calculate timestamp based on frame count and FPS
            timestamp = window_start_time + (frame_count / fps)
            frame_count += 1
            
            try:
                results = DeepFace.analyze(
                    img_path=frame,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                faces = results if isinstance(results, list) else [results]
                for face in faces:
                    if 'dominant_emotion' in face:
                        emo = face['dominant_emotion']
                        confidence = face.get('emotion', {}).get(emo, None)
                        emotion_scores = face.get('emotion', {})
                        
                        # Store this frame's emotion data for current window
                        frame_emotions.append({
                            'timestamp': timestamp,
                            'emotion': emo,
                            'confidence': confidence,
                            'emotion_scores': emotion_scores
                        })
                    else:
                        print("No face detected or emotion data unavailable.")
            except Exception as e:
                print(f"Video analysis error: {e}")
                
            # Check if we've processed enough frames for a 5-second window
            if timestamp - window_start_time >= VIDEO_WINDOW_DURATION and frame_emotions:
                # Calculate average scores for each emotion category
                unified_emotion_scores = {emotion: 0.0 for emotion in UNIFIED_EMOTIONS}
                count = 0
                
                for frame_data in frame_emotions:
                    raw_scores = frame_data.get('emotion_scores', {})
                    # Map DeepFace emotions to our unified set and accumulate scores
                    if 'neutral' in raw_scores:
                        unified_emotion_scores['neutral'] += raw_scores.get('neutral', 0)
                    if 'happy' in raw_scores:
                        unified_emotion_scores['happy'] += raw_scores.get('happy', 0)
                    if 'sad' in raw_scores:
                        unified_emotion_scores['sad'] += raw_scores.get('sad', 0)
                    if 'angry' in raw_scores:
                        unified_emotion_scores['angry'] += raw_scores.get('angry', 0)
                    count += 1
                
                if count > 0:
                    # Average the scores
                    for emotion in unified_emotion_scores:
                        unified_emotion_scores[emotion] /= count
                    
                    # Find the dominant emotion from the averaged scores
                    dominant_emotion = max(unified_emotion_scores.items(), key=lambda x: x[1])
                    dominant_label = dominant_emotion[0]
                    dominant_score = dominant_emotion[1]
                    
                    # Create the aggregated video emotion entry
                    aggregated_entry = {
                        'timestamp': timestamp,  # End of the 5-second window
                        'emotion': dominant_label,
                        'confidence': dominant_score,
                        'emotion_scores': unified_emotion_scores
                    }
                    
                    video_emotions.append(aggregated_entry)
                
                # Reset for next window
                window_start_time = timestamp
                frame_emotions = []
                
        cap.release()
        print("Video file processing complete. Total 5-second windows analyzed:", len(video_emotions))
        # --- Audio chunked processing ---
        print("Processing extracted audio for emotions in chunks...")
        chunk_duration = 10  # seconds
        audio_emotion_log = []
        y, sr = librosa.load(audio_path, sr=16000)
        total_samples = len(y)
        chunk_samples = chunk_duration * sr
        num_chunks = math.ceil(total_samples / chunk_samples)
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min((i + 1) * chunk_samples, total_samples)
            chunk = y[start:end]
            # Write chunk to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_chunk:
                sf.write(temp_chunk.name, chunk, sr)
                chunk_path = temp_chunk.name
            # Transcribe chunk
            text = transcribe_audio_whisper(chunk_path, whisper_model)
            text_emotion_scores = classify_emotion_full(text, classifier) if text else []
            if text_emotion_scores:
                top_text = text_emotion_scores[0]
                text_emotion = top_text['label']
                text_score = top_text['score']
            else:
                text_emotion, text_score = None, None
            text_timestamp = time.time()
            audio_emotion, audio_score, audio_emotion_scores = analyze_audio_emotion_full(chunk_path, ser_model, ser_processor, ser_label_mapping, device)
            audio_timestamp = time.time()
            if text_emotion:
                audio_emotion_log.append({
                    'timestamp': text_timestamp,
                    'modality': 'text',
                    'emotion': text_emotion,
                    'confidence': text_score,
                    'emotion_scores': text_emotion_scores
                })
            if audio_emotion:
                audio_emotion_log.append({
                    'timestamp': audio_timestamp,
                    'modality': 'audio',
                    'emotion': audio_emotion,
                    'confidence': audio_score,
                    'emotion_scores': audio_emotion_scores
                })
            # Clean up chunk file
            os.remove(chunk_path)

        # Clean up temp audio file
        os.remove(audio_path)
    else:
        print("Starting live microphone and video emotion detection (threaded). Press Ctrl+C to stop.")
        video_emotions = []
        audio_emotion_log = []
        video_lock = threading.Lock()
        audio_lock = threading.Lock()
        stop_flag = {'stop': False}
        video_started_event = threading.Event()
        # Start threads
        video_thread = threading.Thread(target=video_processing_loop, args=(video_emotions, video_lock, stop_flag, video_started_event))
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
                    print("\n--- Multimodal Matches (real-time, threaded) ---")
                    for m in matches[-5:]:
                        
                        # Print cosine similarity
                        print(f"    Cosine Similarity (aggregate): {m['cosine_similarity']:.3f}")
                        
                        # Print detailed pairwise similarities
                        if 'pairwise_similarities' in m:
                            ps = m['pairwise_similarities']
                            if ps.get('video_text') is not None:
                                print(f"      Video-Text Similarity: {ps['video_text']:.3f}")
                            if ps.get('video_audio') is not None:
                                print(f"      Video-Audio Similarity: {ps['video_audio']:.3f}")
                            if ps.get('text_audio') is not None:
                                print(f"      Text-Audio Similarity: {ps['text_audio']:.3f}")
                                
                    # Calculate average cosine similarity for recent matches
                    window_size = 3
                    window_matches = matches[-window_size:] if len(matches) >= window_size else matches
                    avg_cosine_sim = sum(match['cosine_similarity'] for match in window_matches) / len(window_matches) if window_matches else 0
                    print(f"Average Cosine Similarity (last {len(window_matches)}): {avg_cosine_sim:.3f}")
                    
                    # Refined windowed consistency metric
                    consistent_count = 0
                    for match in window_matches:
                        if match['facial_emotion'] == match['audio_emotion']:
                            consistent_count += 1
                    if window_matches:
                        consistency_pct = 100.0 * consistent_count / len(window_matches)
                        print(f"Label Consistency (last {len(window_matches)}): {consistency_pct:.1f}%")
                    else:
                        print("No matches yet.")
                        
                    # Mismatch indicator for the most recent match
                    latest = matches[-1]
                    
                    # Consistency indicator
                    cosine_sim = latest['cosine_similarity']
                    if cosine_sim >= 0.8:
                        consistency_level = "High Consistency ✅✅"
                    elif cosine_sim >= 0.6:
                        consistency_level = "Moderate Consistency ✅"
                    elif cosine_sim >= 0.3:
                        consistency_level = "Low Consistency ⚠️"
                    else:
                        consistency_level = "Inconsistent ❌"
                    
                    print(f"Cosine Similarity: {cosine_sim:.3f} - {consistency_level}")
        except KeyboardInterrupt:
            print("Exiting microphone and video emotion detection.")
            stop_flag['stop'] = True
        video_thread.join()
        audio_thread.join()

if __name__ == "__main__":
    main()