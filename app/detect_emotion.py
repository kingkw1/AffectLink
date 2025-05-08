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

# Constants
VIDEO_WINDOW_DURATION = 5  # seconds
AUDIO_WINDOW_DURATION = 5  # seconds

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
    Args:
        video_emotions: list of dicts with 'timestamp', 'emotion', 'confidence'
        audio_emotions: list of dicts with 'timestamp', 'modality', 'emotion', 'confidence'
        time_threshold: max allowed time difference (seconds) for a match
    Returns:
        List of dicts with matched emotion data from both modalities.
    """
    matches = []
    for v in video_emotions:
        for a in audio_emotions:
            if abs(v['timestamp'] - a['timestamp']) <= time_threshold:
                matches.append({
                    'video_timestamp': v['timestamp'],
                    'facial_emotion': v['emotion'],
                    'facial_confidence': v['confidence'],
                    'video_emotion_scores': v.get('emotion_scores', {}),
                    'audio_timestamp': a['timestamp'],
                    'audio_modality': a['modality'],
                    'audio_emotion': a['emotion'],
                    'audio_confidence': a['confidence'],
                    'audio_emotion_scores': a.get('emotion_scores', [])
                })
    return matches

def video_processing_loop(video_emotions, video_lock, stop_flag, video_started_event):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        stop_flag['stop'] = True
        return
    # Signal that video processing has started
    video_started_event.set()
    while not stop_flag['stop']:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read video frame.")
            break
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
                    timestamp = time.time()
                    with video_lock:
                        video_emotions.append({
                            'timestamp': timestamp,
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
                    # print(f"[Video {timestamp:.3f}] Detected emotion: {emo} (confidence: {confidence})")
                else:
                    print("No face detected or emotion data unavailable.")
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
        print(f"DEBUG: Whisper transcribed text: '{text}'") # Added for debugging

        # Get all text emotion scores
        print(f"DEBUG: Text input to emotion classifier: '{text}'") # Added for debugging
        text_emotion_scores_raw = classifier(text, top_k=None) if text else [] # Get raw output
        print(f"DEBUG: Raw output from text emotion classifier: {text_emotion_scores_raw}") # Added for debugging

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
            print(f"DEBUG: Adding to audio_emotion_log (text): {log_entry_text}") # Added for debugging
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
        print("--- Results ---")
        if emotion:
            print(f"[{text_timestamp:.3f}] [Text]    Smoothed emotion: {smoothed_emotion} (confidence: {smoothed_score:.2f})")
        else:
            print("[Text]    Could not detect emotion.")
        if audio_emotion:
            print(f"[{audio_timestamp:.3f}] [Audio]   Smoothed emotion: {smoothed_audio_emotion} (confidence: {smoothed_audio_score:.2f})")
        else:
            print("[Audio]   Could not detect emotion.")

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
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )
    ser_model_id = "superb/hubert-large-superb-er"
    print("Loading audio-based SER model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ser_model = AutoModelForAudioClassification.from_pretrained(ser_model_id).to(device)
    ser_processor = AutoFeatureExtractor.from_pretrained(ser_model_id)
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
        video_emotions = []
        print("Processing video file for facial emotions...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
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
                        timestamp = time.time()
                        video_emotions.append({
                            'timestamp': timestamp,
                            'emotion': emo,
                            'confidence': confidence
                        })
                        print(f"[Video {timestamp:.3f}] Detected emotion: {emo} (confidence: {confidence})")
                    else:
                        print("No face detected or emotion data unavailable.")
            except Exception as e:
                print(f"Video analysis error: {e}")
        cap.release()
        print("Video file processing complete. Total frames analyzed:", len(video_emotions))
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
            # Print results for this chunk
            print(f"--- Audio Results (chunk {i+1}/{num_chunks}) ---")
            if text_emotion and text_timestamp:
                print(f"[{text_timestamp:.3f}] [Text]    Detected emotion: {text_emotion} (confidence: {text_score:.2f})")
                print("    Text emotion scores:")
                for e in text_emotion_scores[:3]:
                    print(f"      {e['label']}: {e['score']:.2f}")
            else:
                print("[Text]    Could not detect emotion.")
            if audio_emotion and audio_timestamp:
                print(f"[{audio_timestamp:.3f}] [Audio]   Detected emotion: {audio_emotion} (confidence: {audio_score:.2f})")
                print("    Audio SER scores:")
                for k, v in sorted(audio_emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"      {k}: {v:.2f}")
            else:
                print("[Audio]   Could not detect emotion.")
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

                print(f"DEBUG: audio_emotion_log content: {audio_emotion_log}") # Added for debugging
                matches = match_multimodal_emotions(video_window, audio_window)
                if matches:
                    print("\n--- Multimodal Matches (real-time, threaded) ---")
                    for m in matches[-5:]:
                        # print(f"DEBUG: Current complete match object m: {m}") # Optional: Full match object debug
                        print(f"[t={m['video_timestamp']:.3f}] Video: {m['facial_emotion']} ({m['facial_confidence']:.2f}) | "
                              f"Audio({m['audio_modality']}): {m['audio_emotion']} ({m['audio_confidence']:.2f}) @ t={m['audio_timestamp']:.3f}")
                        # Print top 3 video emotion scores
                        print("    Video emotion scores:")
                        if isinstance(m.get('video_emotion_scores'), dict):
                            for k, v in sorted(m['video_emotion_scores'].items(), key=lambda x: x[1], reverse=True)[:3]:
                                print(f"      {k}: {v:.2f}")
                        else:
                            print("      (no video scores found or in unexpected format)")

                        # Print top 3 audio emotion scores
                        if m['audio_modality'] == 'audio':
                            print(f"    Audio (audio) emotion scores:")
                            if isinstance(m.get('audio_emotion_scores'), dict):
                                for k, v in sorted(m['audio_emotion_scores'].items(), key=lambda x: x[1], reverse=True)[:3]:
                                    print(f"      {k}: {v:.2f}")
                            else:
                                print("      (no audio scores found or in unexpected format)")
                        elif m['audio_modality'] == 'text':
                            print(f"    Audio (text) emotion scores:")
                            # print(f"DEBUG: audio_emotion_scores for text modality: {m.get('audio_emotion_scores')}") # Debug specific scores
                            retrieved_text_scores = m.get('audio_emotion_scores', [])
                            if isinstance(retrieved_text_scores, list) and retrieved_text_scores:
                                for score_entry in retrieved_text_scores[:3]: # Iterate through top 3 (already sorted)
                                    if isinstance(score_entry, dict) and 'label' in score_entry and 'score' in score_entry:
                                        print(f"      {score_entry['label']}: {score_entry['score']:.2f}")
                                    else:
                                        print(f"      (malformed score entry: {score_entry})")
                            else:
                                print("      (no text scores found or scores in unexpected format)")
                    # Refined windowed consistency metric
                    window_size = 3
                    window_matches = matches[-window_size:] if len(matches) >= window_size else matches
                    consistent_count = 0
                    for match in window_matches:
                        if match['facial_emotion'] == match['audio_emotion']:
                            consistent_count += 1
                    if window_matches:
                        consistency_pct = 100.0 * consistent_count / len(window_matches)
                        print(f"Consistency (last {len(window_matches)}): {consistency_pct:.1f}%")
                    else:
                        print("No matches yet.")
                    # Mismatch indicator for the most recent match
                    latest = matches[-1]
                    facial = latest['facial_emotion']
                    audio = latest['audio_emotion']
                    print(f"Dominant Facial Emotion: {facial}")
                    print(f"Dominant Audio Emotion: {audio}")
                    if (facial == audio):
                        print("Consistency: Consistent ✅")
                    else:
                        print("Consistency: Mismatch ❌")
        except KeyboardInterrupt:
            print("Exiting microphone and video emotion detection.")
            stop_flag['stop'] = True
        video_thread.join()
        audio_thread.join()

if __name__ == "__main__":
    main()