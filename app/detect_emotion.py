import sys
import os
import tempfile
import wave
from collections import deque
import time

import torch
import whisper
from transformers import pipeline, AutoModelForAudioClassification
import sounddevice as sd
import numpy as np
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
                    'audio_timestamp': a['timestamp'],
                    'audio_modality': a['modality'],
                    'audio_emotion': a['emotion'],
                    'audio_confidence': a['confidence']
                })
    return matches

def video_processing_loop(video_emotions, video_lock, stop_flag):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        stop_flag['stop'] = True
        return
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
                    timestamp = time.time()
                    with video_lock:
                        video_emotions.append({
                            'timestamp': timestamp,
                            'emotion': emo,
                            'confidence': confidence
                        })
                    # Draw rectangle and overlay text
                    region = face.get('region', {})
                    x, y, w, h = region.get('x',0), region.get('y',0), region.get('w',0), region.get('h',0)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    text_y = y-10 if y-10>10 else y+h+20
                    cv2.putText(frame, f"{emo}", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    print(f"[Video {timestamp:.3f}] Detected emotion: {emo} (confidence: {confidence})")
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

def audio_processing_loop(audio_emotion_log, audio_lock, stop_flag, whisper_model, classifier, ser_model, ser_processor, ser_label_mapping, device):
    chunk_duration = 5
    smoothing_window = 3
    emotion_window = deque(maxlen=smoothing_window)
    score_window = deque(maxlen=smoothing_window)
    audio_emotion_window = deque(maxlen=smoothing_window)
    audio_score_window = deque(maxlen=smoothing_window)
    while not stop_flag['stop']:
        temp_wav = record_audio_chunk(duration=chunk_duration)
        text = transcribe_audio_whisper(temp_wav, whisper_model)
        audio_emotion, audio_score = analyze_audio_emotion(temp_wav, ser_model, ser_processor, ser_label_mapping, device)
        audio_timestamp = time.time()
        if not text or text.strip() == "":
            print("No speech detected.")
            os.unlink(temp_wav)
            continue
        print(f"Transcribed: {text}")
        emotion, score = classify_emotion(text, classifier)
        text_timestamp = time.time()
        os.unlink(temp_wav)
        # Smoothing text emotions
        if emotion:
            emotion_window.append(emotion)
            score_window.append(score)
            smoothed_emotion = max(set(emotion_window), key=emotion_window.count)
            smoothed_score = moving_average([s for e, s in zip(emotion_window, score_window) if e == smoothed_emotion])
            with audio_lock:
                audio_emotion_log.append({
                    'timestamp': text_timestamp,
                    'modality': 'text',
                    'emotion': smoothed_emotion,
                    'confidence': smoothed_score
                })
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
                    'confidence': smoothed_audio_score
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
        import math
        import soundfile as sf
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
            text_emotion, text_score, text_timestamp = None, None, None
            if text and len(text) > 0:
                # Truncate text to 512 characters for classifier (or split if needed)
                text = text[:512]
                text_emotion, text_score = classify_emotion(text, classifier)
                text_timestamp = time.time()
                if text_emotion:
                    audio_emotion_log.append({
                        'timestamp': text_timestamp,
                        'modality': 'text',
                        'emotion': text_emotion,
                        'confidence': text_score
                    })
            # SER on chunk
            audio_emotion, audio_score, audio_timestamp = None, None, None
            audio_emotion, audio_score = analyze_audio_emotion(chunk_path, ser_model, ser_processor, ser_label_mapping, device)
            audio_timestamp = time.time()
            if audio_emotion:
                audio_emotion_log.append({
                    'timestamp': audio_timestamp,
                    'modality': 'audio',
                    'emotion': audio_emotion,
                    'confidence': audio_score
                })
            # Clean up chunk file
            os.remove(chunk_path)
            # Print results for this chunk
            print(f"--- Audio Results (chunk {i+1}/{num_chunks}) ---")
            if text_emotion and text_timestamp:
                print(f"[{text_timestamp:.3f}] [Text]    Detected emotion: {text_emotion} (confidence: {text_score:.2f})")
            else:
                print("[Text]    Could not detect emotion.")
            if audio_emotion and audio_timestamp:
                print(f"[{audio_timestamp:.3f}] [Audio]   Detected emotion: {audio_emotion} (confidence: {audio_score:.2f})")
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
        # Start threads
        video_thread = threading.Thread(target=video_processing_loop, args=(video_emotions, video_lock, stop_flag))
        audio_thread = threading.Thread(target=audio_processing_loop, args=(audio_emotion_log, audio_lock, stop_flag, whisper_model, classifier, ser_model, ser_processor, ser_label_mapping, device))
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
                        print(f"[t={m['video_timestamp']:.3f}] Video: {m['facial_emotion']} ({m['facial_confidence']}) | "
                              f"Audio({m['audio_modality']}): {m['audio_emotion']} ({m['audio_confidence']}) @ t={m['audio_timestamp']:.3f}")
        except KeyboardInterrupt:
            print("Exiting microphone and video emotion detection.")
            stop_flag['stop'] = True
        video_thread.join()
        audio_thread.join()

if __name__ == "__main__":
    main()