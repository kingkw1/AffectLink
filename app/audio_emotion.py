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

# ---------------------------
# Main script
# ---------------------------
def main():
    print("Speech Emotion Detection (Text & Audio)")
    print("Select input source:")
    print("1. Audio file")
    print("2. Microphone (live)")
    source = input("Enter 1 or 2: ").strip()
    
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

    # List to store timestamped audio emotion results
    audio_emotion_log = []
    
    if source == '1':
        audio_path = input("Enter path to audio file: ").strip()
        if not os.path.isfile(audio_path):
            print("File not found.")
            sys.exit(1)
        print(f"Transcribing audio file: {audio_path}")
        text = transcribe_audio_whisper(audio_path, whisper_model)
        if not text:
            print("No transcription available.")
            sys.exit(1)
        print(f"Transcribed text: {text}")
        # Text-based emotion
        text_emotion, text_score = classify_emotion(text, classifier)
        text_timestamp = time.time()
        if text_emotion:
            audio_emotion_log.append({
                'timestamp': text_timestamp,
                'modality': 'text',
                'emotion': text_emotion,
                'confidence': text_score
            })
        # Audio-based emotion
        audio_emotion, audio_score = analyze_audio_emotion(audio_path, ser_model, ser_processor, ser_label_mapping, device)
        audio_timestamp = time.time()
        if audio_emotion:
            audio_emotion_log.append({
                'timestamp': audio_timestamp,
                'modality': 'audio',
                'emotion': audio_emotion,
                'confidence': audio_score
            })
        print("--- Results ---")
        if text_emotion:
            print(f"[{text_timestamp:.3f}] [Text]    Detected emotion: {text_emotion} (confidence: {text_score:.2f})")
        else:
            print("[Text]    Could not detect emotion.")
        if audio_emotion:
            print(f"[{audio_timestamp:.3f}] [Audio]   Detected emotion: {audio_emotion} (confidence: {audio_score:.2f})")
        else:
            print("[Audio]   Could not detect emotion.")
        # Optionally, print or save the log for later use
        # print(audio_emotion_log)
        # Simulate video_emotions for testing (timestamps near audio_emotion_log)
        now = time.time()
        video_emotions = [
            {'timestamp': text_timestamp-0.5, 'emotion': 'neutral', 'confidence': 0.7},
            {'timestamp': audio_timestamp, 'emotion': 'happy', 'confidence': 0.8}
        ]
        print("\n--- Multimodal Matches (simulated video) ---")
        matches = match_multimodal_emotions(video_emotions, audio_emotion_log)
        for m in matches:
            print(f"[t={m['video_timestamp']:.3f}] Video: {m['facial_emotion']} ({m['facial_confidence']:.2f}) | "
                  f"Audio({m['audio_modality']}): {m['audio_emotion']} ({m['audio_confidence']:.2f}) @ t={m['audio_timestamp']:.3f}")
    elif source == '2':
        chunk_duration = 5
        smoothing_window = 3
        emotion_window = deque(maxlen=smoothing_window)
        score_window = deque(maxlen=smoothing_window)
        audio_emotion_window = deque(maxlen=smoothing_window)
        audio_score_window = deque(maxlen=smoothing_window)
        print("Starting live microphone emotion detection. Press Ctrl+C to stop.")
        # Simulate video_emotions for live mode (append a new entry per chunk)
        video_emotions = []
        try:
            while True:
                temp_wav = record_audio_chunk(duration=chunk_duration)

                # Transcribe audio chunk
                text = transcribe_audio_whisper(temp_wav, whisper_model)

                # Perform audio-based SER first (timestamp after analysis)
                audio_emotion, audio_score = analyze_audio_emotion(temp_wav, ser_model, ser_processor, ser_label_mapping, device)
                audio_timestamp = time.time()  # timestamp right after audio SER

                # Transcribe text and classify emotion
                if not text or text.strip() == "":
                    print("No speech detected.")
                    os.unlink(temp_wav)
                    continue
                print(f"Transcribed: {text}")

                emotion, score = classify_emotion(text, classifier)
                text_timestamp = time.time()  # timestamp right after text classification

                os.unlink(temp_wav)

                # Smoothing text emotions
                if emotion:
                    emotion_window.append(emotion)
                    score_window.append(score)
                    smoothed_emotion = max(set(emotion_window), key=emotion_window.count)
                    smoothed_score = moving_average([s for e, s in zip(emotion_window, score_window) if e == smoothed_emotion])
                    audio_emotion_log.append({
                        'timestamp': text_timestamp,
                        'modality': 'text',
                        'emotion': smoothed_emotion,
                        'confidence': smoothed_score
                    })
                else:
                    smoothed_emotion = None
                    smoothed_score = 0

                # Smoothing audio emotions
                if audio_emotion:
                    audio_emotion_window.append(audio_emotion)
                    audio_score_window.append(audio_score)
                    smoothed_audio_emotion = max(set(audio_emotion_window), key=audio_emotion_window.count)
                    smoothed_audio_score = moving_average([s for e, s in zip(audio_emotion_window, audio_score_window) if e == smoothed_audio_emotion])
                    audio_emotion_log.append({
                        'timestamp': audio_timestamp,
                        'modality': 'audio',
                        'emotion': smoothed_audio_emotion,
                        'confidence': smoothed_audio_score
                    })
                else:
                    smoothed_audio_emotion = None
                    smoothed_audio_score = 0

                # Print results with their respective timestamps
                print("--- Results ---")
                if smoothed_emotion:
                    print(f"[{text_timestamp:.3f}] [Text]    Smoothed emotion: {smoothed_emotion} (confidence: {smoothed_score:.2f})")
                else:
                    print("[Text]    Could not detect emotion.")
                if smoothed_audio_emotion:
                    print(f"[{audio_timestamp:.3f}] [Audio]   Smoothed emotion: {smoothed_audio_emotion} (confidence: {smoothed_audio_score:.2f})")
                else:
                    print("[Audio]   Could not detect emotion.")

                # After each chunk, simulate a video emotion with a timestamp near the audio chunk
                simulated_video_emotion = {
                    'timestamp': time.time(),
                    'emotion': 'neutral',  # or random/constant for now
                    'confidence': 0.75
                }
                video_emotions.append(simulated_video_emotion)
                # Print matches for the most recent video emotion
                matches = match_multimodal_emotions([simulated_video_emotion], audio_emotion_log)
                if matches:
                    print("\n--- Multimodal Matches (simulated video) ---")
                    for m in matches:
                        print(f"[t={m['video_timestamp']:.3f}] Video: {m['facial_emotion']} ({m['facial_confidence']:.2f}) | "
                              f"Audio({m['audio_modality']}): {m['audio_emotion']} ({m['audio_confidence']:.2f}) @ t={m['audio_timestamp']:.3f}")
        except KeyboardInterrupt:
            print("Exiting microphone emotion detection.")
    else:
        print("Invalid input. Exiting.")

if __name__ == "__main__":
    main()