import sys
import os
import tempfile
import wave
from collections import deque

import torch
import whisper
from transformers import pipeline
import sounddevice as sd

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

# ---------------------------
# Main script
# ---------------------------
def main():
    print("Speech Emotion Detection")
    print("Select input source:")
    print("1. Audio file")
    print("2. Microphone (live)")
    source = input("Enter 1 or 2: ").strip()
    
    # Load models
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("base")
    print("Loading emotion classification model...")
    classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=0 if torch.cuda.is_available() else -1
    )
    
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
        emotion, score = classify_emotion(text, classifier)
        if emotion:
            print(f"Detected emotion: {emotion} (confidence: {score:.2f})")
        else:
            print("Could not detect emotion.")
    elif source == '2':
        chunk_duration = 5
        smoothing_window = 3
        emotion_window = deque(maxlen=smoothing_window)
        score_window = deque(maxlen=smoothing_window)
        print("Starting live microphone emotion detection. Press Ctrl+C to stop.")
        try:
            while True:
                temp_wav = record_audio_chunk(duration=chunk_duration)
                text = transcribe_audio_whisper(temp_wav, whisper_model)
                os.unlink(temp_wav)
                if not text or text.strip() == "":
                    print("No speech detected.")
                    continue
                print(f"Transcribed: {text}")
                emotion, score = classify_emotion(text, classifier)
                if emotion:
                    emotion_window.append(emotion)
                    score_window.append(score)
                    smoothed_emotion = max(set(emotion_window), key=emotion_window.count)
                    smoothed_score = moving_average([s for e, s in zip(emotion_window, score_window) if e == smoothed_emotion])
                    print(f"Smoothed emotion: {smoothed_emotion} (confidence: {smoothed_score:.2f})")
                else:
                    print("Could not detect emotion.")
        except KeyboardInterrupt:
            print("Exiting microphone emotion detection.")
    else:
        print("Invalid input. Exiting.")

if __name__ == "__main__":
    main()