import os
import sys
import numpy as np
import soundfile as sf
import logging
import torch
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import whisper
import time # Added for timing model loading
import librosa # Added for loading audio from video
import tempfile
# Add the project root to sys.path to ensure local module imports work
# This assumes 'app' is at the same level as 'constants.py' or in a parent directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming project structure is: project_root/app/audio_emotion_processor.py
# And constants.py might be in project_root/constants.py or project_root/app/constants.py
project_root = os.path.dirname(current_dir)

# Add relevant paths if not already in sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
# If audio_emotion_processor.py is directly in 'app' and 'constants.py' is also in 'app'
if current_dir not in sys.path:
    sys.path.append(current_dir)
from audio_processor import transcribe_audio_whisper

# Set DeepFace logging to error to reduce noise, though not directly used here
logging.getLogger('deepface').setLevel(logging.ERROR)
# Configure a basic logger for this test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the new function and necessary constants
try:
    # Import constants from wherever they reside in your project
    # Adjust this import based on your actual constants.py location
    from constants import SER_TO_UNIFIED, TEXT_TO_UNIFIED, UNIFIED_EMOTIONS 
except ImportError as e:
    logging.error(f"Failed to import necessary modules. Ensure audio_emotion_processor.py and constants.py are accessible via sys.path. Error: {e}")
    logging.error(f"Current sys.path: {sys.path}")
    sys.exit(1)


# --- Configuration for Testing ---
# IMPORTANT: CHANGE THIS PATH to a valid audio file (e.g., .wav, .mp3) on your system.
# A clear voice speaking some words for 5-10 seconds would be ideal for testing transcription.
# AUDIO_FILE_PATH = "C:\\\\Users\\\\kingk\\\\OneDrive\\\\Documents\\\\Projects\\\\AffectLink\\\\data\\\\audio_test.wav" 
VIDEO_FILE_PATH = r"C:\Users\kingk\OneDrive\Documents\Projects\AffectLink\data\WIN_20250529_10_51_21_Pro.mp4"

# Model IDs (these should match what you use in main_processor.py)
TEXT_CLASSIFIER_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SER_MODEL_ID = "superb/hubert-large-superb-er"
WHISPER_MODEL_SIZE = "base" # Or "small", "medium", etc.

def load_models():
    """Loads all necessary AI models for audio processing."""
    logging.info("Starting model loading...")
    start_time = time.time()

    logging.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    logging.info("Whisper model loaded.")

    logging.info(f"Loading text emotion classifier: '{TEXT_CLASSIFIER_MODEL_ID}'...")
    text_emotion_classifier = pipeline("sentiment-analysis", model=TEXT_CLASSIFIER_MODEL_ID)
    logging.info("Text emotion classifier loaded.")

    logging.info(f"Loading audio feature extractor: '{SER_MODEL_ID}'...")
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL_ID)
    logging.info("Audio feature extractor loaded.")

    logging.info(f"Loading audio emotion classifier: '{SER_MODEL_ID}'...")
    audio_emotion_classifier = AutoModelForAudioClassification.from_pretrained(SER_MODEL_ID)
    logging.info("Audio emotion classifier loaded.")

    end_time = time.time()
    logging.info(f"All models loaded in {end_time - start_time:.2f} seconds.")
    return whisper_model, text_emotion_classifier, audio_feature_extractor, audio_emotion_classifier


def process_audio_chunk_from_file(
    audio_chunk_data,  # np.ndarray (float32)
    audio_sample_rate, # int
    whisper_model,
    text_emotion_classifier, # transformers.Pipeline
    audio_feature_extractor, # transformers.AutoFeatureExtractor
    audio_emotion_classifier # transformers.AutoModelForAudioClassification
):
    """
    Processes a single audio chunk from data for transcription and emotion analysis.
    Returns:
        A tuple: (transcribed_text, text_emotion_data, audio_emotion_data, text_emotion_full_scores, audio_emotion_full_scores)
    """
    transcribed_text = ""
    text_emotion_data = ("unknown", 0.0)
    audio_emotion_data = ("unknown", 0.0)
    text_emotion_full_scores = {}
    audio_emotion_full_scores = {}
    temp_audio_file_name = None

    try:
        # 1. Save audio_chunk_data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_audio_file_name = tmpfile.name
            sf.write(temp_audio_file_name, audio_chunk_data, audio_sample_rate)
        logger.debug(f"Temporarily saved audio chunk to {temp_audio_file_name}")

        # 2. Transcription
        try:
            raw_transcription = transcribe_audio_whisper(temp_audio_file_name, whisper_model)
            if raw_transcription == "RESET_BUFFER" or raw_transcription is None:
                transcribed_text = ""
            else:
                transcribed_text = raw_transcription
        except Exception as e_transcribe:
            logger.error(f"Error during transcription of {temp_audio_file_name}: {e_transcribe}", exc_info=True)
            transcribed_text = ""


        # 3. Text Emotion Classification
        if transcribed_text and transcribed_text.strip():
            try:
                text_results_outer = text_emotion_classifier(transcribed_text)

                if text_results_outer and isinstance(text_results_outer, list) and len(text_results_outer) > 0:
                    text_scores_list = text_results_outer[0] # This is the list of score dicts like [{'label': 'sad', 'score': 0.9}, ...]
                    if isinstance(text_scores_list, list) and len(text_scores_list) > 0:
                        text_emotion_full_scores = {item['label']: item['score'] for item in text_scores_list}

                        best_text_emotion = max(text_scores_list, key=lambda x: x['score'])
                        dominant_text_label_raw = best_text_emotion['label']
                        dominant_text_confidence = best_text_emotion['score']

                        unified_text_emotion = TEXT_TO_UNIFIED.get(dominant_text_label_raw.lower(), "unknown")
                        text_emotion_data = (unified_text_emotion, dominant_text_confidence)
                    else:
                        logger.warning(f"Unexpected format for text_scores_list (expected list of dicts): {text_scores_list}")
                        text_emotion_data = ("unknown", 0.0)
                else:
                    logger.warning(f"Unexpected format for text_results_outer (expected list containing list of dicts): {text_results_outer}")
                    text_emotion_data = ("unknown", 0.0)

            except Exception as e_text:
                logger.error(f"Error during text emotion classification for text: '{transcribed_text}' - {e_text}", exc_info=True)
                text_emotion_data = ("error", 0.0)
        else:
            text_emotion_data = ("unknown", 0.0) # No text or only whitespace

        # 4. Audio Emotion Classification
        try:
            if audio_chunk_data is not None and len(audio_chunk_data) > 0:
                inputs = audio_feature_extractor(
                    audio_chunk_data,
                    sampling_rate=audio_sample_rate,
                    return_tensors="pt",
                    padding=True # Add padding for short chunks
                )

                # Move inputs to the same device as the model
                model_device = audio_emotion_classifier.device
                inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = audio_emotion_classifier(**inputs_on_device).logits

                probs = torch.softmax(logits, dim=-1).squeeze()

                model_config_labels = audio_emotion_classifier.config.id2label

                # Ensure probs is iterable and matches model_config_labels size
                if probs.ndim == 0: # Single score
                     probs = probs.unsqueeze(0)

                if len(probs) == len(model_config_labels):
                    audio_emotion_full_scores = {model_config_labels[i]: probs[i].item() for i in range(len(model_config_labels))}

                    if audio_emotion_full_scores:
                        # Determine dominant raw audio emotion
                        dominant_audio_label_raw = max(audio_emotion_full_scores, key=audio_emotion_full_scores.get)
                        audio_confidence_raw = audio_emotion_full_scores[dominant_audio_label_raw]

                        # Map to unified emotion - this part requires careful handling if multiple raw map to one unified
                        # For simplicity, find the unified emotion corresponding to the most dominant raw emotion
                        unified_audio_emotion_label = SER_TO_UNIFIED.get(dominant_audio_label_raw.lower(), "unknown")

                        # If you need to aggregate scores for unified emotions (e.g. multiple 'anger_x' map to 'anger')
                        # you would iterate through audio_emotion_full_scores and sum them up for unified labels.
                        # For now, using the confidence of the top raw mapped emotion.
                        audio_emotion_data = (unified_audio_emotion_label, audio_confidence_raw)
                    else:
                        audio_emotion_data = ("unknown", 0.0)
                else:
                    logger.warning(f"Mismatch between number of probabilities ({len(probs)}) and model labels ({len(model_config_labels)}). Skipping audio emotion.")
                    audio_emotion_data = ("unknown", 0.0)
            else:
                audio_emotion_data = ("unknown", 0.0) # No audio data to process

        except Exception as e_audio:
            logger.error(f"Error during audio emotion analysis: {e_audio}", exc_info=True)
            audio_emotion_data = ("error", 0.0)

    except Exception as e_main:
        logger.error(f"Error processing audio chunk from file: {e_main}", exc_info=True)
        transcribed_text = ""
        text_emotion_data = ("error", 0.0)
        audio_emotion_data = ("error", 0.0)

    finally:
        # 5. Cleanup temporary file
        if temp_audio_file_name and os.path.exists(temp_audio_file_name):
            try:
                os.remove(temp_audio_file_name)
                logger.debug(f"Successfully deleted temporary audio file {temp_audio_file_name}")
            except Exception as e_delete:
                logger.warning(f"Could not delete temporary audio file {temp_audio_file_name}: {e_delete}")

    return transcribed_text, text_emotion_data, audio_emotion_data, text_emotion_full_scores, audio_emotion_full_scores

def run_test():
    if not os.path.exists(VIDEO_FILE_PATH):
        logging.error(f"Error: Video file not found at {VIDEO_FILE_PATH}")
        logging.info("Please update VIDEO_FILE_PATH to a valid video file on your system.")
        return

    logging.info(f"Attempting to load audio from video: {VIDEO_FILE_PATH}")
    try:
        # Load audio from video file using librosa.
        # sr=None preserves the original sampling rate.
        # mono=True converts to mono.
        # dtype=np.float32 ensures the data is float32.
        audio_data, audio_sample_rate = librosa.load(VIDEO_FILE_PATH, sr=None, mono=True, dtype=np.float32)
        logging.info(f"Audio loaded from video. Shape: {audio_data.shape}, Sample Rate: {audio_sample_rate}Hz")
    except Exception as e:
        logging.error(f"Error loading audio from video file {VIDEO_FILE_PATH}: {e}", exc_info=True)
        return

    # Load models (this will download models on first run)
    whisper_model, text_emotion_classifier, audio_feature_extractor, audio_emotion_classifier = load_models()

    # Define a chunk size (e.g., 5 seconds of audio)
    chunk_duration_seconds = 5
    chunk_size_samples = int(chunk_duration_seconds * audio_sample_rate)

    # Take the first chunk for testing, or the whole audio if it's shorter
    test_chunk = audio_data[:min(len(audio_data), chunk_size_samples)]

    if len(test_chunk) == 0:
        logging.error("Test audio chunk is empty after loading. Please use a longer video file or check the video content.")
        return

    logging.info(f"\\nProcessing audio chunk of {len(test_chunk) / audio_sample_rate:.2f} seconds...")

    # Call the new function
    transcribed_text, text_emotion_data, audio_emotion_data = \
        process_audio_chunk_from_file(
            test_chunk,
            audio_sample_rate,
            whisper_model,
            text_emotion_classifier,
            audio_feature_extractor,
            audio_emotion_classifier
        )

    # Print results
    logging.info(f"\n--- Audio Emotion Analysis Results ---")
    logging.info(f"Transcribed Text: \"{transcribed_text}\"")
    logging.info(f"Text Emotion: {text_emotion_data[0]} (Confidence: {text_emotion_data[1]:.2f})")
    logging.info(f"Audio Emotion: {audio_emotion_data[0]} (Confidence: {audio_emotion_data[1]:.2f})")
    logging.info(f"Test complete. Check for valid transcription and emotion data above.")

if __name__ == "__main__":
    run_test()