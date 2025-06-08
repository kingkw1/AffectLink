import os
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import logging
import soundfile as sf # Needed to save dummy audio

# Set up logging for clearer output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
SER_MODEL_ID = "superb/hubert-large-superb-er"
DUMMY_AUDIO_PATH = "/tmp/dummy_audio.wav"

def create_dummy_audio(file_path, duration_seconds=1, sr=16000):
    """Creates a dummy audio file for testing."""
    try:
        # Generate a simple sine wave as dummy audio
        t = np.linspace(0, duration_seconds, int(sr * duration_seconds), endpoint=False)
        frequency = 440  # A4 note
        amplitude = 0.5
        dummy_waveform = amplitude * np.sin(2 * np.pi * frequency * t)

        # Save to WAV file
        sf.write(file_path, dummy_waveform, sr)
        logger.info(f"Dummy audio created at: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create dummy audio: {e}")
        return False

def test_ser_model_loading_and_prediction():
    logger.info("Starting test for Speech Emotion Recognition (SER) model.")

    # 1. Create a dummy audio file
    if not create_dummy_audio(DUMMY_AUDIO_PATH):
        logger.error("Skipping SER test due to dummy audio creation failure.")
        return

    try:
        logger.info(f"Loading feature extractor: {SER_MODEL_ID}")
        ser_processor = AutoFeatureExtractor.from_pretrained(SER_MODEL_ID)
        logger.info("Feature extractor loaded successfully.")

        logger.info(f"Loading model: {SER_MODEL_ID}")
        # Ensure the model is loaded to CPU for basic testing unless GPU is guaranteed
        ser_model = AutoModelForAudioClassification.from_pretrained(SER_MODEL_ID)
        logger.info("Model loaded successfully.")

        # If a GPU is available, move model to GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ser_model.to(device)
        logger.info(f"Model moved to device: {device}")

        # 2. Load and process the dummy audio
        logger.info(f"Loading dummy audio from {DUMMY_AUDIO_PATH}")
        waveform, sr = librosa.load(DUMMY_AUDIO_PATH, sr=16000, mono=True)
        logger.info(f"Dummy audio loaded. Shape: {waveform.shape}, Sample Rate: {sr}")

        inputs = ser_processor(waveform, sampling_rate=16000, return_tensors="pt")
        logger.info("Audio processed by feature extractor.")

        # Move inputs to the same device as the model
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        logger.info("Inputs moved to model device.")

        # 3. Perform inference
        logger.info("Performing inference with the SER model...")
        with torch.no_grad():
            logits = ser_model(**inputs).logits
        scores = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
        logger.info(f"Inference complete. Raw logits shape: {logits.shape}, Scores shape: {scores.shape}")

        # 4. Get emotion labels and results (similar to your analyze_audio_emotion_full)
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
            logger.info(f"Top emotion: {top_emotion} with score: {top_score:.4f}")
            logger.info(f"Full results: {result_sorted}")
            logger.info("SER model test PASSED.")
        else:
            logger.warning("No results from SER model. Test FAILED.")
            
    except Exception as e:
        logger.error(f"An error occurred during SER model test: {e}", exc_info=True)
        logger.error("SER model test FAILED.")
    finally:
        # Clean up dummy audio file
        if os.path.exists(DUMMY_AUDIO_PATH):
            os.remove(DUMMY_AUDIO_PATH)
            logger.info(f"Cleaned up dummy audio file: {DUMMY_AUDIO_PATH}")

if __name__ == "__main__":
    test_ser_model_loading_and_prediction()