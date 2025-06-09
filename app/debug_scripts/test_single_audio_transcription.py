import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import os
import tempfile
import time
import traceback

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.audio_processor import transcribe_audio_whisper
from app.main_processor import logger


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
        whisper_model = whisper.load_model(model_name)
        logger.info(f"Whisper model '{model_name}' loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        logger.error("Please ensure Whisper is installed correctly (e.g., pip install openai-whisper).")
        logger.error(traceback.format_exc())
        return None, None

    # 2. Record audio chunk
    fs = 16000  # Whisper prefers 16kHz
    logger.info(f"Recording {duration} seconds of audio at {fs}Hz. Please speak clearly.")
    try:
        # Ensure sounddevice (sd) and numpy (np) are imported
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', blocking=True)
        sd.wait() # Wait for recording to complete
        logger.info("Audio recording complete.")
    except Exception as e:
        logger.error(f"Audio recording failed: {e}")
        logger.error("Please ensure your microphone is connected and sounddevice is installed correctly.")
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
    try:
        logger.info(f"Deleting temporary audio file: {temp_wav_path}")
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
    except Exception as e:
        logger.error(f"Could not delete temporary file {temp_wav_path}: {e}")

    logger.info("=== Single Audio Transcription Test Finished ===")
    return temp_wav_path, transcription