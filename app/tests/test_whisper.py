# filepath: c:\Users\kingk\OneDrive\Documents\Projects\AffectLink\app\tests\test_whisper.py
import os
import sys
import logging
import time
import whisper
import torch # For checking CUDA availability for fp16

# --- Path Setup ---
# Add project root and app directory to sys.path for potential helper imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # Goes up two levels from app/tests to project_root
app_dir = os.path.abspath(os.path.join(current_dir, '..')) # Goes up one level from app/tests to app

if project_root not in sys.path:
    sys.path.append(project_root)
if app_dir not in sys.path:
    sys.path.append(app_dir)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Test Configuration ---
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base") # Default to "base", can be overridden
TEST_AUDIO_PATH = os.path.join(project_root, "data", "sample_audio.wav") # Standard test audio file

def load_whisper_model(model_size: str):
    """Loads the specified Whisper model."""
    logger.info(f"Attempting to load Whisper model: '{model_size}'...")
    start_time = time.time()
    try:
        model = whisper.load_model(model_size)
        end_time = time.time()
        logger.info(f"Whisper model '{model_size}' loaded successfully in {end_time - start_time:.2f} seconds.")
        logger.info(f"Model is using device: {model.device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model '{model_size}': {e}", exc_info=True)
        return None

def run_whisper_inference(model, audio_path: str):
    """Runs inference using the loaded Whisper model on the given audio file."""
    if not os.path.exists(audio_path):
        logger.error(f"Test audio file not found: {audio_path}")
        return False

    logger.info(f"Attempting to transcribe audio file: {audio_path}...")
    start_time = time.time()
    try:
        # Determine if fp16 can be used
        use_fp16 = torch.cuda.is_available()
        logger.info(f"Transcribing with fp16: {use_fp16}")

        result = model.transcribe(audio_path, language="en", fp16=use_fp16)
        transcribed_text = result["text"].strip()
        end_time = time.time()

        logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds.")
        if transcribed_text:
            logger.info(f"Transcription result: \"{transcribed_text}\"")
            return True
        else:
            logger.warning("Transcription resulted in empty text.")
            # For sample_audio.wav, we expect text. If empty, it might indicate an issue or very faint audio.
            # However, the primary goal is to check if inference runs without error.
            return True 
    except Exception as e:
        logger.error(f"Error during Whisper transcription: {e}", exc_info=True)
        return False

def main():
    """Main function to run the Whisper test."""
    logger.info("--- Starting Whisper Load and Inference Test ---")

    model = load_whisper_model(WHISPER_MODEL_SIZE)

    if model:
        logger.info("Whisper model loaded. Proceeding to inference test...")
        inference_successful = run_whisper_inference(model, TEST_AUDIO_PATH)
        if inference_successful:
            logger.info("✅ Whisper inference test completed successfully.")
        else:
            logger.error("❌ Whisper inference test failed.")
    else:
        logger.error("❌ Whisper model loading failed. Cannot proceed with inference test.")

    logger.info("--- Whisper Load and Inference Test Finished ---")

if __name__ == "__main__":
    main()
