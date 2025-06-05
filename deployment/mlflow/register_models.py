import mlflow
import whisper
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
from deepface import DeepFace
from deepface.modules import modeling # Keep this import!
import tensorflow as tf
import sys
import subprocess
import logging
import numpy as np # Import numpy for array creation
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout # Added Dropout
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

# Set up logging for clearer output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration for MLflow ---
# If HP AI Studio has a central MLflow server, ensure MLFLOW_TRACKING_URI is set
# in your environment before running this script.
# Example (run this in your terminal):
# export MLFLOW_TRACKING_URI="http://your-hp-ai-studio-mlflow-server:5000"
# If not set, MLflow will default to './mlruns' in the current working directory.

# --- Model IDs from detect_emotion.py or your setup ---
TEXT_CLASSIFIER_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SER_MODEL_ID = "superb/hubert-large-superb-er"
WHISPER_MODEL_SIZE = "base" # As used in whisper.load_model()

# --- Utility to get pip requirements ---
def get_pip_requirements():
    """Captures the current Python environment's installed packages for MLflow."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=True # Raise an error for non-zero exit codes
        )
        # Filter out editable packages if necessary
        return [line for line in result.stdout.splitlines() if not line.startswith('-e')]
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get pip requirements: {e}")
        return []

# --- Main Registration Logic ---
def register_affectlink_models():
    # Set the experiment name before starting the run
    mlflow.set_experiment("AffectLink_Model_Registration")

    # Start an MLflow run to log model artifacts
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Get common pip requirements for all models.
        # This can be refined to be more specific per model if needed.
        pip_requirements = get_pip_requirements()
        if not pip_requirements:
            logger.warning("Could not gather pip requirements. Model reproducibility might be affected.")


        # 1. Register DeepFace's Facial Emotion Model
        logger.info("Loading and registering DeepFace Facial Emotion Model...")
        try:
            # Trigger DeepFace to download and load its emotion model internally
            dummy_img_array = np.zeros((48, 48, 3), dtype=np.uint8)
            _ = DeepFace.analyze(dummy_img_array, actions=['emotion'], enforce_detection=False)
            logger.info("DeepFace analyze called to ensure emotion model is downloaded/loaded.")

            # Correctly build the DeepFace Emotion model using its documented API.
            emotion_client = DeepFace.build_model(task="facial_attribute", model_name="Emotion")
            facial_emotion_model = emotion_client.model
            
            logger.info("Successfully retrieved DeepFace Emotion Model using DeepFace.build_model().")

            # Define explicit ModelSignature
            input_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, 48, 48, 1), "input_image")
            ])
            output_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, 7), "emotion_scores")
            ])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            mlflow.keras.log_model(
                facial_emotion_model,
                artifact_path="deepface_facial_emotion_model",
                registered_model_name="DeepFaceEmotionModel",
                pip_requirements=pip_requirements,
                signature=signature # Rely solely on the signature
                # Removed input_example to avoid the serialization issue
            )
            logger.info("Registered DeepFace Facial Emotion Model.")
        except Exception as e:
            logger.error(f"Failed to register DeepFace Facial Emotion Model: {e}")


        # # 2. Register Whisper ASR Model
        # logger.info(f"Loading and registering Whisper '{WHISPER_MODEL_SIZE}' ASR Model...")
        # try:
        #     # Whisper loads a PyTorch model.
        #     whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

        #     # Log the PyTorch model. mlflow.pytorch is suitable here.
        #     # If you were loading a Hugging Face model/pipeline *directly* via AutoModelForSpeechSeq2Seq,
        #     # mlflow.transformers would be even more ideal, but for `whisper.load_model`, PyTorch flavor works.
        #     mlflow.pytorch.log_model(
        #         pytorch_model=whisper_model,
        #         artifact_path=f"whisper_asr_model_{WHISPER_MODEL_SIZE}",
        #         registered_model_name=f"WhisperASRModel_{WHISPER_MODEL_SIZE}",
        #         pip_requirements=pip_requirements
        #     )
        #     logger.info(f"Registered Whisper '{WHISPER_MODEL_SIZE}' ASR Model.")
        # except Exception as e:
        #     logger.error(f"Failed to register Whisper ASR Model: {e}")

        # # 3. Register Text Emotion Classification Model (Hugging Face Pipeline)
        # logger.info(f"Loading and registering Text Emotion Model ({TEXT_CLASSIFIER_MODEL_ID})...")
        # try:
        #     # Load the text emotion pipeline directly from Hugging Face
        #     text_emotion_pipeline = pipeline("sentiment-analysis", model=TEXT_CLASSIFIER_MODEL_ID)

        #     # mlflow.transformers is specifically designed for Hugging Face models and pipelines
        #     mlflow.transformers.log_model(
        #         transformers_model=text_emotion_pipeline,
        #         artifact_path="text_emotion_classifier_model",
        #         registered_model_name="TextEmotionClassifierModel",
        #         pip_requirements=pip_requirements
        #     )
        #     logger.info(f"Registered Text Emotion Model ({TEXT_CLASSIFIER_MODEL_ID}).")
        # except Exception as e:
        #     logger.error(f"Failed to register Text Emotion Model: {e}")

        # # 4. Register Audio Emotion Classification Model (Hugging Face AutoModel)
        # logger.info(f"Loading and registering Audio Emotion Model ({SER_MODEL_ID})...")
        # try:
        #     # Load the model and its associated feature extractor
        #     audio_emotion_model = AutoModelForAudioClassification.from_pretrained(SER_MODEL_ID)
        #     feature_extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL_ID)
            
        #     # Create a simple wrapper dictionary for the transformers model and its components
        #     # This allows mlflow.transformers to save the entire context (model + feature_extractor)
        #     transformers_model_dict = {
        #         "model": audio_emotion_model,
        #         "feature_extractor": feature_extractor
        #     }

        #     mlflow.transformers.log_model(
        #         transformers_model=transformers_model_dict,
        #         artifact_path="audio_emotion_classifier_model",
        #         registered_model_name="AudioEmotionClassifierModel",
        #         pip_requirements=pip_requirements
        #     )
        #     logger.info(f"Registered Audio Emotion Model ({SER_MODEL_ID}).")
        # except Exception as e:
        #     logger.error(f"Failed to register Audio Emotion Model: {e}")

        # logger.info("All specified models attempted for registration.")

if __name__ == "__main__":
    logger.info("Starting MLflow model registration process...")
    register_affectlink_models()
    logger.info("MLflow model registration process completed.")