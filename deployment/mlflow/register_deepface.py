import mlflow
from transformers import pipeline
from deepface import DeepFace
import sys
import subprocess
import logging
import numpy as np 

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

# Set up logging for clearer output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
def register_deepface_model():
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

        # Register DeepFace's Facial Emotion Model
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

if __name__ == "__main__":
    logger.info("Starting MLflow model registration process...")
    register_deepface_model()
    logger.info("MLflow model registration process completed.")