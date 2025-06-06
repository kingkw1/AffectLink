import mlflow
from deepface import DeepFace
import logging
import numpy as np

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

# Note: deepface requires one of the linux libraries installed with sudo apt-get

# Set up logging for clearer output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Explicit Pip Requirements for DeepFace Model Deployment ---
# Based on your 'pip show' output and common MLflow deployment needs.
# This list is minimal and precise for reproducibility.
DEEPFACE_PIP_REQUIREMENTS = [
    "deepface==0.0.93",
    "tf-keras==2.19.0",
    "opencv-python==4.11.0.86",
    "numpy",
    "pandas",
    "mlflow",
    "tensorflow==2.19.0",
]

# --- Main Registration Logic ---
def register_deepface_model():
    # Set the experiment name before starting the run
    mlflow.set_experiment("AffectLink_Model_Registration")

    # Start an MLflow run to log model artifacts
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        logger.info(f"Using explicit pip requirements for DeepFace: {DEEPFACE_PIP_REQUIREMENTS}")

        # Register DeepFace's Facial Emotion Model
        logger.info("Loading and registering DeepFace Facial Emotion Model...")
        try:
            # Trigger DeepFace to download and load its emotion model internally
            # It's important that this `analyze` call happens before `build_model`
            # to ensure the necessary weights are downloaded.
            dummy_img_array = np.zeros((48, 48, 3), dtype=np.uint8)
            _ = DeepFace.analyze(dummy_img_array, actions=['emotion'], enforce_detection=False)
            logger.info("DeepFace analyze called to ensure emotion model is downloaded/loaded.")

            # Correctly build the DeepFace Emotion model using its documented API.
            emotion_client = DeepFace.build_model(task="facial_attribute", model_name="Emotion")
            facial_emotion_model = emotion_client.model
            
            logger.info("Successfully retrieved DeepFace Emotion Model using DeepFace.build_model().")

            # Define explicit ModelSignature
            # Input is a 48x48 grayscale image (batch_size, 48, 48, 1)
            input_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, 48, 48, 1), "input_image")
            ])
            # Output is 7 emotion probabilities (batch_size, 7)
            output_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, 7), "emotion_scores")
            ])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            mlflow.keras.log_model(
                facial_emotion_model,
                artifact_path="deepface_facial_emotion_model",
                registered_model_name="DeepFaceEmotionModel",
                pip_requirements=DEEPFACE_PIP_REQUIREMENTS, # Use the explicit list
                signature=signature
            )
            logger.info("Registered DeepFace Facial Emotion Model.")
        except Exception as e:
            logger.error(f"Failed to register DeepFace Facial Emotion Model: {e}")
            # Re-raise the exception to make errors more visible during development
            raise e 

if __name__ == "__main__":
    logger.info("Starting MLflow model registration process...")
    register_deepface_model()
    logger.info("MLflow model registration process completed.")