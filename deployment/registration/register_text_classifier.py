import mlflow
from transformers import pipeline # Only pipeline is strictly needed for this model
import logging
import torch # Import torch as the model is PyTorch-based
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
import numpy as np

# Set up logging for clearer output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model IDs ---
TEXT_CLASSIFIER_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# --- Explicit Pip Requirements for Text Emotion Model Deployment ---
# These are the minimal requirements for running the Hugging Face text-classification pipeline
# using the PyTorch backend.
TEXT_EMOTION_MODEL_PIP_REQUIREMENTS = [
    "transformers",
    "torch", # Model uses PyTorch backend
    # You might consider adding specific versions if you encounter issues:
    # f"transformers=={transformers.__version__}",
    # f"torch=={torch.__version__}",
    "mlflow", # Required for MLflow to run the model
    "accelerate", # Often a beneficial dependency for transformers for performance
]

# --- Main Registration Logic ---
def register_text_classifier_model(): # Renamed function for clarity
    # Set the experiment name before starting the run
    mlflow.set_experiment("AffectLink_Model_Registration")

    # Start an MLflow run to log model artifacts
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Register Text Emotion Classification Model (Hugging Face Pipeline)
        logger.info(f"Loading and registering Text Emotion Model ({TEXT_CLASSIFIER_MODEL_ID})...")
        try:
            # 1. Load the text emotion pipeline directly from Hugging Face
            # Important: The task is "text-classification" for this model, not "sentiment-analysis"
            # Setting top_k=None ensures we get all emotion scores back.
            text_emotion_pipeline = pipeline(
                "text-classification",
                model=TEXT_CLASSIFIER_MODEL_ID,
                top_k=None # Ensure all emotion scores are returned
            )
            logger.info("Text emotion pipeline loaded successfully locally.")

            # 2. Define the model signature
            # Input schema: a list of strings (text)
            input_schema = Schema([
                TensorSpec(np.dtype(str), (-1,), name="text") # -1 for variable batch size
            ])
            # Output schema: This is tricky for a pipeline, as it returns complex JSON.
            # We'll define it as a generic JSON for simplicity, or omit if mlflow.transformers handles it.
            # mlflow.transformers usually handles this well. For robustness, we can define a generic output.
            output_schema = Schema([
                TensorSpec(np.dtype(str), (-1,), name="predictions") # Output will be stringified JSON
            ])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            # 3. Log the model using mlflow.transformers.log_model
            # This function is specifically designed for Hugging Face models and pipelines.
            # It automatically saves the tokenizer, model, and configuration.
            mlflow.transformers.log_model(
                transformers_model=text_emotion_pipeline,
                artifact_path="text_emotion_classifier_model",
                registered_model_name="TextEmotionClassifierModel_english-distilroberta-base",
                pip_requirements=TEXT_EMOTION_MODEL_PIP_REQUIREMENTS,
                # Setting `inference_config` is important for deployments that use the pipeline's predict method.
                # It should define the task.
                inference_config={"task": "text-classification", "model": TEXT_CLASSIFIER_MODEL_ID, "top_k": None},
                # signature=signature # mlflow.transformers usually infers a good signature
            )
            logger.info(f"Registered Text Emotion Model as TextEmotionClassifierModel")
        except Exception as e:
            logger.error(f"Failed to register Text Emotion Model: {e}")
            logger.error(f"Error details: {e}", exc_info=True) # Log full traceback

if __name__ == "__main__":
    logger.info("Starting MLflow model registration process...")
    register_text_classifier_model()
    logger.info("MLflow model registration process completed.")