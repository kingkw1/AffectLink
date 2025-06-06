import mlflow
import whisper
import logging
# We don't need tensorflow, sys, or subprocess imports for this streamlined script.

# Set up logging for clearer output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Configuration ---
WHISPER_MODEL_SIZE = "base"

# --- Explicit Pip Requirements for Whisper Model Deployment ---
# These are the core dependencies for the 'whisper' library and PyTorch.
# Use 'pip show whisper torch' to get the exact versions if you want to pin them.
# For example: "whisper==1.1.5", "torch==2.3.0+cu121"
WHISPER_PIP_REQUIREMENTS = [
    "openai-whisper", # This is the package name for `import whisper`
    "torch",          # Whisper relies on PyTorch
    "torchaudio",     # PyTorch audio library, often needed for audio processing
    "ffmpeg-python",  # Often a dependency for audio handling with Whisper
    "mlflow",         # MLflow itself must be available in the deployment environment
    "soundfile",      # Sometimes needed for audio file operations
    "numpy",          # Core numeric library, always a good idea
    "pandas",         # For MLflow's internal data handling
    # Add CUDA/cuDNN related dependencies if you have a GPU-specific PyTorch build, e.g., "nvidia-cuda-runtime-cu12"
    # Ensure these versions are compatible with your deployment environment's base image.
]

# --- Main Registration Logic ---
def register_whisper_model(): # Renamed the function for clarity, as it now focuses on Whisper
    mlflow.set_experiment("AffectLink_Model_Registration")

    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        logger.info(f"Using explicit pip requirements for Whisper: {WHISPER_PIP_REQUIREMENTS}")

        # Register Whisper ASR Model
        logger.info(f"Loading and registering Whisper '{WHISPER_MODEL_SIZE}' ASR Model...")
        try:
            # Whisper loads a PyTorch model.
            whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
            logger.info(f"Successfully loaded Whisper '{WHISPER_MODEL_SIZE}' model.")

            # For `whisper.load_model`, mlflow.pyfunc.log_model is often the most flexible,
            # as it creates a Python Function model. This allows for custom predict logic.
            # However, since whisper.load_model returns a PyTorch model object,
            # mlflow.pytorch.log_model is also a valid and potentially more direct choice.
            # Let's stick with mlflow.pytorch.log_model as you had it, but ensure compatibility.

            # Important: For PyTorch models, defining a ModelSignature can be more complex
            # as it depends on the exact input/output of the model's `forward` method.
            # For simplicity, we'll omit `signature` and `input_example` initially for Whisper
            # if the `pytorch_model` flavor handles it implicitly, or if you plan to wrap it
            # in a custom `pyfunc` for more control over inference.
            # If deployment fails, we might revisit adding a signature.

            mlflow.pytorch.log_model(
                pytorch_model=whisper_model,
                artifact_path=f"whisper_asr_model_{WHISPER_MODEL_SIZE}",
                registered_model_name=f"WhisperASRModel_{WHISPER_MODEL_SIZE}",
                pip_requirements=WHISPER_PIP_REQUIREMENTS
                # Omit signature and input_example for now, as PyTorch models often require
                # more specific handling or a custom `pyfunc` wrapper for inference.
            )
            logger.info(f"Registered Whisper '{WHISPER_MODEL_SIZE}' ASR Model.")
        except Exception as e:
            logger.error(f"Failed to register Whisper ASR Model: {e}")
            raise e # Re-raise to see the full traceback if an error occurs

if __name__ == "__main__":
    logger.info("Starting MLflow model registration process...")
    register_whisper_model()
    logger.info("MLflow model registration process completed.")