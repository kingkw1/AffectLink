import mlflow
from deepface import DeepFace
import logging
import numpy as np
import os
import shutil
import tempfile
import pandas as pd
import base64 # Import base64 for image encoding/decoding
import cv2 # Import cv2 for image processing

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from mlflow.pyfunc import PythonModel

# Set up logging for clearer output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Explicit Pip Requirements for DeepFace Model Deployment ---
# IMPORTANT: Ensure these versions match your pip freeze output exactly!
DEEPFACE_PIP_REQUIREMENTS = [
    "deepface==0.0.93",
    "tf-keras==2.19.0", 
    "opencv-python==4.11.0.86",
    "numpy==1.26.4",
    "pandas==2.2.0",
    "mlflow==2.18.0",
    "tensorflow==2.19.0", 
    "keras==3.8.0", 
    "pillow==10.3.0", 
    "gunicorn==22.0.0", 
    "Flask==3.1.0", 
    "flask-cors==6.0.0", 
    "protobuf==5.29.3", 
    "h5py==3.11.0", 
    "scipy==1.12.0", 
    "imutils", # DeepFace uses this internally
    "PyYAML", # Used by DeepFace for model configs
    "tqdm", # Used by DeepFace for progress bars during downloads
    "pydantic" # DeepFace dependencies often pull this in
    # Removed "frida" as it wasn't in your pip freeze.
    # Added "opt_einsum", "flatbuffers", "termcolor", "grpcio", "absl-py", "gast"
    # as these are core TensorFlow dependencies seen in your pip freeze.
    "opt_einsum==3.4.0",
    "flatbuffers==25.1.21",
    "termcolor==2.5.0",
    "grpcio==1.69.0",
    "absl-py==2.1.0",
    "gast==0.6.0",
    # Added core image/matrix processing libs from your pip freeze
    "scikit-image==0.21.0",
    "PyWavelets==1.8.0",
    "imageio==2.37.0",
    "tifffile==2025.1.10",
    "matplotlib==3.8.2", # Often used for visualization
    "kiwisolver==1.4.8", # Matplotlib dependency
    "fonttools==4.55.3", # Matplotlib dependency
    "cycler==0.12.1", # Matplotlib dependency
    # Also added deepface's internal model dependencies
    "mtcnn==1.0.0", 
    "retina-face==0.0.17",
    "huggingface-hub==0.32.4", # DeepFace can download models from Hugging Face
    "safetensors==0.5.3", # Related to Hugging Face models
    "tokenizers==0.21.1", # Related to Hugging Face models
    "transformers==4.52.4", # Related to Hugging Face models
]

# --- System Library Paths for DeepFace ---
# This path is confirmed by your `ldd` output.
LOCAL_LIB_DIR = "/lib/x86_64-linux-gnu/"

# List of specific shared libraries to bundle for DeepFace/OpenCV.
# This list is directly derived from your `ldd` output, excluding fundamental libs.
DEEPFACE_REQUIRED_LIBS = [
    "libGL.so.1",
    "libGLdispatch.so.0",
    "libGLX.so.0",
    "libX11.so.6",
    "libxcb.so.1",
    "libXau.so.6",
    "libXdmcp.so.6",
    "libbsd.so.0",
    "libmd.so.0",
    # Note: libXext.so.6 was in the previous example but not in your `ldd` output, so it's removed.
    # If during deployment, you still get errors about missing Xext, you might need to add it.
    # However, for now, we rely strictly on your ldd output.
]

# --- Custom Pyfunc Model for DeepFace ---
class DeepFacePyfuncModel(PythonModel):
    def load_context(self, context):
        logger.info("Loading DeepFacePyfuncModel context...")

        # 1. Set LD_LIBRARY_PATH for bundled system libraries
        bundled_lib_path = context.artifacts["deepface_libs"]
        
        current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        # Prepend our bundled path to ensure it's found first
        os.environ["LD_LIBRARY_PATH"] = f"{bundled_lib_path}:{current_ld_library_path}"
        
        logger.info(f"Set LD_LIBRARY_PATH to: {os.environ['LD_LIBRARY_PATH']}")

        # Verify if cv2 can be imported after setting LD_LIBRARY_PATH
        try:
            import cv2
            logger.info(f"OpenCV (cv2) imported successfully after setting LD_LIBRARY_PATH. Version: {cv2.__version__}")
        except ImportError as e:
            logger.error(f"Failed to import cv2 in deployed environment even after setting LD_LIBRARY_PATH: {e}")
            raise RuntimeError(f"OpenCV (cv2) not accessible in deployment environment: {e}")

        # 2. DeepFace automatically downloads models on first use if not found.
        # We'll trigger a dummy analysis to ensure models are downloaded to the
        # correct cache location within the deployed environment.
        try:
            # Pass a dummy image (a small black square) for initial analysis to download models.
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8) # Create a 100x100 black image
            
            logger.info("Triggering DeepFace.analyze to ensure models are downloaded...")
            # Use enforce_detection=False for dummy image as it might not contain a detectable face.
            _ = DeepFace.analyze(dummy_image, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            logger.info("DeepFace models (Emotion, RetinaFace, etc.) successfully downloaded/verified within load_context.")
        except Exception as e:
            logger.error(f"Failed to trigger DeepFace model download/verification in load_context: {e}")
            # Do not re-raise yet, let the predict method handle if it truly fails
            pass 

    def predict(self, context, model_input):        
        if isinstance(model_input, dict):
            # MLflow serving often wraps single-row inputs in a dict with 'inputs' key
            if "inputs" in model_input and isinstance(model_input["inputs"], list):
                model_input = pd.DataFrame(model_input["inputs"])
            elif "image_base64" in model_input:
                model_input = pd.DataFrame({"image_base64": model_input["image_base64"]})
            else:
                model_input = pd.DataFrame(model_input)
        elif not isinstance(model_input, pd.DataFrame):
             raise TypeError("Input must be a pandas DataFrame or a dictionary convertible to one.")

        if 'image_base64' not in model_input.columns:
            raise ValueError("Input DataFrame must contain an 'image_base64' column.")

        results = []
        for index, row in model_input.iterrows():
            image_base64 = row['image_base64']
            
            try:
                # Decode base64 string to bytes
                image_bytes = base64.b64decode(image_base64)
                
                # Use numpy and cv2 to load image from bytes
                image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                if image_array is None:
                    raise ValueError("Could not decode image from base64 string. Ensure it's a valid image format.")

                logger.info(f"Received and decoded image with shape: {image_array.shape}")

                # Perform analysis using DeepFace.analyze
                analysis_results = DeepFace.analyze(image_array, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
                
                if analysis_results and len(analysis_results) > 0:
                    dominant_emotion = analysis_results[0].get('dominant_emotion', 'No emotion found')
                    results.append(dominant_emotion)
                    logger.info(f"Analyzed dominant emotion: {dominant_emotion}")
                else:
                    results.append("No face detected or analysis failed.")
                    logger.warning("No face detected in the image or analysis yielded no results.")

            except Exception as e:
                logger.error(f"Error during DeepFace analysis for row {index}: {e}")
                results.append(f"Error: {e}") 

        return pd.DataFrame({'dominant_emotion': results})


# --- Main Registration Logic ---
def register_deepface_model():
    mlflow.set_experiment("AffectLink_Model_Registration")

    # Create a temporary directory to copy system libraries into
    temp_deepface_lib_dir = tempfile.mkdtemp()

    try:
        # Copy required system libraries to the temporary directory
        for lib_name in DEEPFACE_REQUIRED_LIBS:
            specific_lib_path = os.path.join(LOCAL_LIB_DIR, lib_name)
            
            if os.path.exists(specific_lib_path):
                shutil.copy(specific_lib_path, temp_deepface_lib_dir)
                logger.info(f"Copied shared library {lib_name} to: {temp_deepface_lib_dir}")
            else:
                logger.warning(f"Shared library {lib_name} not found at {specific_lib_path}. This might cause issues.")
        
        # Define artifacts to bundle
        artifacts = {
            "deepface_libs": temp_deepface_lib_dir,
        }
        
        with mlflow.start_run() as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            logger.info(f"Using explicit pip requirements for DeepFace: {DEEPFACE_PIP_REQUIREMENTS}")

            # Prepare data for input_example
            SAMPLE_IMAGE_LOCAL_PATH = "/home/jovyan/AffectLink/data/sample_image.png"
            if not os.path.exists(SAMPLE_IMAGE_LOCAL_PATH):
                logger.warning(f"Sample image not found at {SAMPLE_IMAGE_LOCAL_PATH}. Creating a dummy image for input_example.")
                dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.imwrite(SAMPLE_IMAGE_LOCAL_PATH, dummy_img)
                logger.info(f"Created a dummy sample image at {SAMPLE_IMAGE_LOCAL_PATH}")

            with open(SAMPLE_IMAGE_LOCAL_PATH, "rb") as f:
                sample_image_bytes = f.read()
            sample_image_base64 = base64.b64encode(sample_image_bytes).decode('utf-8')

            # Define explicit ModelSignature
            input_schema = Schema([TensorSpec(np.dtype(str), (-1,), name="image_base64")])
            output_schema = Schema([TensorSpec(np.dtype(str), (-1,), name="dominant_emotion")])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            
            # input_example uses the base64 string
            input_example = pd.DataFrame({"image_base64": [sample_image_base64]})

            # Log the custom Pyfunc model
            mlflow.pyfunc.log_model(
                python_model=DeepFacePyfuncModel(),
                artifact_path="deepface_emotion_model_pyfunc",
                registered_model_name="DeepFaceEmotionModel",
                pip_requirements=DEEPFACE_PIP_REQUIREMENTS,
                artifacts=artifacts, 
                signature=signature,
                input_example=input_example
            )
            logger.info("Registered DeepFace Emotion Model with bundled system libraries.")

    except Exception as e:
        logger.error(f"Failed to register DeepFace Emotion Model: {e}")
        raise e
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_deepface_lib_dir):
            shutil.rmtree(temp_deepface_lib_dir)
            logger.info(f"Cleaned up temporary directory: {temp_deepface_lib_dir}")

if __name__ == "__main__":
    logger.info("Starting MLflow model registration process for DeepFace...")
    register_deepface_model()
    logger.info("MLflow model registration process completed for DeepFace.")