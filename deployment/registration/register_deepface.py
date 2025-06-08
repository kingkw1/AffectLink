import mlflow
from deepface import DeepFace
import logging
import numpy as np
import os
import shutil
import tempfile
import pandas as pd
import base64
import cv2

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
    "tf-keras==2.19.0", # tf-keras is the official Keras 3 distribution for TensorFlow backend
    "opencv-python==4.11.0.86",
    "transformers==4.52.4", # DeepFace might use this for some backends, good to include
    "numpy==1.26.4",
    "pandas==2.2.0",
    "mlflow==2.18.0",
    # Explicitly add TensorFlow if tf-keras is a wrapper and not standalone
    "tensorflow==2.19.0",
    # Protobuf is a common conflict, let's include it
    "protobuf==5.29.3",
    # H5py and SciPy are also often needed for TensorFlow/Keras models
    "h5py==3.11.0",
    "scipy==1.12.0",
    # Add flask/gunicorn/flask-cors if the AI Studio deployment
    # environment uses them explicitly outside of MLflow's default server.
    # However, MLflow's pyfunc server usually handles this. Let's omit for now
    # to simplify.
    # "Flask==3.1.0",
    # "flask-cors==6.0.0",
    # "gunicorn==22.0.0",
]

# --- System Library Paths for DeepFace ---
LOCAL_LIB_DIR = "/lib/x86_64-linux-gnu/"

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
]

# --- Custom Pyfunc Model for DeepFace ---
class DeepFacePyfuncModel(PythonModel):
    def load_context(self, context):
        logger.info("Loading DeepFacePyfuncModel context...")

        # 1. Set LD_LIBRARY_PATH for bundled system libraries
        bundled_lib_path = context.artifacts["deepface_libs"]
        current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{bundled_lib_path}:{current_ld_library_path}"
        logger.info(f"Set LD_LIBRARY_PATH to: {os.environ['LD_LIBRARY_PATH']}")

        # Verify if cv2 can be imported after setting LD_LIBRARY_PATH
        try:
            import cv2
            logger.info(f"OpenCV (cv2) imported successfully after setting LD_LIBRARY_PATH. Version: {cv2.__version__}")
        except ImportError as e:
            logger.error(f"Failed to import cv2 in deployed environment even after setting LD_LIBRARY_PATH: {e}")
            raise RuntimeError(f"OpenCV (cv2) not accessible in deployment environment: {e}")

        # 2. Copy bundled DeepFace models to expected cache location
        deepface_weights_source = context.artifacts["deepface_models_cache"]
        deepface_weights_dest = os.path.expanduser("~/.deepface/weights/")

        # Ensure the destination directory exists
        os.makedirs(deepface_weights_dest, exist_ok=True)

        logger.info(f"Copying DeepFace models from {deepface_weights_source} to {deepface_weights_dest}")
        for item in os.listdir(deepface_weights_source):
            s = os.path.join(deepface_weights_source, item)
            d = os.path.join(deepface_weights_dest, item)
            if os.path.isfile(s):
                shutil.copy2(s, d) # copy2 preserves metadata
                logger.info(f"Copied DeepFace model: {item}")
        logger.info("Finished copying DeepFace models.")

        # 3. Trigger DeepFace.analyze to ensure it initializes with pre-loaded models
        try:
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8) 
            logger.info("Triggering DeepFace.analyze to ensure initialization with pre-bundled models...")
            _ = DeepFace.analyze(dummy_image, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            logger.info("DeepFace initialization with pre-bundled models successful.")
        except Exception as e:
            logger.error(f"Failed DeepFace initialization in load_context: {e}")
            raise RuntimeError(f"DeepFace failed to initialize: {e}") 

    def predict(self, context, model_input):
        logger.info("DeepFacePyfuncModel predict called.")

        if isinstance(model_input, dict):
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
                image_bytes = base64.b64decode(image_base64)
                image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

                if image_array is None:
                    raise ValueError("Could not decode image from base64 string. Ensure it's a valid image format.")

                logger.info(f"Received and decoded image with shape: {image_array.shape}")

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

    temp_deepface_lib_dir = tempfile.mkdtemp()
    temp_deepface_models_dir = tempfile.mkdtemp() # New temporary dir for models

    try:
        # Copy required system libraries
        for lib_name in DEEPFACE_REQUIRED_LIBS:
            specific_lib_path = os.path.join(LOCAL_LIB_DIR, lib_name)
            if os.path.exists(specific_lib_path):
                shutil.copy(specific_lib_path, temp_deepface_lib_dir)
                logger.info(f"Copied shared library {lib_name} to: {temp_deepface_lib_dir}")
            else:
                logger.warning(f"Shared library {lib_name} not found at {specific_lib_path}. This might cause issues.")

        # Copy DeepFace model weights
        deepface_local_weights_path = os.path.expanduser("~/.deepface/weights/")
        if not os.path.exists(deepface_local_weights_path):
            logger.error(f"DeepFace local weights directory not found at {deepface_local_weights_path}. Please run DeepFace.analyze once to download models.")
            raise FileNotFoundError(f"DeepFace weights not found at {deepface_local_weights_path}")

        for item in os.listdir(deepface_local_weights_path):
            s = os.path.join(deepface_local_weights_path, item)
            d = os.path.join(temp_deepface_models_dir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
                logger.info(f"Copied DeepFace model weight: {item} to {temp_deepface_models_dir}")

        # Define artifacts to bundle
        artifacts = {
            "deepface_libs": temp_deepface_lib_dir,
            "deepface_models_cache": temp_deepface_models_dir # New artifact for models
        }

        with mlflow.start_run() as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            logger.info(f"Using explicit pip requirements for DeepFace: {DEEPFACE_PIP_REQUIREMENTS}")

            SAMPLE_IMAGE_LOCAL_PATH = "/home/jovyan/AffectLink/data/sample_image.png"
            if not os.path.exists(SAMPLE_IMAGE_LOCAL_PATH):
                logger.warning(f"Sample image not found at {SAMPLE_IMAGE_LOCAL_PATH}. Creating a dummy image for input_example.")
                dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
                cv2.imwrite(SAMPLE_IMAGE_LOCAL_PATH, dummy_img)
                logger.info(f"Created a dummy sample image at {SAMPLE_IMAGE_LOCAL_PATH}")

            with open(SAMPLE_IMAGE_LOCAL_PATH, "rb") as f:
                sample_image_bytes = f.read()
            sample_image_base64 = base64.b64encode(sample_image_bytes).decode('utf-8')

            input_schema = Schema([TensorSpec(np.dtype(str), (-1,), name="image_base64")])
            output_schema = Schema([TensorSpec(np.dtype(str), (-1,), name="dominant_emotion")])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)

            input_example = pd.DataFrame({"image_base64": [sample_image_base64]})

            mlflow.pyfunc.log_model(
                python_model=DeepFacePyfuncModel(),
                artifact_path="deepface_emotion_model",
                registered_model_name="DeepFaceEmotionModel", # Consistent name
                pip_requirements=DEEPFACE_PIP_REQUIREMENTS,
                artifacts=artifacts, 
                signature=signature,
                input_example=input_example
            )
            logger.info("Registered DeepFace Emotion Pyfunc Model with bundled system libraries and models.")

    except Exception as e:
        logger.error(f"Failed to register DeepFace Emotion Pyfunc Model: {e}")
        raise e
    finally:
        if os.path.exists(temp_deepface_lib_dir):
            shutil.rmtree(temp_deepface_lib_dir)
            logger.info(f"Cleaned up temporary lib directory: {temp_deepface_lib_dir}")
        if os.path.exists(temp_deepface_models_dir):
            shutil.rmtree(temp_deepface_models_dir)
            logger.info(f"Cleaned up temporary models directory: {temp_deepface_models_dir}")

if __name__ == "__main__":
    logger.info("Starting MLflow model registration process for DeepFace...")
    register_deepface_model()
    logger.info("MLflow model registration process completed for DeepFace.")