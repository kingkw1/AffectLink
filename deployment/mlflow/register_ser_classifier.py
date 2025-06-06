import mlflow
import torch
import librosa
import numpy as np
import pandas as pd
import base64
import soundfile as sf # Used for creating dummy audio
import logging
import os
import tempfile
import shutil

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from mlflow.pyfunc import PythonModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
SER_MODEL_ID = "superb/hubert-large-superb-er"

# --- Pip Requirements ---
# IMPORTANT: Use pinned versions from your pip freeze for robustness.
# If test_tonal_classifier.py runs, then use the versions of these specific libraries.
# If you don't have a full pip freeze for this environment yet,
# run 'pip freeze > production_requirements_tone.txt' in your working env.
TONAL_CLASSIFIER_PIP_REQUIREMENTS = [
    "mlflow",
    "pandas",
    "torch",
    "torchaudio", # Often a companion to torch, good to include
    "transformers==4.40.2",
    "librosa",
    "numpy",
    "soundfile", # For handling audio files
    "scipy", # librosa dependency
    "resampy", # librosa dependency
    "accelerate", # Common transformer dependency, can cause issues if missing
    "audioread", # librosa dependency
    "filelock", # huggingface_hub dependency
    "huggingface_hub", # transformers dependency
    "safetensors", # transformers dependency
    "tokenizers", # transformers dependency
    "tqdm", # transformers dependency
]

# --- Custom Pyfunc Model for Speech Emotion Recognition ---
class SERPyfuncModel(PythonModel):
    def load_context(self, context):
        logger.info("Loading SERPyfuncModel context...")
        # Ensure the model and processor are loaded when the model is initialized
        try:
            self.processor = AutoFeatureExtractor.from_pretrained(SER_MODEL_ID)
            self.model = AutoModelForAudioClassification.from_pretrained(SER_MODEL_ID)

            # Move model to GPU if available for inference
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info(f"SER model and processor loaded successfully and moved to {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load SER model or processor in load_context: {e}", exc_info=True)
            raise RuntimeError(f"SER model/processor initialization failed: {e}")
        

    # Inside SERPyfuncModel class
    def predict(self, context, model_input):
        logger.info("SERPyfuncModel predict called.")

        # Ensure model_input is a pandas DataFrame, handling various MLflow input formats
        if isinstance(model_input, dict):
            # Attempt to convert directly from dict to DataFrame
            # This handles cases where MLflow might pass the dict directly
            # with column names as keys, or a list of dicts.
            try:
                # If it's like {"audio_base64": ["base64_string"]}, pd.DataFrame.from_dict works
                model_input = pd.DataFrame(model_input)
            except ValueError:
                # If it's like {"dataframe_records": [{"audio_base64": "base64_string"}]},
                # or {"dataframe_split": {"columns": ["audio_base64"], "data": [["base64_string"]]}}
                # This explicitly handles the two standard MLflow DataFrame JSON formats.
                if "dataframe_split" in model_input:
                    model_input = pd.DataFrame(model_input["dataframe_split"]["data"], 
                                            columns=model_input["dataframe_split"]["columns"])
                elif "dataframe_records" in model_input:
                    model_input = pd.DataFrame(model_input["dataframe_records"])
                else:
                    # If it's a dict but none of the above, raise an error
                    raise ValueError(f"Unsupported input dictionary format: {model_input.keys()}")
        elif not isinstance(model_input, pd.DataFrame):
            # If it's not a dict and not a DataFrame, it's an unexpected type
            raise TypeError(f"Input must be a pandas DataFrame or a dictionary. Got: {type(model_input)}")

        # Now model_input is guaranteed to be a DataFrame (or an error is raised)
        # Continue with your existing logic.

        if 'audio_base64' not in model_input.columns:
            raise ValueError("Input DataFrame must contain an 'audio_base64' column.")

        results = []
        for index, row in model_input.iterrows():
            audio_base64 = row['audio_base64']

            try:
                # ... rest of your prediction logic (no changes here) ...
                # Decode base64 to bytes
                audio_bytes = base64.b64decode(audio_base64)

                # Save bytes to a temporary WAV file for librosa to load
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                    temp_audio_file.write(audio_bytes)
                    temp_audio_file_path = temp_audio_file.name
                    logger.debug(f"Temporary audio saved to: {temp_audio_file_path}")

                    # Load audio with librosa
                    waveform, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
                    logger.debug(f"Audio loaded with librosa. Shape: {waveform.shape}, SR: {sr}")

                # Process audio with feature extractor
                inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")

                # Move inputs to model device
                for k in inputs:
                    inputs[k] = inputs[k].to(self.device)

                # Perform inference
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                scores = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()

                # Get emotion labels
                model_labels_dict = self.model.config.id2label
                ser_actual_labels = [model_labels_dict[i] for i in sorted(model_labels_dict.keys())]

                all_emotions = []
                for i, score in enumerate(scores):
                    emotion = ser_actual_labels[i] if i < len(ser_actual_labels) else f"unknown-{i}"
                    all_emotions.append({"emotion": emotion, "score": float(score)})

                result_sorted = sorted(all_emotions, key=lambda x: x["score"], reverse=True)

                if result_sorted:
                    top_emotion = result_sorted[0]["emotion"]
                    results.append(top_emotion)
                    logger.info(f"Analyzed dominant audio emotion: {top_emotion}")
                else:
                    results.append("No emotion detected.")
                    logger.warning("SER analysis yielded no results.")

            except Exception as e:
                logger.error(f"Error during SER analysis for row {index}: {e}", exc_info=True)
                results.append(f"Error: {e}")

        return pd.DataFrame({'dominant_audio_emotion': results})


# --- Main Registration Logic ---
def register_tone_classifier_model():
    mlflow.set_experiment("AffectLink_Model_Registration")

    # Create a dummy audio file for input example
    sample_audio_path = "/tmp/sample_audio_for_mlflow.wav"
    duration_seconds = 1
    sr = 16000
    t = np.linspace(0, duration_seconds, int(sr * duration_seconds), endpoint=False)
    frequency = 440
    amplitude = 0.5
    sample_waveform = amplitude * np.sin(2 * np.pi * frequency * t)
    
    try:
        sf.write(sample_audio_path, sample_waveform, sr)
        logger.info(f"Created sample audio for MLflow input_example at: {sample_audio_path}")

        # Read dummy audio into base64
        with open(sample_audio_path, "rb") as f:
            sample_audio_bytes = f.read()
        sample_audio_base64 = base64.b64encode(sample_audio_bytes).decode('utf-8')

        input_schema = Schema([TensorSpec(np.dtype(str), (-1,), name="audio_base64")])
        output_schema = Schema([TensorSpec(np.dtype(str), (-1,), name="dominant_audio_emotion")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        input_example = pd.DataFrame({"audio_base64": [sample_audio_base64]})

        with mlflow.start_run() as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            logger.info(f"Using pip requirements for Tonal Classifier: {TONAL_CLASSIFIER_PIP_REQUIREMENTS}")

            mlflow.pyfunc.log_model(
                python_model=SERPyfuncModel(),
                artifact_path="tonal_classifier_model",
                registered_model_name="SpeechEmotionRecognitionModel",
                pip_requirements=TONAL_CLASSIFIER_PIP_REQUIREMENTS,
                signature=signature,
                input_example=input_example
            )
            logger.info("Registered Speech Emotion Recognition Pyfunc Model.")

    except Exception as e:
        logger.error(f"Failed to register Speech Emotion Recognition Model: {e}", exc_info=True)
        raise e
    finally:
        if os.path.exists(sample_audio_path):
            os.remove(sample_audio_path)
            logger.info(f"Cleaned up sample audio file: {sample_audio_path}")

if __name__ == "__main__":
    logger.info("Starting MLflow model registration process for Speech Emotion Recognition...")
    register_tone_classifier_model()
    logger.info("MLflow model registration process completed for Speech Emotion Recognition.")