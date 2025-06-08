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


class SERPyfuncModel(PythonModel):
    # ADDED: load_context method to initialize model and processor
    def load_context(self, context):
        logger.info("SERPyfuncModel load_context called. Loading model and processor...")
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # MLflow passes artifacts via context.artifacts
            model_local_path = context.artifacts["ser_model_artifact_path"]
            logger.info(f"Loading model from artifact path: {model_local_path}")

            self.processor = AutoFeatureExtractor.from_pretrained(model_local_path)
            self.model = AutoModelForAudioClassification.from_pretrained(model_local_path)
            self.model.to(self.device)

            self.labels = [self.model.config.id2label[i] for i in sorted(self.model.config.id2label.keys())]
            logger.info(f"SER model and processor loaded successfully from {model_local_path}")
            logger.info(f"SER Model Labels: {self.labels}")
        except Exception as e:
            logger.error(f"Error loading SER model in load_context: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load SER model components: {e}")

    def predict(self, context, model_input):
        logger.info("SERPyfuncModel predict called.")

        if isinstance(model_input, dict):
            try:
                model_input = pd.DataFrame(model_input)
            except ValueError:
                if "dataframe_split" in model_input:
                    model_input = pd.DataFrame(model_input["dataframe_split"]["data"], 
                                                 columns=model_input["dataframe_split"]["columns"])
                elif "dataframe_records" in model_input:
                    model_input = pd.DataFrame(model_input["dataframe_records"])
                else:
                    raise ValueError(f"Unsupported input dictionary format: {model_input.keys()}")
        elif not isinstance(model_input, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame or a dictionary. Got: {type(model_input)}")

        if 'audio_base64' not in model_input.columns:
            raise ValueError("Input DataFrame must contain an 'audio_base64' column.")

        results_for_df = [] 

        for index, row in model_input.iterrows():
            audio_base64 = row['audio_base64']
            
            dominant_audio_emotion = "unknown"
            dominant_audio_emotion_score = 0.0
            full_audio_emotion_scores = []
            
            # New fields to capture raw logits and softmax scores
            raw_logits_output = None 
            softmax_scores_output = None

            try:
                audio_bytes = base64.b64decode(audio_base64)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
                    temp_audio_file.write(audio_bytes)
                    temp_audio_file_path = temp_audio_file.name

                    waveform, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
                    logger.debug(f"Audio loaded with librosa. Shape: {waveform.shape}, SR: {sr}, Min: {np.min(waveform):.4f}, Max: {np.max(waveform):.4f}, Mean Abs: {np.mean(np.abs(waveform)):.4f}")

                # Use self.processor and self.model which are initialized in load_context
                inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")

                for k in inputs:
                    inputs[k] = inputs[k].to(self.device)

                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                raw_logits_output = logits.cpu().numpy().tolist() # Convert to list for JSON serialization
                logger.info(f"Raw logits from model (shape: {logits.shape}): {raw_logits_output}")
                
                scores = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()
                
                softmax_scores_output = scores.tolist() # Convert to list for JSON serialization
                logger.info(f"Softmax scores (shape: {scores.shape}): {softmax_scores_output}")
                
                if np.sum(scores) < 1e-6: # If sum is too small, there's a problem
                    logger.error(f"WARNING: Sum of softmax scores is extremely low ({np.sum(scores):.8f}), indicating potential issue with model output.")


                all_emotions = []
                for i, score_val in enumerate(scores):
                    emotion = self.labels[i] if i < len(self.labels) else f"unknown-{i}"
                    all_emotions.append({"emotion": emotion, "score": float(score_val)})

                result_sorted = sorted(all_emotions, key=lambda x: x["score"], reverse=True)

                if result_sorted:
                    dominant_audio_emotion = result_sorted[0]["emotion"]
                    dominant_audio_emotion_score = result_sorted[0]["score"]
                    full_audio_emotion_scores = result_sorted 
                    logger.info(f"Analyzed dominant audio emotion: {dominant_audio_emotion} with score: {dominant_audio_emotion_score:.4f}")
                else:
                    logger.warning("SER analysis yielded no results for this sample.")

            except Exception as e:
                logger.error(f"Error during SER analysis for row {index}: {e}", exc_info=True)

            results_for_df.append({
                'dominant_audio_emotion': dominant_audio_emotion,
                'dominant_audio_emotion_score': dominant_audio_emotion_score,
                'full_audio_emotion_scores': full_audio_emotion_scores,
                'raw_logits': raw_logits_output,         
                'softmax_probabilities': softmax_scores_output 
            })

        return pd.DataFrame(results_for_df)


# --- Main Registration Logic ---
def register_tone_classifier_model():
    mlflow.set_experiment("AffectLink_Model_Registration")

    # ADDED: Create a temporary directory to save the Hugging Face model locally
    # MLflow will then copy these artifacts into the model's environment
    temp_hf_model_dir = "ser_model_hf_cache" # A more distinct name
    os.makedirs(temp_hf_model_dir, exist_ok=True)
    
    try:
        # ADDED: Download and save the Hugging Face model components locally
        logger.info(f"Downloading and saving Hugging Face model '{SER_MODEL_ID}' to '{temp_hf_model_dir}'...")
        processor_for_saving = AutoFeatureExtractor.from_pretrained(SER_MODEL_ID)
        model_for_saving = AutoModelForAudioClassification.from_pretrained(SER_MODEL_ID)
        processor_for_saving.save_pretrained(temp_hf_model_dir)
        model_for_saving.save_pretrained(temp_hf_model_dir)
        logger.info("Hugging Face model components successfully saved locally.")

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
            # MODIFIED: Output schema to reflect new fields for better validation
            output_schema = Schema([
                TensorSpec(np.dtype(str), (-1,), name="dominant_audio_emotion"),
                TensorSpec(np.dtype(np.float64), (-1,), name="dominant_audio_emotion_score"),
                TensorSpec(np.dtype(str), (-1,), name="full_audio_emotion_scores"), # Keep as str for now, as it's a list of dicts
                TensorSpec(np.dtype(np.float64), (-1, -1), name="raw_logits"), # Define shape based on actual output
                TensorSpec(np.dtype(np.float64), (-1, -1), name="softmax_probabilities") # Define shape based on actual output
            ])
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            input_example = pd.DataFrame({"audio_base64": [sample_audio_base64]})

            with mlflow.start_run() as run:
                logger.info(f"MLflow Run ID: {run.info.run_id}")
                logger.info(f"Using pip requirements for Tonal Classifier: {TONAL_CLASSIFIER_PIP_REQUIREMENTS}")

                # MODIFIED: Add artifacts to log_model
                artifacts = {
                    "ser_model_artifact_path": temp_hf_model_dir 
                }

                mlflow.pyfunc.log_model(
                    python_model=SERPyfuncModel(),
                    artifact_path="tonal_classifier_model",
                    registered_model_name="SERClassifierModel_hubert-large-superb-er",
                    pip_requirements=TONAL_CLASSIFIER_PIP_REQUIREMENTS,
                    signature=signature,
                    input_example=input_example,
                    artifacts=artifacts # ADDED: Pass the artifacts
                )
                logger.info("Registered Speech Emotion Recognition Pyfunc Model.")

        except Exception as e:
            logger.error(f"Failed to register Speech Emotion Recognition Model: {e}", exc_info=True)
            raise e
        finally:
            if os.path.exists(sample_audio_path):
                os.remove(sample_audio_path)
                logger.info(f"Cleaned up sample audio file: {sample_audio_path}")
    
    except Exception as e:
        logger.error(f"Error during Hugging Face model pre-download/save: {e}", exc_info=True)
        raise e # Re-raise to indicate a critical failure before MLflow logging

    finally:
        # ADDED: Clean up the temporary HF model cache directory
        if os.path.exists(temp_hf_model_dir):
            shutil.rmtree(temp_hf_model_dir)
            logger.info(f"Cleaned up temporary Hugging Face model directory: {temp_hf_model_dir}")


if __name__ == "__main__":
    logger.info("Starting MLflow model registration process for Speech Emotion Recognition...")
    register_tone_classifier_model()
    logger.info("MLflow model registration process completed for Speech Emotion Recognition.")