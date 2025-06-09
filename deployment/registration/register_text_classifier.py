import logging
import pandas as pd
import numpy as np
import os
import torch
import shutil
import sys # You already have this for sys.path.append, just confirming it's there
import json 

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models import infer_signature # <--- ENSURE THIS IS AT THE TOP LEVEL!
from mlflow.models.signature import ModelSignature # This is also at top level
from mlflow.types.schema import Schema, TensorSpec, ColSpec

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model IDs ---
TEXT_CLASSIFIER_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

# --- Explicit Pip Requirements for Text Emotion Model Deployment ---
TEXT_EMOTION_MODEL_PIP_REQUIREMENTS = [
    "transformers",
    "torch",
    "mlflow",
    "accelerate",
]

class TextClassifierPyfuncModel(PythonModel):
    def load_context(self, context):
        """
        This method is called once when the model is loaded.
        It loads the tokenizer and model from the artifact path.
        """
        logger.info("TextClassifierPyfuncModel load_context called. Loading model components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        model_path = context.artifacts["text_model_artifact_path"]
        logger.info(f"Loading tokenizer and model from artifact path: {model_path}")

        try:
            # Explicitly set attributes to None initially to avoid confusion
            self.tokenizer = None
            self.model = None
            self.emotion_pipeline = None
            self.labels = None

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            
            # Initialize the pipeline directly after loading components
            self._initialize_pipeline() 
            
            self.labels = self.emotion_pipeline.model.config.id2label.values()
            logger.info(f"Text emotion pipeline loaded successfully from {model_path}")
            logger.info(f"Text Model Labels: {list(self.labels)}")
        except Exception as e:
            logger.error(f"Error loading model components in load_context: {e}", exc_info=True)
            raise

    def _initialize_pipeline(self):
        """Helper to initialize the Hugging Face pipeline."""
        if self.tokenizer is None or self.model is None:
            logger.error("Tokenizer or model not loaded before pipeline initialization attempt.")
            raise ValueError("Model components (tokenizer, model) must be loaded first.")
        
        # Ensure it's not already initialized or re-initialize if needed
        if self.emotion_pipeline is None: # Only initialize if not already set
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1, # Pass device as integer for pipeline
                    top_k=None # Return all labels
                )
                logger.info("Hugging Face pipeline successfully initialized within _initialize_pipeline.")
            except Exception as e:
                logger.error(f"Error initializing Hugging Face pipeline: {e}", exc_info=True)
                raise

    def predict(self, context, model_input, params=None):
        logger.info("TextClassifierPyfuncModel predict called.")

        if self.emotion_pipeline is None:
            logger.warning("emotion_pipeline not found in predict. Attempting re-initialization.")
            try:
                self._initialize_pipeline()
            except Exception as e:
                logger.error(f"Failed to re-initialize pipeline in predict: {e}", exc_info=True)
                raise RuntimeError("Model pipeline not initialized and failed to re-initialize.")

        texts = []

        # --- FINAL REVISION OF INPUT HANDLING LOGIC ---
        if isinstance(model_input, pd.DataFrame):
            logger.info("Input is a pandas DataFrame.")
            if "text" in model_input.columns:
                texts = model_input["text"].tolist()
            else:
                raise TypeError("Input DataFrame must contain a 'text' column.")
        elif isinstance(model_input, dict):
            logger.warning(f"Received dictionary input with keys: {model_input.keys()}. Attempting to parse.")
            
            # --- Primary check for 'text' key ---
            if 'text' in model_input:
                text_value = model_input['text']
                if isinstance(text_value, str):
                    logger.info("Processing direct 'text' key (single string) from dictionary input.")
                    texts = [text_value]
                elif isinstance(text_value, list) and all(isinstance(item, str) for item in text_value):
                    logger.info("Processing direct 'text' key (list of strings) from dictionary input.")
                    texts = text_value
                elif isinstance(text_value, np.ndarray): # <--- ADD THIS CHECK
                    logger.info("Processing 'text' key (numpy array) from dictionary input.")
                    # Convert numpy array to a list for the pipeline
                    texts = text_value.tolist() 
                else:
                    # If 'text' key exists but its value is not a string, list of strings, or numpy array
                    raise TypeError(
                        f"Value for 'text' key in dictionary input is not a string, list of strings, or numpy array. "
                        f"Got type: {type(text_value)}."
                    )
            # --- Fallback to MLflow specific dictionary formats if 'text' not directly usable ---
            # These are less likely to be hit now if MLflow consistently converts to {'text': np.ndarray}
            elif "dataframe_split" in model_input:
                logger.info("Processing 'dataframe_split' dictionary input.")
                if isinstance(model_input["dataframe_split"]["data"], list) and \
                   all(isinstance(row, list) for row in model_input["dataframe_split"]["data"]):
                    df_data = model_input["dataframe_split"]["data"]
                    df_columns = model_input["dataframe_split"]["columns"]
                    if "text" in df_columns:
                        text_col_idx = df_columns.index("text")
                        texts = [row[text_col_idx] for row in df_data]
                    else:
                        raise TypeError("Input dictionary (dataframe_split) must contain 'text' column data.")
                else:
                    raise TypeError("Invalid 'data' format in 'dataframe_split'. Expected list of lists.")
            elif "dataframe_records" in model_input:
                logger.info("Processing 'dataframe_records' dictionary input.")
                records = model_input["dataframe_records"]
                texts = [record["text"] for record in records if "text" in record]
                if len(texts) != len(records):
                     raise ValueError("Some records in 'dataframe_records' did not contain a 'text' key.")
            else:
                raise TypeError(
                    f"Unsupported dictionary input format. Received keys: {list(model_input.keys())}. "
                    f"Expected 'dataframe_split', 'dataframe_records', or a direct 'text' key (with string/list/ndarray value)."
                )
        # Case 3: Input is a direct list of strings
        elif isinstance(model_input, list) and all(isinstance(item, str) for item in model_input):
            logger.info("Processing direct list of strings input.")
            texts = model_input
        else:
            raise TypeError(
                f"Unsupported input format. Expected pandas DataFrame, or dictionary "
                f"matching MLflow's input spec, or a list of strings. "
                f"Got: {type(model_input)} -> {str(model_input)[:100]}..." 
            )
        # --- END FINAL REVISION OF INPUT HANDLING LOGIC ---

        if not texts:
            logger.warning("No texts found in input, returning empty results.")
            return []

        # Perform inference using the pipeline
        try:
            predictions = self.emotion_pipeline(texts)

            # Format the predictions into the desired output structure
            results = []
            for i, text_input in enumerate(texts):
                # Ensure predictions[i] is a list of dictionaries for top_k=None
                emotion_scores = predictions[i] if isinstance(predictions[i], list) else [predictions[i]]

                # Sort emotions by score in descending order
                sorted_emotions = sorted(emotion_scores, key=lambda x: x['score'], reverse=True)

                dominant_emotion = sorted_emotions[0]['label'] if sorted_emotions else 'neutral'
                dominant_emotion_score = sorted_emotions[0]['score'] if sorted_emotions else 0.0

                # Prepare full emotion scores as a list of dicts with 'emotion' and 'score'
                full_emotion_scores_formatted = []
                for item in sorted_emotions:
                    full_emotion_scores_formatted.append({
                        "emotion": item['label'],
                        "score": float(item['score'])
                    })

                results.append({
                    "text_input": text_input,
                    "dominant_emotion": dominant_emotion,
                    "dominant_emotion_score": dominant_emotion_score,
                    "full_emotion_scores": full_emotion_scores_formatted
                })
            return results
        except Exception as e:
            logger.error(f"Error during text classification prediction: {e}", exc_info=True)
            raise RuntimeError(f"Prediction failed: {e}")
        

# --- Main Registration Logic (unchanged from previous) ---
def register_text_classifier_model():
    mlflow.set_experiment("AffectLink_Model_Registration")

    # Define a temporary directory to save the HF model locally before logging
    temp_hf_model_dir = "text_model_hf_cache"
    os.makedirs(temp_hf_model_dir, exist_ok=True)

    try:
        # 1. Pre-download and save the Hugging Face model components locally
        logger.info(f"Downloading and saving Hugging Face model '{TEXT_CLASSIFIER_MODEL_ID}' to '{temp_hf_model_dir}'...")
        tokenizer_for_saving = AutoTokenizer.from_pretrained(TEXT_CLASSIFIER_MODEL_ID)
        model_for_saving = AutoModelForSequenceClassification.from_pretrained(TEXT_CLASSIFIER_MODEL_ID)
        tokenizer_for_saving.save_pretrained(temp_hf_model_dir)
        model_for_saving.save_pretrained(temp_hf_model_dir)
        logger.info("Hugging Face model components successfully saved locally.")

        # 2. Define the model signature
        input_schema = Schema([
            TensorSpec(np.dtype(str), (-1,), name="text") # Input is a list of strings
        ])
        
        output_schema = Schema([
            TensorSpec(np.dtype(str), (-1,), name="text_input"),
            TensorSpec(np.dtype(str), (-1,), name="dominant_emotion"),
            TensorSpec(np.dtype(np.float64), (-1,), name="dominant_emotion_score"),
            TensorSpec(np.dtype(str), (-1,), name="full_emotion_scores"), # Keep as str for list of dicts
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # --- THIS IS THE CRITICAL CHANGE ---
        # Create a pandas DataFrame directly for input_example
        input_example = pd.DataFrame({"text": ["This is a test sentence."]})
        logger.info(f"Using input example DataFrame for MLflow logging:\n{input_example.to_string()}")
        # --- END CRITICAL CHANGE ---

        logger.info(f"Using pip requirements for Text Classifier: {TEXT_EMOTION_MODEL_PIP_REQUIREMENTS}")

        with mlflow.start_run() as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            artifacts = {
                "text_model_artifact_path": temp_hf_model_dir
            }

            mlflow.pyfunc.log_model(
                python_model=TextClassifierPyfuncModel(),
                artifact_path="text_emotion_classifier_model",
                registered_model_name="TextEmotionClassifierModel_english-distilroberta-base",
                pip_requirements=TEXT_EMOTION_MODEL_PIP_REQUIREMENTS,
                signature=signature, # Keep the signature
                # input_example=input_example, # REMOVE THIS LINE
                artifacts=artifacts
            )

            logger.info(f"Registered Text Emotion Model as TextEmotionClassifierModel_english-distilroberta-base.")

    except Exception as e:
        logger.error(f"Failed to register Text Emotion Model: {e}", exc_info=True)
        raise e
    finally:
        if os.path.exists(temp_hf_model_dir):
            shutil.rmtree(temp_hf_model_dir)
            logger.info(f"Cleaned up temporary Hugging Face model directory: {temp_hf_model_dir}")


# This is the primary block that runs when the script is executed
if __name__ == "__main__":
    logger.info("Starting MLflow model registration process...")

    logger.info("\n--- Performing local test of TextClassifierPyfuncModel ---")
    local_test_hf_cache_dir = "local_test_text_model_hf_cache"
    local_test_mlflow_model_path = "local_test_mlflow_model"

    logger.info(f"Downloading/saving '{TEXT_CLASSIFIER_MODEL_ID}' to '{local_test_hf_cache_dir}' for local test...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_CLASSIFIER_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_CLASSIFIER_MODEL_ID)
        os.makedirs(local_test_hf_cache_dir, exist_ok=True)
        tokenizer.save_pretrained(local_test_hf_cache_dir)
        model.save_pretrained(local_test_hf_cache_dir)
        logger.info("Hugging Face model components saved for local test.")
    except Exception as e:
        logger.error(f"Failed to download or save Hugging Face model for local test: {e}", exc_info=True)
        exit(1)

    logger.info(f"Saving TextClassifierPyfuncModel locally to '{local_test_mlflow_model_path}' for local test...")
    try:
        input_example_local_test = pd.DataFrame({"text": ["Test sentence for local save."]})
        dummy_output_local_test = [{
            "text_input": "Test sentence for local save.",
            "dominant_emotion": "neutral",
            "dominant_emotion_score": 0.5,
            "full_emotion_scores": [
                {"emotion": "neutral", "score": 0.5}
            ]
        }]
        local_test_signature = infer_signature(input_example_local_test, dummy_output_local_test)

        mlflow.pyfunc.save_model(
            path=local_test_mlflow_model_path,
            python_model=TextClassifierPyfuncModel(),
            artifacts={"text_model_artifact_path": local_test_hf_cache_dir},
            conda_env={
                "channels": ["defaults", "conda-forge"],
                "dependencies": [
                    f"python={sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "pip",
                    {
                        "pip": TEXT_EMOTION_MODEL_PIP_REQUIREMENTS
                    },
                ],
                "name": "mlflow-env-local-test",
            },
            signature=local_test_signature,
        )
        logger.info("TextClassifierPyfuncModel saved locally for testing.")

        logger.info(f"Loading TextClassifierPyfuncModel from '{local_test_mlflow_model_path}' for local test...")
        loaded_model = mlflow.pyfunc.load_model(local_test_mlflow_model_path)
        logger.info("pyfunc model loaded locally.")

        # --- MODIFIED LOCAL TEST INPUTS ---
        # Convert all test inputs to Pandas DataFrames before passing to loaded_model.predict
        # This simulates how MLflow's serving endpoint would typically normalize inputs.
        test_inputs_dfs = {
            "Pandas DataFrame": pd.DataFrame({
                "text": [
                    "I am incredibly happy today, this is the best news ever!",
                    "I feel so sad and lonely right now."
                ]
            }),
            "dataframe_records": pd.DataFrame([
                {"text": "What a wonderful day!"},
                {"text": "I am so disappointed with this outcome."}
            ]),
            "dataframe_split": pd.DataFrame(
                data=[
                    ["This is a neutral observation."],
                    ["I am very excited about the future!"]
                ],
                columns=["text"]
            ),
            "list of strings": pd.DataFrame({"text": [
                "I feel nothing.",
                "This makes me angry."
            ]}),
            "simple dict with text key": pd.DataFrame({"text": [
                "This is a simple text input for testing."
            ]}),
             "simple dict with text key (list)": pd.DataFrame({"text": [
                "This is a simple text input.", "Another simple input."
            ]})
        }
        # --- END MODIFIED LOCAL TEST INPUTS ---

        for input_type, test_input_data_df in test_inputs_dfs.items():
            logger.info(f"\nTesting with '{input_type}' input (converted to DataFrame for predict):")
            logger.info("TextClassifierPyfuncModel predict called.")
            predictions = loaded_model.predict(test_input_data_df) # Pass the DataFrame here
            logger.info(f"Predictions ('{input_type}' input):")
            logger.info(json.dumps(predictions, indent=2))

    except Exception as e:
        logger.error(f"Error during local test: {e}", exc_info=True)
        try:
            if os.path.exists(local_test_hf_cache_dir):
                shutil.rmtree(local_test_hf_cache_dir)
            if os.path.exists(local_test_mlflow_model_path):
                shutil.rmtree(local_test_mlflow_model_path)
            logger.info(f"Cleaned up local test temporary directories.")
        except Exception as cleanup_e:
            logger.warning(f"Failed to clean up temporary directories after error: {cleanup_e}")
        exit(1)

    try:
        if os.path.exists(local_test_hf_cache_dir):
            shutil.rmtree(local_test_hf_cache_dir)
        if os.path.exists(local_test_mlflow_model_path):
            shutil.rmtree(local_test_mlflow_model_path)
        logger.info(f"Cleaned up local test temporary directories: {local_test_hf_cache_dir}, {local_test_mlflow_model_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up local test temporary directory {local_test_hf_cache_dir} or {local_test_mlflow_model_path}: {e}")

    logger.info("\nMLflow model registration process completed (or skipped for local test).")
    
    register_text_classifier_model()