# text_emotion_api.py
"""
API endpoint for text emotion classification.
Uses a Hugging Face transformer model to predict emotion from text input.
"""

import logging
from transformers import pipeline
import os
import sys

# This script is intended to be part of the AffectLink project.
# It assumes that the project root (AffectLink/) is in the Python path,
# allowing for absolute imports like 'from app.constants import ...'.
# For direct execution (e.g., for testing using `if __name__ == "__main__":`),
# the following lines adjust sys.path to include the project root.
# The project structure is assumed to be:
# AffectLink/
#   app/
#     constants.py
#   deployment/
#     swagger/
#       text_emotion_api.py (this file)

# The prompt mentioned "constants.py is located in the parent directory (../app/constants.py)".
# This path adjustment (../../) correctly locates 'app/constants.py' from 'deployment/swagger/'
# relative to the project root 'AffectLink/'.
_executed_as_script = (__name__ == "__main__") or (__package__ is None)
if _executed_as_script:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.abspath(os.path.join(_current_dir, '..', '..'))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

# Attempt to import constants
try:
    from app.constants import TEXT_CLASSIFIER_MODEL_ID, TEXT_TO_UNIFIED, UNIFIED_EMOTIONS
    _constants_loaded = True
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import constants from app.constants: {e}.")
    print(f"Current sys.path: {sys.path}")
    print("Ensure that 'AffectLink/app/constants.py' exists and the project root is in sys.path.")
    print("The API will not function correctly without these constants.")
    TEXT_CLASSIFIER_MODEL_ID = None
    TEXT_TO_UNIFIED = {}
    UNIFIED_EMOTIONS = []
    _constants_loaded = False
    # For a production API, you might want to raise the ImportError or exit.

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the text emotion classification pipeline once when the module is loaded
text_emotion_pipeline = None
if _constants_loaded and TEXT_CLASSIFIER_MODEL_ID:
    try:
        logger.info(f"Loading text emotion classification model: {TEXT_CLASSIFIER_MODEL_ID}")
        text_emotion_pipeline = pipeline("text-classification", model=TEXT_CLASSIFIER_MODEL_ID)
        logger.info("Text emotion classification model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load text emotion classification model '{TEXT_CLASSIFIER_MODEL_ID}': {e}", exc_info=True)
        # text_emotion_pipeline remains None
else:
    if not _constants_loaded:
        logger.critical("Constants not loaded. Text emotion pipeline cannot be initialized.")
    elif not TEXT_CLASSIFIER_MODEL_ID: # This check is only relevant if _constants_loaded is True
        logger.critical("TEXT_CLASSIFIER_MODEL_ID is not defined in constants or is None. Text emotion pipeline cannot be initialized.")


def predict_text_emotion(text_input: str) -> dict:
    """
    Predicts emotion from text input using a pre-loaded transformer model.

    Args:
        text_input: The string containing the text to analyze.

    Returns:
        A dictionary with "emotion" and "confidence" keys.
        Example: {"emotion": "happy", "confidence": 0.95}
        Returns {"emotion": "unknown", "confidence": 0.0} if input is invalid,
        prediction fails, or emotion cannot be mapped.
    """
    default_response = {"emotion": "unknown", "confidence": 0.0}

    if not _constants_loaded:
        logger.error("Constants (TEXT_TO_UNIFIED, UNIFIED_EMOTIONS) are not loaded. Cannot process emotion.")
        return default_response

    if text_emotion_pipeline is None:
        logger.error("Text emotion pipeline is not initialized. Cannot perform prediction.")
        return default_response

    if not text_input or text_input.isspace():
        logger.warning("Input text is empty or whitespace. Returning 'unknown' emotion.")
        return default_response

    try:
        # Perform emotion prediction
        raw_results = text_emotion_pipeline(text_input)
        logger.debug(f"Raw pipeline output for '{text_input}': {raw_results}")

        if not raw_results or not isinstance(raw_results, list) or not raw_results[0]:
            logger.warning(f"No valid emotion detected by the pipeline for input: '{text_input}'. Raw: {raw_results}")
            return default_response

        top_result = raw_results[0]
        raw_label = top_result.get('label')
        raw_score = top_result.get('score')

        if raw_label is None or raw_score is None:
            logger.warning(f"Pipeline returned malformed result: {top_result} for input: '{text_input}'")
            return default_response

        try:
            confidence = float(raw_score)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert score '{raw_score}' from model to float. Using 0.0.")
            confidence = 0.0

        # Map to unified emotion using TEXT_TO_UNIFIED.
        # Assuming TEXT_TO_UNIFIED keys are lowercase, and model labels might vary in case.
        unified_emotion_category = TEXT_TO_UNIFIED.get(raw_label.lower())

        if unified_emotion_category is None or unified_emotion_category not in UNIFIED_EMOTIONS:
            logger.info(
                f"Raw emotion '{raw_label}' (mapped to '{unified_emotion_category}') "
                f"is not in UNIFIED_EMOTIONS or maps to None. Reporting as 'unknown'."
            )
            return default_response

        return {"emotion": unified_emotion_category, "confidence": round(confidence, 4)}

    except Exception as e:
        logger.error(f"Error during text emotion prediction for input '{text_input}': {e}", exc_info=True)
        return default_response


if __name__ == "__main__":
    # The sys.path modification at the top of the file handles import paths for direct execution.
    print("--- Text Emotion API Test ---")

    if not _constants_loaded:
        print("\nCRITICAL: Constants (app.constants) failed to load.")
        print("The test examples below will likely fail or return 'unknown' due to missing configuration.")
    elif text_emotion_pipeline is None:
        print("\nCRITICAL: Text emotion pipeline failed to load.")
        print("This could be due to missing TEXT_CLASSIFIER_MODEL_ID or issues with the model itself.")
        print("Test examples will likely return 'unknown'.")
    else:
        print("\nConstants and pipeline seem to be loaded. Running examples...")
        # Ensure UNIFIED_EMOTIONS and TEXT_TO_UNIFIED are available for meaningful tests
        if not UNIFIED_EMOTIONS or not TEXT_TO_UNIFIED:
             print("Warning: UNIFIED_EMOTIONS or TEXT_TO_UNIFIED is empty. Mapping may not work as expected.")


    sample_texts = [
        "I am so happy today!",
        "This is really sad news.",
        "I'm feeling angry about this situation.",
        "This is a neutral statement.",
        "I feel ecstatic and overjoyed!",
        "What a terrifying experience, I was so scared.",
        "",
        "   ",
        "This book is okay, not great but not bad either.",
        "I am utterly disgusted by this behavior."
    ]

    for i, text in enumerate(sample_texts):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input: '{text}'")
        result = predict_text_emotion(text)
        print(f"Output: {result}")

    print("\n--- Test Finished ---")
    if not _constants_loaded or text_emotion_pipeline is None:
        print("Please check console logs for critical errors regarding constants or model loading.")

