# app/tests/test_text_classifier.py

import os
import sys
import logging
from transformers import pipeline

# Configure basic logging for visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define a cache directory for Hugging Face models
# This will still be used by transformers internally, even if not explicitly passed here
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
transformers_cache_dir = os.path.join(project_root, 'models', 'transformers_cache')

# Set the Hugging Face cache directory as an environment variable
# This is a more robust way to control the cache location for transformers
os.environ['TRANSFORMERS_CACHE'] = transformers_cache_dir
os.environ['HF_HOME'] = transformers_cache_dir # Also for general Hugging Face files

def test_text_classifier(): # Renamed function as per your note
    """
    Loads the text emotion classifier locally and runs a sample inference.
    """
    logger.info("Starting local text classifier test...")
    logger.info(f"Transformers models will be cached in: {os.environ.get('TRANSFORMERS_CACHE')}")

    # Ensure the cache directory exists
    os.makedirs(transformers_cache_dir, exist_ok=True)

    text_classifier = None
    try:
        logger.info("Initializing text emotion classifier...")
        text_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None, # To get all emotion scores
            # REMOVED: cache_dir=transformers_cache_dir # This was causing the error
        )
        logger.info("Text emotion classifier initialized successfully.")

    except Exception as e:
        logger.error(f"Error loading text classifier: {e}")
        logger.error("Please ensure you have an internet connection for the first run "
                     "to download the model, or that your 'transformers' library "
                     "is correctly installed along with its dependencies (e.g., PyTorch/TensorFlow).")
        return

    # --- Run a sample inference ---
    test_texts = [
        "I am so happy today, this is fantastic news!",
        "I feel terrible, everything is going wrong.",
        "That's a surprising turn of events!",
        "I'm just sitting here, feeling pretty neutral.",
        "I'm absolutely furious about this situation!",
        "I'm scared of what might happen next."
    ]

    for i, text in enumerate(test_texts):
        logger.info(f"\n--- Testing text {i+1}: '{text}' ---")
        try:
            # The pipeline call returns a list of lists of dictionaries when top_k is not None
            # or when processing multiple texts, but for single text it's usually [[{label, score}]]
            text_emotion_scores_raw = text_classifier(text)
            logger.info(f"Raw text emotion scores: {text_emotion_scores_raw}")

            if text_emotion_scores_raw and isinstance(text_emotion_scores_raw, list) and len(text_emotion_scores_raw) > 0:
                actual_scores_to_process = text_emotion_scores_raw[0] # Get the inner list of scores

                # Sort for easy readability
                sorted_scores = sorted(actual_scores_to_process, key=lambda x: x['score'], reverse=True)

                logger.info("Sorted emotion scores:")
                for emotion_data in sorted_scores:
                    logger.info(f"  - {emotion_data['label']}: {emotion_data['score']:.4f}")
                logger.info(f"Dominant emotion: {sorted_scores[0]['label']} (Score: {sorted_scores[0]['score']:.4f})")
            else:
                logger.warning("Classifier returned no valid scores or unexpected format.")

        except Exception as e:
            logger.error(f"Error during inference for text '{text}': {e}")

    logger.info("\nLocal text classifier test complete.")

if __name__ == "__main__":
    test_text_classifier()