
import cv2
import os
import sys
import logging
import pprint

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
app_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add necessary directories to sys.path
for path in [current_dir, project_root, app_dir]:
    if path not in sys.path:
        sys.path.append(path)

# Configure DeepFace model cache
deepface_cache_dir = os.path.join(project_root, "models", "deepface_cache")
os.environ['DEEPFACE_HOME'] = deepface_cache_dir
os.makedirs(deepface_cache_dir, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Reduce DeepFace's internal logging output
logging.getLogger('deepface').setLevel(logging.ERROR)

# Import DeepFace after setting environment variables
from deepface import DeepFace

# Set path to test image
TEST_IMAGE_PATH = os.path.join(project_root, "data", "sample_image.png")

def test_deepface():
    """Simple test to verify DeepFace is working correctly."""
    logger.info(f"Testing DeepFace with image: {TEST_IMAGE_PATH}")

    # Check if the test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        logger.error(f"Test image not found: {TEST_IMAGE_PATH}")
        return False

    try:
        # Load the image using OpenCV
        image = cv2.imread(TEST_IMAGE_PATH)
        if image is None:
            logger.error(f"Failed to load image: {TEST_IMAGE_PATH}")
            return False

        logger.info(f"Image loaded successfully: {image.shape}")

        # Run DeepFace analysis
        logger.info("Running DeepFace analysis...")
        analysis = DeepFace.analyze(
            image,
            actions=['emotion'],
            enforce_detection=False,
            silent=True
        )

        # Check if a face was detected and analyzed
        if analysis and isinstance(analysis, list) and len(analysis) > 0:
            logger.info(f"Face detected! Analysis successful.")
            
            # Extract emotion data from the first detected face
            face_analysis = analysis[0]
            emotions = face_analysis['emotion']
            
            # Print the emotion results
            logger.info("\n--- Emotion Analysis Results ---")
            logger.info(f"Dominant emotion: {max(emotions, key=emotions.get)}")
            logger.info("All emotions:")
            for emotion, score in emotions.items():
                logger.info(f"  {emotion}: {score:.2f}%")
                
            # Also print the complete analysis with pretty formatting
            logger.info("\nComplete analysis result:")
            pp = pprint.PrettyPrinter(indent=2)
            pp.pprint(face_analysis)
            
            return True
        else:
            logger.warning("No face detected in the image.")
            return False

    except Exception as e:
        logger.error(f"Error testing DeepFace: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting DeepFace test...")
    success = test_deepface()
    
    if success:
        logger.info("✅ DeepFace test completed successfully!")
    else:
        logger.error("❌ DeepFace test failed.")