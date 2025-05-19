import os
import importlib.util
from pathlib import Path
import logging

# Get a logger instance
logger = logging.getLogger(__name__)

def get_video_processing_function():
    """
    Dynamically imports and returns the 'process_frame_for_emotions' 
    function from video_processing.py.
    """
    try:
        # Construct the absolute path to video_processing.py
        # Assumes video_module_loader.py and video_processing.py are in the same directory
        module_path = Path(__file__).parent / "video_processing.py"
        
        if not module_path.exists():
            logger.error(f"video_processing.py not found at {module_path}")
            return None

        # Create a module spec
        spec = importlib.util.spec_from_file_location("video_processing_module", str(module_path))
        
        if spec and spec.loader:
            video_module = importlib.util.module_from_spec(spec)
            # Add to sys.modules to handle relative imports within video_processing if any
            # sys.modules["video_processing_module"] = video_module 
            spec.loader.exec_module(video_module)
            
            if hasattr(video_module, "process_frame_for_emotions"):
                logger.info("Successfully imported 'process_frame_for_emotions' from video_processing.py")
                return getattr(video_module, "process_frame_for_emotions")
            else:
                logger.error("'process_frame_for_emotions' function not found in video_processing.py")
                return None
        else:
            logger.error("Could not create module spec or loader for video_processing.py")
            return None
            
    except ImportError as e:
        logger.error(f"ImportError when trying to load video_processing.py: {e}")
        logger.error("Please ensure 'video_processing.py' exists and is correctly structured.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_video_processing_function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
