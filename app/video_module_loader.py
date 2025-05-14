import os
import time
import threading
import cv2
import importlib.util
from pathlib import Path

# Check if our dedicated video processing module is available
def get_video_processing_function():
    """Try to import video processing from the dedicated module"""
    try:
        # First try video_processing_fixed.py
        module_path = Path(__file__).parent / "video_processing_fixed.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("video_processing_fixed", module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "video_processing_loop"):
                    print("Using video_processing_fixed.py implementation")
                    return module.video_processing_loop
        
        # Then try video_processing.py
        module_path = Path(__file__).parent / "video_processing.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("video_processing", module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "video_processing_loop"):
                    print("Using video_processing.py implementation")
                    return module.video_processing_loop
    
    except Exception as e:
        print(f"Error importing video processing module: {e}")
    
    # Return None if no external implementation found
    print("No external video processing module found")
    return None
