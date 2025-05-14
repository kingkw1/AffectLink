"""
Camera utilities for the AffectLink application.
This module provides functions to work with cameras for emotion detection.
"""

import cv2
import time
import os

def find_available_camera(preferred_index=0, use_directshow=True):
    """
    Find an available camera by trying multiple indices and backends.
    
    Args:
        preferred_index: The camera index to try first (default: 0)
        use_directshow: Whether to try DirectShow backend first (default: True)
    
    Returns:
        tuple: (camera_index, backend_type, cv2.VideoCapture object) or (None, None, None) if no camera is found
    """
    print(f"Searching for available cameras (preferred index: {preferred_index})")
    
    # First try the preferred camera index with DirectShow if enabled
    if use_directshow:
        try:
            print(f"Trying camera {preferred_index} with DirectShow...")
            cap = cv2.VideoCapture(preferred_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Successfully opened camera {preferred_index} with DirectShow")
                    return preferred_index, "directshow", cap
                cap.release()
        except Exception as e:
            print(f"Error with DirectShow camera {preferred_index}: {e}")
    
    # Try the preferred camera index with default backend
    try:
        print(f"Trying camera {preferred_index} with default backend...")
        cap = cv2.VideoCapture(preferred_index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Successfully opened camera {preferred_index} with default backend")
                return preferred_index, "default", cap
            cap.release()
    except Exception as e:
        print(f"Error with default backend camera {preferred_index}: {e}")
    
    # Try other camera indices with both backends
    for idx in range(4):  # Try indices 0-3
        if idx == preferred_index:
            continue  # Already tried this one
            
        # Try with DirectShow
        if use_directshow:
            try:
                print(f"Trying camera {idx} with DirectShow...")
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print(f"Successfully opened camera {idx} with DirectShow")
                        return idx, "directshow", cap
                    cap.release()
            except Exception as e:
                print(f"Error with DirectShow camera {idx}: {e}")
        
        # Try with default backend
        try:
            print(f"Trying camera {idx} with default backend...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Successfully opened camera {idx} with default backend")
                    return idx, "default", cap
                cap.release()
        except Exception as e:
            print(f"Error with default backend camera {idx}: {e}")
    
    # No camera found
    print("No available camera found")
    return None, None, None
