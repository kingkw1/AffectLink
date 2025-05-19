"""
Enhanced video processing module for AffectLink.
This module provides a robust implementation of the video processing loop
with better camera handling and error recovery.
"""

import os
import time
import cv2
import numpy as np
from deepface import DeepFace

# Camera utilities import will happen within the function to avoid import errors

# Constants (will be overridden by the caller's constants)
VIDEO_WINDOW_DURATION = 5  # seconds
UNIFIED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry']

def process_frame_for_emotions(frame, face_cascade=None): # Added face_cascade for signature consistency
    """
    Processes a single video frame to detect facial emotions.

    Args:
        frame: The video frame (numpy array) to analyze.
        face_cascade: Placeholder for compatibility, not directly used by DeepFace backend.

    Returns:
        A tuple containing:
            - dominant_emotion_raw (str): The dominant emotion detected (e.g., 'happy', 'sad').
                                          Returns None if no face or emotion is detected.
            - confidence (float): The confidence score for the dominant emotion.
                                  Returns None if no face or emotion is detected.
            - processed_frame (numpy.ndarray): The frame with emotion annotations drawn on it.
                                               Returns the original frame if no face is detected.
            - facial_emotions_full_dict (dict): A dictionary containing all detected emotions
                                                and their scores (e.g., {'happy': 0.8, 'sad': 0.1}).
                                                Returns an empty dict if no face/emotion.
    """
    dominant_emotion_raw = None
    confidence = None
    facial_emotions_full_dict = {}
    processed_frame = frame.copy() # Start with a copy of the original frame

    try:
        # Process this frame with DeepFace
        # Using 'opencv' backend for face detection as it's generally robust
        # enforce_detection=False allows analysis even if a face isn't perfectly clear,
        # but we need to handle cases where 'region' or 'dominant_emotion' might be missing.
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            enforce_detection=False,  # Try to analyze even if detection is not strong
            detector_backend='opencv',
            silent=True # Suppress DeepFace's own console logs for cleaner output
        )
        
        # DeepFace returns a list of dictionaries, one for each detected face.
        # We'll process the first detected face if available.
        face_data = results[0] if isinstance(results, list) and len(results) > 0 else (results if isinstance(results, dict) else None)

        if face_data and 'dominant_emotion' in face_data and 'emotion' in face_data:
            dominant_emotion_raw = face_data['dominant_emotion']
            facial_emotions_full_dict = face_data['emotion'] # This is the full dictionary of scores
            
            # Get confidence for the dominant emotion
            if dominant_emotion_raw in facial_emotions_full_dict:
                confidence = facial_emotions_full_dict[dominant_emotion_raw]
            else: # Should not happen if dominant_emotion is from the dict, but as a safeguard
                confidence = 0.0 

            # Draw rectangle and overlay text if region data is available
            region = face_data.get('region')
            if region and all(k in region for k in ['x', 'y', 'w', 'h']):
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text_to_display = f"{dominant_emotion_raw} ({confidence:.2f})" if confidence is not None else dominant_emotion_raw
                text_y = y - 10 if y - 10 > 10 else y + h + 20
                cv2.putText(processed_frame, text_to_display, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # No face detected or emotion data unavailable in the first result
            # Return values will remain None/empty dict as initialized
            pass
            
    except Exception as e:
        # Log error, but don't crash. Return Nones and original frame.
        # Using print for now as logger might not be configured here when run standalone
        # In a real app, this should use a logger instance.
        print(f"Error in process_frame_for_emotions: {e}")
        # Ensure defaults are returned on error
        dominant_emotion_raw = None
        confidence = None
        facial_emotions_full_dict = {}
        # processed_frame is already a copy of the original frame

    return dominant_emotion_raw, confidence, processed_frame, facial_emotions_full_dict