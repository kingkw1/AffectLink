#!/usr/bin/env python3
# app/realtime_emotion_detection.py

import cv2
from deepface import DeepFace
import logging
import time

# Suppress DeepFace logging for cleaner console output
logging.getLogger().setLevel(logging.ERROR)

def main():
    """
    Capture webcam video, detect emotions in real-time, overlay results, and print to console.
    Timestamp each result and store in a list for later synchronization.
    """
    # Initialize video capture (0 = default webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return

    print("Press 'q' to exit.")
    
    # List to store timestamped emotion results
    video_emotion_log = []
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        try:
            # Analyze emotions; use faster OpenCV detector, no exception if no face
            results = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )

            # Handle single or multiple faces
            faces = results if isinstance(results, list) else [results]
            for face in faces:
                if 'dominant_emotion' in face:
                    emo = face['dominant_emotion']
                    region = face.get('region', {})
                    x, y, w, h = region.get('x',0), region.get('y',0), region.get('w',0), region.get('h',0)
                    # Draw rectangle and overlay text
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                    text_y = y-10 if y-10>10 else y+h+20
                    cv2.putText(frame, f"{emo}", (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    # Get confidence score for the dominant emotion
                    confidence = face.get('emotion', {}).get(emo, None)
                    # Record timestamped result
                    timestamp = time.time()  # Unix timestamp (float, seconds)
                    video_emotion_log.append({
                        'timestamp': timestamp,
                        'emotion': emo,
                        'confidence': confidence
                    })
                    print(f"[{timestamp:.3f}] Detected emotion: {emo} (confidence: {confidence})")
                else:
                    print("No face detected or emotion data unavailable.")

        except Exception as e:
            print(f"Analysis error: {e}")

        # Display the frame
        cv2.imshow('Real-time Emotion Detection', frame)
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    # Optionally, print or save the log for later use
    # print(video_emotion_log)

if __name__ == '__main__':
    main()