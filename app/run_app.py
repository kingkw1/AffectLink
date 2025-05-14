#!/usr/bin/env python3
# filepath: c:\Users\kingk\OneDrive\Documents\Projects\AffectLink\app\run_app.py
"""
run_app.py - Main script to run the emotion detection and dashboard processes
using multiprocessing for inter-process communication.
"""

import multiprocessing
import time
import signal
import os
import sys
import subprocess
import cv2

# Add the current directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Variables for inter-process communication
stop_event = None

def run_detector(emotion_queue, stop_event):
    """Run the emotion detection process with queue for IPC"""
    import detect_emotion
    
    print("Starting emotion detection process...")
    try:
        # Run the main function from detect_emotion.py with the queue
        detect_emotion.main(emotion_queue=emotion_queue, stop_event=stop_event)
    except Exception as e:
        print(f"Error in emotion detection process: {e}")
    finally:
        print("Emotion detection process stopped")

def run_dashboard():
    """Run the Streamlit dashboard as a separate process using streamlit run."""
    dashboard_path = os.path.join(current_dir, "dashboard.py")
    try:
        # Launch the Streamlit dashboard using subprocess
        process = subprocess.Popen(
            ["streamlit", "run", dashboard_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return process
    except Exception as e:
        print(f"Error starting Streamlit dashboard: {e}")
        return None

def test_webcam():
    """Test if webcam is available and functioning."""
    print("Testing webcam availability...")
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Warning: Cannot access webcam. Some features may not work properly.")
            return False
        
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame from webcam. Some features may not work properly.")
            cap.release()
            return False
            
        print("Webcam test successful.")
        cap.release()
        return True
    except Exception as e:
        print(f"Warning: Webcam test failed with error: {e}")
        return False

def signal_handler(signum, frame):
    """Handle Ctrl+C to gracefully stop all processes"""
    print("Received interrupt signal. Shutting down processes...")
    if stop_event:
        stop_event.set()

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Test webcam availability before starting processes
    webcam_available = test_webcam()
    
    # Create shared stop event for signaling processes to stop
    stop_event = multiprocessing.Event()
    
    # Create a queue for inter-process communication
    emotion_queue = multiprocessing.Queue()
    
    # Create and start the emotion detection process
    detector_process = multiprocessing.Process(
        target=run_detector,
        args=(emotion_queue, stop_event)
    )

    # Start the Streamlit dashboard as a separate process
    dashboard_process = run_dashboard()

    try:
        print("Starting AffectLink multimodal emotion analysis system...")
        detector_process.start()

        # Wait for processes to finish
        while not stop_event.is_set():
            time.sleep(1)

            # Check if processes are still alive
            if not detector_process.is_alive() or (dashboard_process and dashboard_process.poll() is not None):
                print("One of the processes has terminated. Shutting down...")
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down...")
        stop_event.set()

    finally:
        # Give processes time to clean up
        time.sleep(1)

        # Wait for detector process to terminate
        if detector_process.is_alive():
            print("Waiting for detector process to terminate...")
            detector_process.join(timeout=5)
            if detector_process.is_alive():
                print("Detector process did not terminate gracefully. Terminating...")
                detector_process.terminate()

        # Terminate the Streamlit dashboard process
        if dashboard_process and dashboard_process.poll() is None:
            print("Terminating Streamlit dashboard process...")
            dashboard_process.terminate()

        print("AffectLink system shut down.")
