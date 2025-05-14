#!/usr/bin/env python3
"""
run_app_fixed.py - Enhanced main script for AffectLink with improved frame sharing
between the emotion detection and dashboard processes.
"""

import multiprocessing
import time
import signal
import os
import sys
import subprocess
import threading
import cv2

# Add the current directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our camera utility
from camera_utils import find_available_camera

# Global variables
stop_event = None
dashboard_process = None
detector_process = None

def run_detector(emotion_queue, stop_event, shared_frame_queue=None):
    """Run the emotion detection process with queue for IPC"""
    print("Starting emotion detection process...")
    try:
        # Try to import our fixed version first
        try:
            import detect_emotion_fixed as detect_emotion
            print("Using detect_emotion_fixed module")
        except ImportError:
            # Fall back to original if fixed version isn't available
            import detect_emotion
            print("Using original detect_emotion module")
            
        # Get camera index from environment if available
        camera_index = int(os.environ.get('WEBCAM_INDEX', '0'))
        print(f"Using camera index: {camera_index}")
        
        # Modified to use a dictionary for the stop flag and shared data
        # This combines stop_event functionality with frame sharing
        shared_stop_dict = {'stop': False, 'shared_frame_data': shared_frame_queue}
        
        # Create a thread to monitor the real stop_event and update our dict
        def monitor_stop_event():
            while not stop_event.is_set():
                time.sleep(0.1)
            shared_stop_dict['stop'] = True
            print("Stop signal propagated to emotion detection")
        
        stop_monitor = threading.Thread(target=monitor_stop_event)
        stop_monitor.daemon = True
        stop_monitor.start()
        
        # Run the main function with the queue, our stop dict, and camera index
        detect_emotion.main(
            emotion_queue=emotion_queue,
            stop_event=shared_stop_dict,  # Pass the dict instead of the Event
            camera_index=camera_index
        )
    except Exception as e:
        print(f"Error in emotion detection process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Emotion detection process stopped")

def run_dashboard():
    """Run the Streamlit dashboard as a separate process using streamlit run."""
    dashboard_path = os.path.join(current_dir, "dashboard.py")
    try:
        # Use Python executable to launch Streamlit module for better compatibility
        python_exe = sys.executable
        # Windows flags to detach process
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        cmd = [python_exe, "-m", "streamlit", "run", dashboard_path]
        # Launch the Streamlit dashboard as a detached process
        process = subprocess.Popen(
            cmd,
            cwd=current_dir,
            creationflags=creationflags,
            close_fds=True
        )
        print(f"Streamlit dashboard started with PID {process.pid}")
        return process
    except Exception as e:
        print(f"Error starting Streamlit dashboard: {e}")
        return None

def test_webcam():
    """Test if webcam is available and functioning."""
    print("Testing webcam availability...")
    try:
        # Using our imported camera_utils module
        camera_idx, backend, cap = find_available_camera()
        
        if cap is not None:
            # Store the working camera info in environment variables
            os.environ['WEBCAM_INDEX'] = str(camera_idx)
            os.environ['WEBCAM_BACKEND'] = backend
            print(f"Successfully found camera {camera_idx} with {backend} backend")
            cap.release()
            return True
        
        print("Warning: Cannot access any webcam. Some features may not work properly.")
        return False
    except Exception as e:
        print(f"Warning: Webcam test failed with error: {e}")
        return False

def signal_handler(signum, frame):
    """Handle Ctrl+C to gracefully stop all processes"""
    global stop_event, dashboard_process, detector_process
    
    print("Received interrupt signal. Shutting down processes...")
    if stop_event:
        stop_event.set()
        
    # Give processes time to clean up
    time.sleep(1)
    
    # Explicitly terminate processes if they're still running
    if detector_process and detector_process.is_alive():
        try:
            detector_process.terminate()
            print("Terminated detector process")
        except Exception:
            pass
            
    if dashboard_process and dashboard_process.poll() is None:
        try:
            dashboard_process.terminate()
            print("Terminated dashboard process")
        except Exception:
            pass
            
    print("Shutdown complete")
    sys.exit(0)

def main():
    """Main function to run the AffectLink system"""
    global stop_event, dashboard_process, detector_process
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Test webcam availability before starting processes
    webcam_available = test_webcam()
    if not webcam_available:
        print("Warning: No webcam available. Continuing with audio-only mode.")
    
    # Create shared stop event for signaling processes to stop
    stop_event = multiprocessing.Event()
    
    # Create queues for inter-process communication
    emotion_queue = multiprocessing.Queue()
    shared_frame_queue = multiprocessing.Queue(maxsize=5)
    
    # Store the shared frame queue in an environment variable so the dashboard can access it
    os.environ['SHARED_FRAME_QUEUE'] = str(id(shared_frame_queue))
    
    # Create and start the emotion detection process
    detector_process = multiprocessing.Process(
        target=run_detector,
        args=(emotion_queue, stop_event, shared_frame_queue)
    )
    
    # Start the emotion detection process
    print("Starting emotion detection process...")
    detector_process.start()
    
    # Give the detector process a moment to initialize
    time.sleep(2)
    
    # Start the Streamlit dashboard process
    print("Starting dashboard process...")
    dashboard_process = run_dashboard()
    
    try:
        print("Starting AffectLink multimodal emotion analysis system...")
        
        # Wait for processes to finish or for a keyboard interrupt
        while not stop_event.is_set():
            time.sleep(0.5)
            
            # Check if processes are still alive
            if not detector_process.is_alive():
                print("Detector process has terminated.")
                stop_event.set()
                break
                
            if dashboard_process and dashboard_process.poll() is not None:
                print("Dashboard process has terminated.")
                stop_event.set()
                break
                
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down...")
        stop_event.set()
        
    finally:
        # Cleanup
        signal_handler(signal.SIGINT, None)
        
if __name__ == "__main__":
    main()
