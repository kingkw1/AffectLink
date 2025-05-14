"""
Test script to verify frame sharing between processes works correctly.
This will open a camera and share frames with a Streamlit dashboard.
"""

import cv2
import os
import sys
import time
import threading
import multiprocessing
import subprocess
import signal
from camera_utils import find_available_camera

# Add the current directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def camera_process(frame_queue, stop_event):
    """Process that captures camera frames and shares them via the queue"""
    print("Testing webcam availability...")
    
    # Find an available camera
    camera_idx, backend, cap = find_available_camera()
    
    if cap is None:
        print("Error: Cannot access any webcam.")
        stop_event.set()
        return
        
    print(f"Successfully found camera {camera_idx} with {backend} backend")
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break
                
            # Convert to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Try to add to queue without blocking
            try:
                if not frame_queue.full():
                    frame_queue.put(frame_rgb, block=False)
            except Exception:
                pass  # Queue might be full or have other issues
                
            time.sleep(0.03)  # ~30 FPS
    except Exception as e:
        print(f"Camera process error: {e}")
    finally:
        if cap and cap.isOpened():
            cap.release()
        print("Camera process terminated")

def signal_handler(signum, frame):
    """Handle Ctrl+C to gracefully stop all processes"""
    print("Received interrupt signal. Shutting down...")
    if stop_event:
        stop_event.set()

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create shared stop event and frame queue
    stop_event = multiprocessing.Event()
    frame_queue = multiprocessing.Queue(maxsize=5)
    
    # Store the frame queue ID in environment
    os.environ['SHARED_FRAME_QUEUE'] = str(id(frame_queue))
    
    # Create and start the camera process
    cam_process = multiprocessing.Process(
        target=camera_process,
        args=(frame_queue, stop_event)
    )
    
    # Start Streamlit dashboard
    dashboard_path = os.path.join(current_dir, "dashboard.py")
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", dashboard_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        print("Starting camera process...")
        cam_process.start()
        
        # Wait for processes to finish
        while not stop_event.is_set():
            time.sleep(1)
            
            # Check if processes are still alive
            if not cam_process.is_alive() or streamlit_process.poll() is not None:
                print("One of the processes has terminated. Shutting down...")
                stop_event.set()
                break
                
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down...")
        stop_event.set()
        
    finally:
        # Give processes time to clean up
        time.sleep(1)
        
        # Wait for camera process to terminate
        if cam_process.is_alive():
            print("Waiting for camera process to terminate...")
            cam_process.join(timeout=5)
            if cam_process.is_alive():
                print("Camera process did not terminate gracefully. Terminating...")
                cam_process.terminate()
                
        # Terminate the Streamlit dashboard process
        if streamlit_process.poll() is None:
            print("Terminating Streamlit dashboard process...")
            streamlit_process.terminate()
            
        print("Test completed.")
