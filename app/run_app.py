#!/usr/bin/env python3
"""
run_app.py - Main script for AffectLink with improved frame sharing
between the emotion detection and dashboard processes.
"""

import multiprocessing
import time
import signal
import os
import sys
import subprocess
import threading
import tempfile # Added import
import cv2
import sounddevice as sd # Added import

# Add the current directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import our camera utility
import main_processor

# Global variables
stop_event = None
dashboard_process = None
detector_process = None

# Moved functions from start_realtime.py
def clear_affectlink_json_files():
    """Deletes stale AffectLink JSON and image files from the temp directory."""
    files_to_delete = ["affectlink_emotion.json", "affectlink_frame.jpg"]
    temp_dir = tempfile.gettempdir()
    for filename in files_to_delete:
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 Pre-cleaned old file: {file_path}")
            # else:
                # print(f"No pre-existing file to clean: {file_path}") # Optional: log if not found
        except Exception as e:
            print(f"⚠️ Error pre-cleaning file {file_path}: {e}")


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

def check_webcam():
    """Check if webcam is available"""
    print("Checking webcam availability...")
    try:
        # find_available_camera is imported from camera_utils at the top of the file
        camera_idx, backend, cap = find_available_camera()
        
        if cap is not None:
            print(f"✅ Webcam found: Camera #{camera_idx} with {backend} backend.")
            cap.release()
            return True, camera_idx, backend
        else:
            print("❌ No webcam detected.")
            return False, None, None
    except Exception as e:
        print(f"❌ Error checking webcam: {e}")
        return False, None, None

def check_microphone():
    """Check if microphone is available"""
    print("Checking microphone availability...")
    try:
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        
        if default_input:
            print(f"✅ Microphone found: {default_input.get('name', 'Unknown')}")
            return True, default_input.get('name')
        else:
            print("❌ No microphone detected.")
            return False, None
    except Exception as e:
        print(f"❌ Error checking microphone: {e}")
        return False, None

def run_detector(emotion_queue, stop_event, shared_frame_queue=None):
    """Run the emotion detection process with queue for IPC"""
    try:
        # Import the emotion detection module
        print("Using main_processor module") # Updated print statement
            
        # Get camera index from environment if available
        camera_index = int(os.environ.get('WEBCAM_INDEX', '0'))
        print(f"Using camera index: {camera_index}")
        
        # Pass the frame queue directly in the shared_stop_dict
        shared_stop_dict = {
            'stop': False, 
            'shared_frame_data': shared_frame_queue
        }
        print(f"Frame queue is None? {shared_frame_queue is None}")
        
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
        main_processor.main(
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
    
    # Call clear_affectlink_json_files at the very beginning
    clear_affectlink_json_files()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Integrated device checks from start_realtime.py
    print("\n=== AffectLink Real-time Processing Mode ===")
    print("\n🔄 Starting AffectLink in real-time processing mode...")

    # Check for required devices
    has_webcam, webcam_idx, webcam_backend = check_webcam()
    has_microphone, mic_name = check_microphone()

    if not has_webcam and not has_microphone:
        print("\n❌ ERROR: Neither webcam nor microphone detected.")
        print("AffectLink requires at least one of these devices to function properly.")
        input("\nPress Enter to exit...")
        return # Exit the application

    # Set environment variables based on detected devices
    if has_webcam:
        os.environ["WEBCAM_INDEX"] = str(webcam_idx)
        if webcam_backend == "directshow":
            os.environ["WEBCAM_BACKEND"] = "directshow"
    
    # Create shared stop event and Manager for IPC queues
    stop_event = multiprocessing.Event()
    manager = multiprocessing.Manager()
    emotion_queue = manager.Queue()
    shared_frame_queue = manager.Queue(maxsize=5)
    
    # Launch emotion detector in its own process using Manager queues
    detector_process = multiprocessing.Process(
        target=run_detector,
        args=(emotion_queue, stop_event, shared_frame_queue)
    )
    
    # Start the emotion detection process
    print("Starting emotion detection process...")
    detector_process.start()
    
    # Give the detector process a moment to initialize
    time.sleep(2)
    
    # Create a simpler approach for dashboard integration
    print("Starting AffectLink dashboard automatically...")
    
    # Set an environment variable so the dashboard knows a detector is running
    os.environ["AFFECTLINK_DETECTOR_RUNNING"] = "1"
    
    try:
        # Launch the dashboard using Streamlit with comprehensive configuration
        # to prevent any auto-reloading behavior
        dashboard_cmd = [
            sys.executable, "-m", "streamlit", "run", 
            os.path.join(current_dir, "dashboard.py"),
            "--logger.level=error",
            "--server.runOnSave=false",           # Disable auto-reload on save
            "--server.fileWatcherType=none",      # Disable file watcher
            "--client.showErrorDetails=false",    # Simplify error display
            "--browser.gatherUsageStats=false",   # Disable analytics
            "--server.enableCORS=false",          # Reduce complexity
            "--server.enableXsrfProtection=false" # Simplify request handling
        ]
        
        print(f"Running dashboard with command: {' '.join(dashboard_cmd)}")
        dashboard_process = subprocess.Popen(
            dashboard_cmd,
            env=os.environ.copy()
        )
        print("Dashboard started with Streamlit")
        
        # Provide instructions for manual launch if needed
        print("If the dashboard doesn't appear, you can launch it manually with:")
        print(f"    cd {current_dir} && streamlit run dashboard.py --logger.level=error")
    except Exception as e:
        print(f"Warning: Could not start dashboard automatically: {e}")
        print("You can manually start the dashboard with:")
        print(f"    cd {current_dir} && streamlit run dashboard.py")
    
    try:
        print("Starting AffectLink multimodal emotion analysis system...")
        
        while not stop_event.is_set():
            time.sleep(0.5)
            # Check if the detector process is still alive
            if not detector_process.is_alive():
                print("Detector process has terminated.")
                stop_event.set()
                break
            # Dashboard is run in a child process; we do not terminate on its exit
                
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down...")
        stop_event.set()
        
    finally:
        # Cleanup
        signal_handler(signal.SIGINT, None)
        
if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
