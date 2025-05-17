# Suppress missing ScriptRunContext and set Streamlit log level before import
import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'
import warnings
warnings.filterwarnings("ignore", ".*missing ScriptRunContext.*")
import logging
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

# Then import Streamlit
import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
import os
import sys
import multiprocessing
from multiprocessing.queues import Empty as MPQueueEmpty

# Add the current directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import shared data structures from detect_emotion.py
# We'll use a queue-based approach for communication between processes

# Dev mode flag for standalone testing
DEV_MODE = True if __name__ == "__main__" else False

import queue  # Local placeholder
# Global queues, will be set in main()
video_frame_queue = None
emotion_data_queue = None

# Store latest emotion data
latest_data = {
    "facial_emotion": ("unknown", 0.0),
    "text_emotion": ("unknown", 0.0),
    "audio_emotion": ("unknown", 0.0),
    "transcribed_text": "",
    "cosine_similarity": 0.0,
    "consistency_level": "Unknown"
}

# Define emotion colors for visualization
EMOTION_COLORS = {
    'neutral': '#AAAAAA',
    'happy': '#66BB6A',
    'sad': '#42A5F5',
    'angry': '#EF5350',
    'unknown': '#E0E0E0'
}

# Flag to control when the dashboard should stop
should_stop = False

def get_consistency_level(cosine_sim):
    """Convert cosine similarity to consistency level label"""
    if cosine_sim >= 0.8:
        return "High Consistency ✅✅", "green"
    elif cosine_sim >= 0.6:
        return "Moderate Consistency ✅", "yellow"
    elif cosine_sim >= 0.3:
        return "Low Consistency ⚠️", "orange"
    else:
        return "Inconsistent ❌", "red"

def video_capture_thread():
    """
    Thread to receive video frames from the emotion detection process.
    We don't open the camera here - we just take frames from the shared queue.
    """
    print("Dashboard video capture thread started")
    
    # Keep checking the queue until should_stop is set
    while not should_stop:
        if not video_frame_queue.full():
            # No need to do anything - the emotion detection process
            # is adding frames to the queue directly
            pass
        
        # Slow down a bit to reduce CPU usage
        time.sleep(0.03)
        
    print("Dashboard video capture thread stopped")

# Add a thread-safe queue for passing data to the main thread
ui_update_queue = queue.Queue()

def update_dashboard():
    """Update dashboard with latest emotion data"""
    global latest_data

    # Process updates from the UI update queue
    while not ui_update_queue.empty():
        try:
            latest_data = ui_update_queue.get_nowait()
        except queue.Empty:
            pass    # Get latest video frame
    frame = None
    try:
        # Check if we have a shared queue or a local queue
        if video_frame_queue and hasattr(video_frame_queue, 'get_nowait'):
            # Try to get a frame without blocking
            try:
                frame = video_frame_queue.get_nowait()
            except (queue.Empty, MPQueueEmpty):
                # No new frame available
                pass
        elif video_frame_queue and hasattr(video_frame_queue, 'get'):
            # For multiprocessing.Queue
            try:
                frame = video_frame_queue.get(block=False)
            except (MPQueueEmpty, Exception):
                pass
                
        # If we couldn't get a frame from the queue, check for saved frame file
        if frame is None:
            # Check for frame saved to temporary file by detector
            import tempfile
            import os
            
            frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                try:
                    # Track file modification time to avoid reloading the same frame repeatedly
                    mod_time = os.path.getmtime(frame_path)
                    
                    # Only load if file has been modified since last check
                    if not hasattr(update_dashboard, 'last_frame_time') or mod_time > update_dashboard.last_frame_time:
                        frame = cv2.imread(frame_path)
                        update_dashboard.last_frame_time = mod_time
                except Exception as e:
                    print(f"Error reading frame file: {e}")
            
    except Exception as e:
        print(f"Error retrieving video frame: {e}")
        pass
        
    if frame is not None:
        # Display the video frame
        video_container.image(frame, channels="RGB", use_container_width=True)
        # Store the frame for later use
        if 'last_frame' not in st.session_state:
            st.session_state.last_frame = frame

    # Update metrics
    # Check if facial_emotion is a tuple (old format) or dict (new format)
    if isinstance(latest_data["facial_emotion"], dict):
        facial_emotion = latest_data["facial_emotion"]["emotion"]
        facial_confidence = latest_data["facial_emotion"]["confidence"]
    else:
        # Fallback for older format
        facial_emotion, facial_confidence = latest_data["facial_emotion"]
        
    facial_emotion_container.metric(
        "Facial Emotion", 
        f"{facial_emotion.capitalize()}", 
        f"{facial_confidence:.2f}"
    )

    # Update transcribed text
    text_container.markdown(f"**Latest transcription:**  \n{latest_data['transcribed_text']}")

    # Update audio emotions
    if isinstance(latest_data["text_emotion"], dict):
        text_emotion = latest_data["text_emotion"]["emotion"]
        text_confidence = latest_data["text_emotion"]["confidence"]
    else:
        # Fallback for older format
        text_emotion, text_confidence = latest_data["text_emotion"]
    
    text_emotion_container.metric(
        "Text Emotion", 
        f"{text_emotion.capitalize()}", 
        f"{text_confidence:.2f}"
    )

    if isinstance(latest_data["audio_emotion"], dict):
        audio_emotion = latest_data["audio_emotion"]["emotion"]
        audio_confidence = latest_data["audio_emotion"]["confidence"]
    else:
        # Fallback for older format
        audio_emotion, audio_confidence = latest_data["audio_emotion"]
    
    audio_emotion_container.metric(
        "Audio (SER) Emotion", 
        f"{audio_emotion.capitalize()}", 
        f"{audio_confidence:.2f}"
    )

    # Update consistency
    consistency_level, color = get_consistency_level(latest_data["cosine_similarity"])
    consistency_container.metric(
        "Emotion Consistency", 
        consistency_level,
        f"{latest_data['cosine_similarity']:.2f}"
    )

def receive_emotion_data_thread(mp_queue, stop_event):
    """Thread to receive emotion data from detect_emotion.py via multiprocessing queue"""
    global should_stop

    while not should_stop:
        if stop_event and stop_event.is_set():
            should_stop = True
            print("Stop event detected in emotion data receiver thread")
            break

        # First try to get data from the queue if available
        if mp_queue:
            try:
                # Try to get data from the multiprocessing queue with a timeout
                data = mp_queue.get(timeout=0.1)  # Use shorter timeout to check file more frequently
                # Pass data to the UI update queue
                ui_update_queue.put(data)
                continue  # Skip file check if we got data from queue
            except (queue.Empty, MPQueueEmpty):
                # If no data is available, we'll check the file next
                pass
            except Exception as e:
                print(f"Error receiving emotion data from queue: {e}")

        # Check for emotion data saved to file as backup
        try:
            import os
            import tempfile
            import json
            
            emotion_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
            if os.path.exists(emotion_path):
                # Check file modification time to avoid reloading same data
                mod_time = os.path.getmtime(emotion_path)
                
                # Only process if file has been modified since last check
                if not hasattr(receive_emotion_data_thread, 'last_emotion_time') or mod_time > receive_emotion_data_thread.last_emotion_time:
                    with open(emotion_path, 'r') as f:
                        try:
                            data = json.load(f)
                            # Pass data to the UI update queue
                            ui_update_queue.put(data)
                            receive_emotion_data_thread.last_emotion_time = mod_time
                        except json.JSONDecodeError:
                            # File might be partially written, skip for now
                            pass
        except Exception as e:
            print(f"Error checking emotion file: {e}")
            
        # Short sleep to avoid burning CPU
        time.sleep(0.1)

# Main Streamlit app
st.set_page_config(
    page_title="AffectLink Emotion Dashboard",
    page_icon="😊",
    layout="wide"
)

st.title("AffectLink Real-time Multimodal Emotion Analysis")

# Create placeholders for dynamic content
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Video Feed")
    video_container = st.empty()
    
    st.subheader("Transcribed Audio")
    text_container = st.empty()

with col2:
    st.subheader("Emotion Analysis")
    
    facial_emotion_container = st.empty()
    st.markdown("---")
    
    st.markdown("### Audio Analysis")
    text_emotion_container = st.empty()
    audio_emotion_container = st.empty()
    st.markdown("---")
    
    st.markdown("### Overall Consistency")
    consistency_container = st.empty()

# Define the main function for the dashboard application
def main(emotion_queue=None, stop_event=None, frame_queue=None):
    """Start the Streamlit dashboard using provided IPC queues"""
    global should_stop, video_frame_queue, emotion_data_queue
    
    # Assign provided queues to global variables
    video_frame_queue = frame_queue
    emotion_data_queue = emotion_queue
    # Start video capture thread
    video_thread = threading.Thread(target=video_capture_thread)
    video_thread.daemon = True
    video_thread.start()
    
    # Start emotion data receiving thread if a queue is provided
    if emotion_data_queue is not None:
        emotion_thread = threading.Thread(
            target=receive_emotion_data_thread,
            args=(emotion_data_queue, stop_event)
        )
        emotion_thread.daemon = True
        emotion_thread.start()
    
    # Update dashboard in a loop
    while not should_stop:
        update_dashboard()
        
        # Check if stop event is set
        if stop_event and stop_event.is_set():
            should_stop = True
            print("Stop event detected in dashboard main loop")
            break
            
        time.sleep(0.1)  # Update every 100ms
        
    print("Dashboard loop exited")

# If run directly (for testing)
if __name__ == "__main__":
    # Check for environment variable that indicates if detector is running
    import os
    detector_running = os.environ.get("AFFECTLINK_DETECTOR_RUNNING") == "1"
    
    print(f"Detector running: {detector_running}")
    
    if detector_running:
        # Check for temporary files from detector
        import tempfile
        frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")
        emotion_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
        
        # Create empty queues to pass to main
        frame_queue = queue.Queue(maxsize=5)
        emotion_queue = queue.Queue(maxsize=10)
        
        print(f"Running in detector-connected mode")
        print(f"Will look for frames at: {frame_path}")
        print(f"Will look for emotions at: {emotion_path}")
        
        # Run main with the queues
        main(emotion_queue, None, frame_queue)
    else:
        # Run in demo mode with generated data
        print("Starting dashboard in standalone demo mode")
        # For standalone testing without Manager, use local queues
        local_frame_queue = queue.Queue(maxsize=5)
        local_emotion_queue = queue.Queue(maxsize=10)
        
        # Create a demo thread to simulate frames and emotion data
        def demo_data_provider():
            """Generate demo data for testing the dashboard standalone"""
            import numpy as np
            import time
            
            while True:
                # Create a demo frame (black with timestamp)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                timestamp = time.strftime("%H:%M:%S")
                cv2.putText(frame, f"Demo Mode - {timestamp}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Add the frame to the queue
                try:
                    local_frame_queue.put(frame, block=False)
                except queue.Full:
                    pass
                    
                # Create demo emotion data
                import random
                emotions = ['neutral', 'happy', 'sad', 'angry']
                demo_data = {
                    "facial_emotion": {
                        "emotion": random.choice(emotions), 
                        "confidence": random.random()
                    },
                    "text_emotion": {
                        "emotion": random.choice(emotions), 
                        "confidence": random.random()
                    },
                    "audio_emotion": {
                        "emotion": random.choice(emotions), 
                        "confidence": random.random()
                    },
                    "transcribed_text": f"Demo transcription at {timestamp}",
                    "cosine_similarity": random.random(),
                    "consistency_level": "Demo Consistency"
                }
                
                try:
                    local_emotion_queue.put(demo_data, block=False)
                except queue.Full:
                    pass
                    
                time.sleep(1)
        
        # Start the demo thread
        demo_thread = threading.Thread(target=demo_data_provider)
        demo_thread.daemon = True
        demo_thread.start()
        
        # Run the dashboard with our local demo queues
        main(local_emotion_queue, None, local_frame_queue)
        
    # End of main standalone execution