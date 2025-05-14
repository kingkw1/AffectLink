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

# Add the current directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import shared data structures from detect_emotion.py
# We'll use a queue-based approach for communication between processes

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
            pass

    # Get latest video frame
    frame = None
    try:
        # Check if we have a shared queue or a local queue
        if hasattr(video_frame_queue, 'get_nowait'):
            # Try to get a frame without blocking
            frame = video_frame_queue.get_nowait()
        elif hasattr(video_frame_queue, 'get'):
            # For multiprocessing.Queue
            try:
                frame = video_frame_queue.get(block=False)
            except Exception:
                pass
    except queue.Empty:
        # No new frame available
        pass
        
    if frame is not None:
        # Display the video frame
        video_container.image(frame, channels="RGB", use_container_width=True)
        # Store the frame for later use
        if 'last_frame' not in st.session_state:
            st.session_state.last_frame = frame

    # Update metrics
    facial_emotion, facial_confidence = latest_data["facial_emotion"]
    facial_emotion_container.metric(
        "Facial Emotion", 
        f"{facial_emotion.capitalize()}", 
        f"{facial_confidence:.2f}"
    )

    # Update transcribed text
    text_container.markdown(f"**Latest transcription:**  \n{latest_data['transcribed_text']}")

    # Update audio emotions
    text_emotion, text_confidence = latest_data["text_emotion"]
    text_emotion_container.metric(
        "Text Emotion", 
        f"{text_emotion.capitalize()}", 
        f"{text_confidence:.2f}"
    )

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

        try:
            # Try to get data from the multiprocessing queue with a timeout
            data = mp_queue.get(timeout=0.5)

            # Pass data to the UI update queue
            ui_update_queue.put(data)

        except (queue.Empty, multiprocessing.queues.Empty):
            # If no data is available, continue the loop
            time.sleep(0.1)
        except Exception as e:
            print(f"Error receiving emotion data: {e}")
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
    # For standalone testing without Manager, use local queues
    local_frame_queue = queue.Queue(maxsize=5)
    local_emotion_queue = queue.Queue(maxsize=10)
    main(local_emotion_queue, None, local_frame_queue)