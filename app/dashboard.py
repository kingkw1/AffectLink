import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from collections import deque
import os
import sys

# Add the current directory to sys.path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import shared data structures from detect_emotion.py
# We'll use a queue-based approach for communication between processes
# In a real implementation, these would be imported or shared via IPC mechanisms

# Global variables for shared data
video_frame_queue = queue.Queue(maxsize=5)
emotion_data_queue = queue.Queue(maxsize=10)

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
    """Thread to capture video frames from webcam and put them in queue"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # If queue is full, remove oldest frame
        if video_frame_queue.full():
            try:
                video_frame_queue.get_nowait()
            except queue.Empty:
                pass
                
        video_frame_queue.put(frame_rgb)
        time.sleep(0.03)  # ~30 FPS
        
    cap.release()

# Mock function to simulate receiving emotion data from detect_emotion.py
# In a real implementation, this would receive data from detect_emotion.py
def receive_emotion_data_thread():
    """Thread to simulate receiving emotion data from detect_emotion.py"""
    emotions = ['neutral', 'happy', 'sad', 'angry']
    sample_texts = [
        "I'm feeling pretty good today.",
        "This is an interesting conversation.",
        "I'm not sure how I feel about that.",
        "That's amazing news!",
        "I'm a bit disappointed with the results."
    ]
    
    while True:
        # Simulate receiving emotion data
        # In a real implementation, this would get data from detect_emotion.py
        
        # Get facial emotion
        facial_emotion = np.random.choice(emotions)
        facial_confidence = np.random.uniform(0.6, 0.95)
        
        # Get audio/text emotion
        text_emotion = np.random.choice(emotions)
        text_confidence = np.random.uniform(0.6, 0.95)
        
        # Get SER emotion
        audio_emotion = np.random.choice(emotions)
        audio_confidence = np.random.uniform(0.6, 0.95)
        
        # Get transcribed text
        transcribed_text = np.random.choice(sample_texts)
        
        # Get cosine similarity
        cosine_similarity = np.random.uniform(0.3, 0.9)
        
        # Create data packet
        data = {
            "facial_emotion": (facial_emotion, facial_confidence),
            "text_emotion": (text_emotion, text_confidence),
            "audio_emotion": (audio_emotion, audio_confidence),
            "transcribed_text": transcribed_text,
            "cosine_similarity": cosine_similarity,
            "consistency_level": get_consistency_level(cosine_similarity)[0]
        }
        
        # Update queue
        if emotion_data_queue.full():
            try:
                emotion_data_queue.get_nowait()
            except queue.Empty:
                pass
                
        emotion_data_queue.put(data)
        time.sleep(1.0)  # Update every second

def update_dashboard():
    """Update dashboard with latest emotion data"""
    global latest_data
    
    # Get latest data
    try:
        latest_data = emotion_data_queue.get_nowait()
    except queue.Empty:
        pass
          # Get latest video frame
    try:
        frame = video_frame_queue.get_nowait()
        video_container.image(frame, channels="RGB", use_container_width=True)
    except queue.Empty:
        pass
        
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

# Start background threads
video_thread = threading.Thread(target=video_capture_thread)
emotion_thread = threading.Thread(target=receive_emotion_data_thread)
video_thread.daemon = True
emotion_thread.daemon = True
video_thread.start()
emotion_thread.start()

# Update dashboard in a loop
while True:
    update_dashboard()
    time.sleep(0.1)  # Update every 100ms