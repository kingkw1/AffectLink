#!/usr/bin/env python3
"""
run_app.py - Main script for AffectLink with improved frame sharing
between the emotion detection and dashboard processes.
This script sets up the environment and starts the dashboard and emotion detection processes.
It handles device detection and initialization.
Usage:
    python run_app.py
"""

# Suppress missing ScriptRunContext and set Streamlit log level before import
import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'
import warnings
warnings.filterwarnings("ignore", ".*missing ScriptRunContext.*")
import logging
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

# Then import Streamlit
import streamlit as st
st.set_page_config(
    page_title="AffectLink Emotion Dashboard",
    page_icon="😊",
    layout="wide"
)

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
import pandas as pd
import plotly.express as px

# Initialize session state variables at the very beginning
if 'enable_video' not in st.session_state:
    st.session_state.enable_video = True
if 'enable_audio' not in st.session_state:
    st.session_state.enable_audio = True
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
# Add a timestamp for the session start to detect fresh frames
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()
# Flag to track if we've seen a new frame this session
if 'new_frame_received' not in st.session_state:
    st.session_state.new_frame_received = False
# Add a flag to track if we've shown the loading animation
if 'loading_shown' not in st.session_state:
    st.session_state.loading_shown = True  # Start by showing loading animation

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
    "facial_emotion": {"emotion": "unknown", "confidence": 0.0},
    "text_emotion": {"emotion": "unknown", "confidence": 0.0},
    "audio_emotion": {"emotion": "unknown", "confidence": 0.0},
    "transcribed_text": "Waiting for transcription...", # Default text
    "facial_emotion_full_scores": {},
    "audio_emotion_full_scores": [],
    "text_emotion_smoothed": {"emotion": "unknown", "confidence": 0.0}, # Add smoothed placeholders
    "audio_emotion_smoothed": {"emotion": "unknown", "confidence": 0.0},# Add smoothed placeholders
    "overall_emotion": "unknown", # Add overall emotion placeholder
    "cosine_similarity": 0.0,
    "consistency_level": "Unknown",
    "update_id": "initial_0"
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

# Define global variables to track thread state
video_thread = None
audio_thread = None
stop_event = threading.Event() if not 'stop_event' in globals() else globals()['stop_event']

def get_consistency_level(cosine_sim):
    """Convert cosine similarity to consistency level label"""
    if cosine_sim >= 0.8:
        return "High"
    elif cosine_sim >= 0.6:
        return "Medium"
    elif cosine_sim >= 0.3:
        return "Low"
    elif cosine_sim <= 0.01:
        return "Unknown"
    else:
        return "Very Low"

def video_capture_thread():
    """
    Thread to receive video frames from the emotion detection process.
    We don't open the camera here - we just take frames from the shared queue.
    """
    print("Dashboard video capture thread started")
    
    # Keep checking the queue until should_stop is set or video toggle is disabled
    while not should_stop:
        # Safely check the session state
        try:
            if not st.session_state.enable_video:
                time.sleep(0.5)  # Sleep longer when disabled
                continue
        except (AttributeError, KeyError):
            # Session state might not be ready yet, assume enabled
            pass
            
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
    import time  # Make sure time is available throughout the function
    import json # For parsing emotion data if it comes as a string

    # Process updates from the UI update queue
    update_received = False
    force_update = False
    new_data_from_queue = None
    
    while not ui_update_queue.empty():
        try:
            raw_data_from_queue = ui_update_queue.get_nowait()
            
            # Ensure data from queue is a dictionary
            if isinstance(raw_data_from_queue, str):
                try:
                    new_data_from_queue = json.loads(raw_data_from_queue)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from queue: {raw_data_from_queue}")
                    continue # Skip this item
            elif isinstance(raw_data_from_queue, dict):
                new_data_from_queue = raw_data_from_queue
            else:
                print(f"Unexpected data type from queue: {type(raw_data_from_queue)}")
                continue # Skip this item

            # Check if this update should force a refresh
            if new_data_from_queue.get('force_update', False):
                force_update = True
                
            # Update our global latest_data with the new data from the queue
            # This is crucial: ensure latest_data is updated here
            latest_data.update(new_data_from_queue) 
            update_received = True
            # Log the update to confirm it's happening
            # print(f"Dashboard: Updated latest_data from queue. New text: {latest_data.get('transcribed_text', '')[:20]}")

        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing UI update queue: {e}")
            
    # Update a counter to force refresh periodically even if no new data
    if not hasattr(update_dashboard, 'refresh_counter'):
        update_dashboard.refresh_counter = 0
        update_dashboard.last_forced_update = time.time()
    
    # Increment counter and check for forced refresh
    current_time = time.time()
    update_dashboard.refresh_counter += 1
    
    # Reduce reliance on forced refreshes to prevent dashboard restarts
    # Only force refresh when absolutely needed (e.g., new transcription)
    # Otherwise rely on Streamlit's natural update cycle
    if force_update:
        # Record when we're forcing an update
        if not hasattr(update_dashboard, 'refresh_count'):
            update_dashboard.refresh_count = 0
        update_dashboard.refresh_count += 1
        
        try:
            # Only log occasionally to avoid console spam
            if update_dashboard.refresh_count % 5 == 0:
                print(f"Forcing UI refresh #{update_dashboard.refresh_count} for new transcription")
            
            # Update the timestamp regardless of whether we actually trigger a rerun
            update_dashboard.last_forced_update = current_time
            
            # For now, DON'T trigger an actual rerun as this may be causing 
            # dashboard restart issues. Instead let Streamlit's normal refresh cycle handle it.
            # This is a workaround to prevent dashboard instability.
            # The UI elements will still update on the next natural refresh.
            
        except Exception as e:
            print(f"Error in refresh handling: {e}")
            update_dashboard.last_forced_update = current_time
    
    # Get latest video frame (only if video processing is enabled)
    frame = None
    
    # Check if video processing is enabled by toggle (with fallback)
    video_enabled = True
    try:
        video_enabled = st.session_state.enable_video
    except (AttributeError, KeyError):
        # Default to enabled if session state not available
        pass
        
    if not video_enabled:
        # If video is disabled, display a message instead of trying to get a frame
        video_container.markdown("### 🚫 Video processing disabled")
        video_container.markdown("*Enable the 'Enable Video Processing' toggle above to view video feed*")
        
        # Add a visual indicator for disabled state
        disabled_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(disabled_frame, "Video Processing Off", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        video_container.image(disabled_frame, channels="BGR", use_container_width=True)
    else:
        try:
            # Check if we have a shared queue or a local queue
            if video_frame_queue and hasattr(video_frame_queue, 'get_nowait'):
                # Try to get a frame without blocking
                try:
                    frame = video_frame_queue.get_nowait()
                    if frame is not None:
                        st.session_state.new_frame_received = True
                        # print("Received frame from queue")
                except (queue.Empty, MPQueueEmpty):
                    # No new frame available
                    pass
            elif video_frame_queue and hasattr(video_frame_queue, 'get'):
                # For multiprocessing.Queue
                try:
                    frame = video_frame_queue.get(block=False)
                    if frame is not None:
                        st.session_state.new_frame_received = True
                        # print("Received frame from multiprocessing queue")
                except (MPQueueEmpty, Exception):
                    pass
                    
            # If we couldn't get a frame from the queue, check for saved frame file
            if frame is None:
                # Check for frame saved to temporary file by detector
                import tempfile
                
                frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    try:
                        # Track file modification time to avoid reloading the same frame repeatedly
                        mod_time = os.path.getmtime(frame_path)
                        current_time = time.time()
                        
                        # Only accept the frame if:
                        # 1. It was modified after our session started OR
                        # 2. We've already received a fresh frame this session
                        fresh_frame = mod_time > st.session_state.session_start_time or st.session_state.new_frame_received
                        
                        # Only load if it's a fresh frame and has been modified since last check or if it's been a while
                        if fresh_frame and (
                            not hasattr(update_dashboard, 'last_frame_time') or 
                            mod_time > update_dashboard.last_frame_time or
                            (hasattr(update_dashboard, 'last_frame_check') and 
                             current_time - update_dashboard.last_frame_check > 2)):
                            
                            # Try to read the file, with better retry logic for potential file access issues
                            for attempt in range(5):  # Try up to 5 times
                                try:
                                    # Use IMREAD_COLOR flag explicitly to ensure correct loading
                                    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                                    if frame is not None and frame.size > 0:
                                        update_dashboard.last_frame_time = mod_time
                                        st.session_state.new_frame_received = True
                                        
                                        # Log successful frame read occasionally
                                        if not hasattr(update_dashboard, 'read_count'):
                                            update_dashboard.read_count = 0
                                        update_dashboard.read_count += 1
                                        
                                        if update_dashboard.read_count % 20 == 0:
                                            print(f"Successfully read frame #{update_dashboard.read_count} from file, shape={frame.shape}")
                                        
                                        break
                                    else:
                                        #print(f"Attempt {attempt+1}: Frame read returned None or empty frame")
                                        time.sleep(0.05)  # Brief pause before retry
                                except Exception as retry_err:
                                    #print(f"Retry {attempt+1} reading frame: {retry_err}")
                                    time.sleep(0.05 * (attempt + 1))  # Progressively longer delay
                        
                        update_dashboard.last_frame_check = current_time
                    except Exception as e:
                        print(f"Error reading frame file: {e}")
            
        except Exception as e:
            print(f"Error retrieving video frame: {e}")
        
    # Always attempt to display the frame
    if frame is not None:
        try:
            # Always convert BGR to RGB for consistent display 
            # OpenCV always returns BGR, so we should always convert
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                display_frame = frame.copy() if frame is not None else None
                
            # Display the video frame
            if display_frame is not None and display_frame.size > 0:
                video_container.image(display_frame, channels="RGB", use_container_width=True)
                
                # Store the frame for later use
                st.session_state.last_frame = display_frame
                
                # Mark that we've received a frame, so we don't show loading animation again
                if not st.session_state.new_frame_received:
                    st.session_state.new_frame_received = True
                    if st.session_state.loading_shown:
                        st.session_state.loading_shown = False
                        print("First webcam frame received. Hiding loading animation.")
                
                # Track frame update time to detect stale frames
                if not hasattr(update_dashboard, 'last_frame_update_time'):
                    update_dashboard.last_frame_update_time = time.time()
                else:
                    update_dashboard.last_frame_update_time = time.time()
            else:
                print("Received invalid frame for display")
        except Exception as e:
            print(f"Error displaying frame: {e}")
            
            # If we've already seen frames, fallback to the last good frame rather than showing loading
            if st.session_state.new_frame_received and st.session_state.last_frame is not None:
                video_container.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
            else:
                # First run or no good frame yet - show loading
                display_loading_indicator()
    else:
        # No frame received
        if st.session_state.new_frame_received and st.session_state.last_frame is not None:
            # If we've previously gotten frames, display the last good one
            video_container.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
        else:
            # Only show loading indicator if we haven't received any frames yet
            display_loading_indicator()

def display_loading_indicator():
    """Display a loading animation while waiting for the webcam to initialize"""
    # Create a loading animation using a pulsing circle
    loading_img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Get current time for the animation
    t = time.time() % 2  # Cycle every 2 seconds
    pulse = int(30 * (1 + np.sin(t * np.pi * 2))) + 10  # Radius oscillates between 10 and 70
    
    # Draw circle in center
    center = (200, 150)
    cv2.circle(loading_img, center, pulse, (50, 120, 200), -1)  # Filled blue circle
    cv2.circle(loading_img, center, pulse, (255, 255, 255), 2)  # White border
    
    # Add text
    cv2.putText(loading_img, "Loading webcam...", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (255, 255, 255), 2)
    cv2.putText(loading_img, "Please wait", (150, 270), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (200, 200, 200), 1)
    
    # Display loading image
    video_container.image(loading_img, channels="BGR", use_container_width=True)
    st.session_state.loading_shown = True

# Define metrics and containers for displaying results
video_placeholder = st.empty()
overall_emotion_container = st.empty()
transcribed_text_area = st.empty()
facial_plot_area = st.empty() # Initialize facial_plot_area
audio_plot_area = st.empty()  # Ensure audio_plot_area is initialized
text_plot_area = st.empty()   # Ensure text_plot_area is initialized
facial_emotion_container = None
text_emotion_container = None
audio_emotion_container = None
consistency_container = None

# Update function for dashboard metrics
def update_metrics():
    global latest_data, facial_emotion_container, text_emotion_container, audio_emotion_container, consistency_container, transcribed_text_area, facial_plot_area, audio_plot_area, text_plot_area

    if not facial_emotion_container: # Ensure containers are initialized
        return

    try:
        # Defensive: convert list/tuple to dict for emotion fields
        def ensure_emotion_dict(val):
            if isinstance(val, dict):
                return val
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return {"emotion": str(val[0]), "confidence": float(val[1])}
            return {"emotion": "unknown", "confidence": 0.0}

        facial_emotion_data = ensure_emotion_dict(latest_data.get("facial_emotion"))
        facial_emotion_name = facial_emotion_data.get("emotion", "unknown")
        facial_emotion_score = facial_emotion_data.get("confidence", 0.0)
        facial_emotion_container.metric(
            label="Facial Emotion",
            value=f"{facial_emotion_name.capitalize()}",
            delta=f"{facial_emotion_score:.2f}",
            help="Dominant facial emotion and confidence score."
        )

        text_emotion_data_smoothed = ensure_emotion_dict(latest_data.get("text_emotion_smoothed"))
        text_emotion_name_smoothed = text_emotion_data_smoothed.get("emotion", "unknown")
        text_emotion_score_smoothed = text_emotion_data_smoothed.get("confidence", 0.0)
        text_emotion_container.metric(
            label="Text Emotion (Smoothed)",
            value=f"{text_emotion_name_smoothed.capitalize()}",
            delta=f"{text_emotion_score_smoothed:.2f}",
            help="Smoothed dominant text-based emotion and confidence score."
        )

        audio_emotion_data_smoothed = ensure_emotion_dict(latest_data.get("audio_emotion_smoothed"))
        audio_emotion_name_smoothed = audio_emotion_data_smoothed.get("emotion", "unknown")
        audio_emotion_score_smoothed = audio_emotion_data_smoothed.get("confidence", 0.0)
        audio_emotion_container.metric(
            label="Audio Emotion (Smoothed)",
            value=f"{audio_emotion_name_smoothed.capitalize()}",
            delta=f"{audio_emotion_score_smoothed:.2f}",
            help="Smoothed dominant audio-based emotion (SER) and confidence score."
        )

        # Overall Emotion
        overall_emotion_value = latest_data.get("overall_emotion", "unknown")
        overall_emotion_container.metric(
            label="Overall Emotion",
            value=f"{overall_emotion_value.capitalize()}",
            help="Estimated overall emotional state."
        )

        # Transcribed Text
        transcribed_text_value = latest_data.get("transcribed_text", "Waiting for transcription...")
        if not transcribed_text_value or transcribed_text_value.strip() == "" or transcribed_text_value.lower() == "waiting for audio transcription...":
            transcribed_text_area.markdown("_Waiting for audio transcription..._")
        else:
            transcribed_text_area.markdown(f"**Transcription:** {transcribed_text_value}")

        # Cosine Similarity / Consistency
        cosine_sim = latest_data.get("cosine_similarity", 0.0)
        consistency_level = get_consistency_level(cosine_sim)
        consistency_container.metric(
            label="Emotion Consistency",
            value=consistency_level,
            delta=f"Cosine Sim: {cosine_sim:.2f}",
            help="Consistency between facial and text emotion vectors."
        )

        # Update Plots
        # Facial Emotion Full Scores Plot
        facial_scores = latest_data.get("facial_emotion_full_scores", {})
        if facial_scores and isinstance(facial_scores, dict) and any(facial_scores.values()):
            # Ensure scores are Python floats
            facial_scores_py = {k: float(v) for k, v in facial_scores.items()}
            df_facial = pd.DataFrame(list(facial_scores_py.items()), columns=['Emotion', 'Score']).sort_values(by='Score', ascending=False)
            fig_facial = px.bar(df_facial, x='Emotion', y='Score', title='Facial Emotion Distribution', color='Emotion', color_discrete_map=EMOTION_COLORS)
            fig_facial.update_layout(yaxis_title="Confidence", xaxis_title="Emotion")
            facial_plot_area.plotly_chart(fig_facial, use_container_width=True)
        else:
            facial_plot_area.empty()


        # Audio Emotion Full Scores Plot (SER)
        audio_scores_list = latest_data.get("audio_emotion_full_scores", [])
        if audio_scores_list and isinstance(audio_scores_list, list) and len(audio_scores_list) > 0:
            # Ensure scores are Python floats
            audio_scores_py = [{"emotion": item.get("emotion", "unknown"), "score": float(item.get("score", 0.0))} for item in audio_scores_list]
            df_audio = pd.DataFrame(audio_scores_py).sort_values(by='score', ascending=False)
            if not df_audio.empty:
                fig_audio = px.bar(df_audio, x='emotion', y='score', title='Audio Emotion (SER) Distribution', color='emotion', color_discrete_map=EMOTION_COLORS)
                fig_audio.update_layout(yaxis_title="Confidence", xaxis_title="Emotion")
                audio_plot_area.plotly_chart(fig_audio, use_container_width=True)
            else:
                audio_plot_area.empty()
        else:
            audio_plot_area.empty()

        # Text Emotion Full Scores (from text_emotion_history, last entry)
        text_history = latest_data.get("text_emotion_history", [])
        if text_history and isinstance(text_history, list) and len(text_history) > 0:
            last_text_scores_entry = text_history[-1] # get the most recent entry
            if isinstance(last_text_scores_entry, dict) and "scores" in last_text_scores_entry:
                text_scores = last_text_scores_entry.get("scores", {})
                if text_scores and isinstance(text_scores, dict) and any(text_scores.values()):
                    # Ensure scores are Python floats
                    text_scores_py = {k: float(v) for k, v in text_scores.items()}
                    df_text = pd.DataFrame(list(text_scores_py.items()), columns=['Emotion', 'Score']).sort_values(by='Score', ascending=False)
                    fig_text = px.bar(df_text, x='Emotion', y='Score', title='Text Emotion Distribution (Current Segment)', color='Emotion', color_discrete_map=EMOTION_COLORS)
                    fig_text.update_layout(yaxis_title="Confidence", xaxis_title="Emotion")
                    text_plot_area.plotly_chart(fig_text, use_container_width=True)
                else:
                    text_plot_area.empty()
            else:
                text_plot_area.empty()
        else:
            text_plot_area.empty()

    except Exception as e:
        # st.error(f"Error updating metrics: {e}") # Avoid st calls if this is run in a thread sometimes
        print(f"Error updating dashboard metrics: {e}")
        import traceback
        print(traceback.format_exc())

# Define shared data receiver thread
def receive_emotion_data_thread(mp_queue, stop_event):
    """Thread to receive emotion data from detect_emotion.py via multiprocessing queue"""
    global should_stop
    import time  # Make sure time is available throughout the function
    import datetime  # Add datetime for better logging

    last_check_time = time.time()
    last_update_count = 0
    last_successful_update = time.time()

    while not should_stop:
        if stop_event and stop_event.is_set():
            should_stop = True
            print("Stop event detected in emotion data receiver thread")
            break
            
        # Safely check the session state
        try:
            if not st.session_state.enable_audio:
                time.sleep(0.5)  # Sleep longer when disabled
                continue
        except (AttributeError, KeyError):
            # Session state might not be ready yet, assume enabled
            pass

        # First try to get data from the queue if available
        queue_data_received = False
        if mp_queue:
            try:
                # Try to get data from the multiprocessing queue with a timeout
                data = mp_queue.get(timeout=0.1)  # Use shorter timeout to check file more frequently
                # Pass data to the UI update queue
                ui_update_queue.put(data)
                queue_data_received = True
                last_successful_update = time.time()
                last_update_count += 1
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Queue update #{last_update_count}: Received emotion data with transcription: '{data.get('transcribed_text', '')[:30]}...'")
                continue  # Skip file check if we got data from queue
            except (queue.Empty, MPQueueEmpty):
                # If no data is available, we'll check the file next
                pass
            except Exception as e:
                print(f"Error receiving emotion data from queue: {e}")

        # Check for emotion data saved to file as backup
        try:
            import tempfile
            import json
            
            emotion_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
            if os.path.exists(emotion_path):
                # Check file modification time to avoid reloading same data
                mod_time = os.path.getmtime(emotion_path)
                current_time = time.time()
                
                # Force check at least every second even if not modified
                time_since_last_check = current_time - last_check_time
                time_since_last_update = current_time - last_successful_update
                
                # Process if:
                # 1. File has been modified since last check, OR
                # 2. It's been more than 1 second since last check (force periodic checks), OR
                # 3. It's been more than 5 seconds since last successful update (recovery mechanism)
                if (not hasattr(receive_emotion_data_thread, 'last_emotion_time') or 
                    mod_time > getattr(receive_emotion_data_thread, 'last_emotion_time', 0) or
                    time_since_last_check > 1.0 or
                    time_since_last_update > 5.0):
                    
                    # Update last check time regardless of whether we read successfully
                    last_check_time = current_time
                    receive_emotion_data_thread.last_check_time = current_time
                    
                    # Try to read the file with retry logic
                    for attempt in range(3):
                        try:
                            with open(emotion_path, 'r') as f:
                                data = json.load(f)
                                
                                # Always update the UI with new data from the file
                                # This ensures we get the latest transcriptions and emotions
                                
                                # Check if the data is truly new or different
                                new_data = True
                                last_text = ""
                                last_update_id = ""
                                
                                if hasattr(receive_emotion_data_thread, 'last_data'):
                                    if (data.get('update_id', '') == getattr(receive_emotion_data_thread, 'last_update_id', '')):
                                        new_data = False
                                    last_text = getattr(receive_emotion_data_thread, 'last_text', '')
                                    last_update_id = getattr(receive_emotion_data_thread, 'last_update_id', '')
                                
                                current_text = data.get('transcribed_text', '')
                                current_update_id = data.get('update_id', '')
                                
                                # Make sure transcription is populated
                                has_transcription = 'transcribed_text' in data and data['transcribed_text']
                                
                                # Improved update logic - check for meaningful changes
                                meaningful_change = False
                                
                                # Check if this is new data or first data
                                if new_data or not hasattr(receive_emotion_data_thread, 'last_data'):
                                    meaningful_change = True
                                
                                # Check specially for transcription changes
                                transcription_changed = current_text != last_text and has_transcription
                                if transcription_changed:
                                    meaningful_change = True
                                
                                # Always update timestamps and tracking variables
                                receive_emotion_data_thread.last_data = data.copy() if data else {}
                                receive_emotion_data_thread.last_text = current_text
                                receive_emotion_data_thread.last_update_id = current_update_id
                                last_successful_update = current_time
                                
                                # Put data in queue if we have a meaningful change
                                if meaningful_change:
                                    # Send normal update
                                    ui_update_queue.put(data)
                                    
                                    # Log if we have a new transcription
                                    if transcription_changed:
                                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] File update: New transcription detected: '{current_text[:30]}...'")
                                        
                                        # Only force an update for transcription changes
                                        # This reduces the number of forced updates
                                        data_with_force = data.copy()
                                        data_with_force['force_update'] = True
                                        ui_update_queue.put(data_with_force)
                                
                                # Update time even if data hasn't changed
                                receive_emotion_data_thread.last_emotion_time = mod_time
                                break
                                
                        except json.JSONDecodeError:
                            # File might be partially written, try again after a short delay
                            time.sleep(0.05)
                        except Exception as e:
                            print(f"Error reading emotion file (attempt {attempt+1}): {e}")
                            time.sleep(0.05)
                            
                    # If it's been too long without updates, log a warning
                    if time_since_last_update > 10.0 and not queue_data_received:
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] WARNING: No emotion updates received for {time_since_last_update:.1f} seconds")
        except Exception as e:
            print(f"Error checking emotion file: {e}")
            
        # Short sleep to avoid burning CPU
        time.sleep(0.1)
    
    # Check which condition caused the thread to stop
    if should_stop:
        print("Audio data receiver thread stopped due to global stop flag")
    else:
        print("Audio data receiver thread stopped due to audio toggle disabled")

# Main Streamlit app
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
    overall_emotion_container = st.empty() # Added in previous step

# Define the main function for the dashboard application
def main(emotion_queue=None, stop_event=None, frame_queue=None):
    """Start the Streamlit dashboard using provided IPC queues"""
    global should_stop, video_frame_queue, emotion_data_queue, video_thread, audio_thread
    
    # Assign provided queues to global variables
    video_frame_queue = frame_queue
    emotion_data_queue = emotion_queue
    
    # Start video capture thread only if video processing is enabled
    # Use our helper function for consistency
    restart_video_thread()
    
    # Start emotion data receiving thread if a queue is provided and audio processing is enabled
    # Use our helper function for consistency
    restart_audio_thread(queue=emotion_data_queue, stop_evt=stop_event)
    
    # Update dashboard in a loop
    while not should_stop:
        update_dashboard()
        update_metrics()  # Refresh emotion analysis metrics
        
        # Check if stop event is set
        if stop_event and stop_event.is_set():
            should_stop = True
            print("Stop event detected in dashboard main loop")
            break
            
        time.sleep(0.1)  # Update every 100ms
        
    print("Dashboard loop exited")

# Set up the app first with the functions before defining the UI
# This ensures functions are defined before use in callbacks

def restart_video_thread(enabled=None):
    """Safely restart the video thread based on toggle state"""
    global video_thread, should_stop
    
    # If no value is provided, check session state (with fallback)
    if enabled is None:
        try:
            enabled = st.session_state.enable_video
        except (AttributeError, KeyError):
            # If session state isn't ready, default to enabled
            enabled = True
    
    # Stop the existing thread if it's running
    if video_thread and video_thread.is_alive():
        # The thread will exit on the next loop when checking should_stop
        # We don't need to set a local stop flag since threads check the global should_stop
        print("Existing video thread will terminate on next loop")
    
    # Only start a new thread if enabled
    if enabled:
        video_thread = threading.Thread(target=video_capture_thread)
        video_thread.daemon = True
        video_thread.start()
        print("Video thread started after toggle change")
    else:
        print("Video processing disabled, no thread started")

def restart_audio_thread(enabled=None, queue=None, stop_evt=None):
    """Safely restart the audio thread based on toggle state"""
    global audio_thread, should_stop, emotion_data_queue
    
    # If no value is provided, check session state (with fallback)
    if enabled is None:
        try:
            enabled = st.session_state.enable_audio
        except (AttributeError, KeyError):
            # If session state isn't ready, default to enabled
            enabled = True
    
    # If no queue provided, use the global one
    if queue is None:
        queue = emotion_data_queue
    
    # Stop the existing thread if it's running
    if audio_thread and audio_thread.is_alive():
        # The thread will exit on the next loop when checking should_stop
        print("Existing audio thread will terminate on next loop")
    
    # Only start a new thread if enabled and we have a queue
    if enabled and queue is not None:
        audio_thread = threading.Thread(
            target=receive_emotion_data_thread,
            args=(queue, stop_evt)
        )
        audio_thread.daemon = True
        audio_thread.start()
        print("Audio thread started after toggle change")
    else:
        if not enabled:
            print("Audio processing disabled, no thread started") 
        else:
            print("No audio queue available, cannot start thread")

# We're now using the implementations defined above, removing this duplicate definition

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