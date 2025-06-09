import os
os.environ['STREAMLIT_LOG_LEVEL'] = 'error' # Configure environment variables first

import warnings
warnings.filterwarnings("ignore", ".*missing ScriptRunContext.*") # Configure warnings next

import logging
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR) # Configure logging

# Standard Library Imports
import datetime
import json
import queue
import sys
import tempfile
import threading
import time
import traceback
from multiprocessing.queues import Empty as MPQueueEmpty

# Third-Party Imports
import cv2
import numpy as np
import streamlit as st

# Streamlit Page Configuration
st.set_page_config(
    page_title="AffectLink Emotion Dashboard",
    page_icon="😊",
    layout="wide"
)

# Session State Initialization
if 'enable_video' not in st.session_state:
    st.session_state.enable_video = True
if 'enable_audio' not in st.session_state:
    st.session_state.enable_audio = True
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = time.time()
if 'new_frame_received' not in st.session_state:
    st.session_state.new_frame_received = False
if 'loading_shown' not in st.session_state:
    st.session_state.loading_shown = True

# Global Variables
# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import shared data structures from detect_emotion.py (Organizational comment)

# Dev mode flag for standalone testing
DEV_MODE = True if __name__ == "__main__" else False

# Queues for inter-process/thread communication
video_frame_queue = None
emotion_data_queue = None
ui_update_queue = queue.Queue() # Thread-safe queue for UI updates

# Storage for latest emotion data
latest_data = {
    "facial_emotion": {"emotion": "unknown", "confidence": 0.0},
    "text_emotion": {"emotion": "unknown", "confidence": 0.0},
    "audio_emotion": {"emotion": "unknown", "confidence": 0.0},
    "transcribed_text": "Waiting for transcription...",
    "facial_emotion_full_scores": {},
    "audio_emotion_full_scores": [],
    "text_emotion_smoothed": {"emotion": "unknown", "confidence": 0.0},
    "audio_emotion_smoothed": {"emotion": "unknown", "confidence": 0.0},
    "overall_emotion": "unknown",
    "cosine_similarity": 0.0,
    "consistency_level": "Unknown",
    "update_id": "initial_0"
}

# Emotion colors for visualization
EMOTION_COLORS = {
    'neutral': '#AAAAAA',
    'happy': '#66BB6A',
    'sad': '#42A5F5',
    'angry': '#EF5350',
    'unknown': '#E0E0E0'
}

# Control flags and thread objects
should_stop = False
video_thread = None
audio_thread = None
stop_event = threading.Event()

# Global UI Placeholders (will be assigned st.empty() in UI layout)
video_container = None
text_container = None # For transcribed audio text
facial_emotion_container = None
text_emotion_container = None
audio_emotion_container = None
consistency_container = None
overall_emotion_container = None
video_placeholder = st.empty() # From original code, ensure it's preserved if used

# Main Streamlit app UI layout
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
    overall_emotion_container = st.empty()

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
        try:
            if not st.session_state.enable_video:
                time.sleep(0.5)
                continue
        except (AttributeError, KeyError):
            pass
            
        if not video_frame_queue.full():
            pass
        
        time.sleep(0.03)
    
    print("Dashboard video capture thread stopped")

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

# Update function for dashboard metrics
def update_metrics():
    global latest_data, facial_emotion_container, text_emotion_container, audio_emotion_container, consistency_container, text_container, overall_emotion_container

    if not facial_emotion_container:
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
            label="Verbal Emotion (Text)",
            value=f"{text_emotion_name_smoothed.capitalize()}",
            delta=f"{text_emotion_score_smoothed:.2f}",
            help="Smoothed dominant text-based emotion and confidence score."
        )

        audio_emotion_data_smoothed = ensure_emotion_dict(latest_data.get("audio_emotion_smoothed"))
        audio_emotion_name_smoothed = audio_emotion_data_smoothed.get("emotion", "unknown")
        audio_emotion_score_smoothed = audio_emotion_data_smoothed.get("confidence", 0.0)
        audio_emotion_container.metric(
            label="Voice Emotion (SER)",
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
            if text_container: # Check if text_container is initialized
                text_container.markdown("_Waiting for audio transcription..._") # Changed
        else:
            if text_container: # Check if text_container is initialized
                text_container.markdown(f"**Transcription:** {transcribed_text_value}") # Changed

        # Cosine Similarity / Consistency
        cosine_sim = latest_data.get("cosine_similarity", 0.0)
        consistency_level = get_consistency_level(cosine_sim)
        consistency_container.metric(
            label="Emotion Consistency",
            value=consistency_level,
            delta=f"Cosine Sim: {cosine_sim:.2f}",
            help="Consistency between facial and text emotion vectors."
        )

    except Exception as e:
        # st.error(f"Error updating metrics: {e}") # Avoid st calls if this is run in a thread sometimes
        print(f"Error updating dashboard metrics: {e}")
        print(traceback.format_exc())

# Define shared data receiver thread
def receive_emotion_data_thread(mp_queue, stop_event):
    """Thread to receive emotion data from detect_emotion.py via multiprocessing queue"""
    global should_stop

    last_check_time = time.time()
    last_update_count = 0
    last_successful_update = time.time()

    while not should_stop:
        if stop_event and stop_event.is_set():
            should_stop = True
            print("Stop event detected in emotion data receiver thread")
            break
            
        try:
            if not st.session_state.enable_audio:
                time.sleep(0.5)
                continue
        except (AttributeError, KeyError):
            pass

        queue_data_received = False
        if mp_queue:
            try:
                data = mp_queue.get(timeout=0.1)
                ui_update_queue.put(data)
                queue_data_received = True
                last_successful_update = time.time()
                last_update_count += 1
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Queue update #{last_update_count}: Received emotion data with transcription: '{data.get('transcribed_text', '')[:30]}...'")
                continue
            except (queue.Empty, MPQueueEmpty):
                pass
            except Exception as e:
                print(f"Error receiving emotion data from queue: {e}")

        try:
            emotion_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
            if os.path.exists(emotion_path):
                mod_time = os.path.getmtime(emotion_path)
                current_time = time.time()
                
                time_since_last_check = current_time - last_check_time
                time_since_last_update = current_time - last_successful_update
                
                if (not hasattr(receive_emotion_data_thread, 'last_emotion_time') or 
                    mod_time > getattr(receive_emotion_data_thread, 'last_emotion_time', 0) or
                    time_since_last_check > 1.0 or
                    time_since_last_update > 5.0):
                    
                    last_check_time = current_time
                    receive_emotion_data_thread.last_check_time = current_time
                    
                    for attempt in range(3):
                        try:
                            with open(emotion_path, 'r') as f:
                                data = json.load(f)
                                
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
                                
                                has_transcription = 'transcribed_text' in data and data['transcribed_text']
                                
                                meaningful_change = False
                                
                                if new_data or not hasattr(receive_emotion_data_thread, 'last_data'):
                                    meaningful_change = True
                                
                                transcription_changed = current_text != last_text and has_transcription
                                if transcription_changed:
                                    meaningful_change = True
                                
                                receive_emotion_data_thread.last_data = data.copy() if data else {}
                                receive_emotion_data_thread.last_text = current_text
                                receive_emotion_data_thread.last_update_id = current_update_id
                                last_successful_update = current_time
                                
                                if meaningful_change:
                                    ui_update_queue.put(data)
                                    
                                    if transcription_changed:
                                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] File update: New transcription detected: '{current_text[:30]}...'")
                                        
                                        data_with_force = data.copy()
                                        data_with_force['force_update'] = True
                                        ui_update_queue.put(data_with_force)
                                
                                receive_emotion_data_thread.last_emotion_time = mod_time
                                break
                                
                        except json.JSONDecodeError:
                            time.sleep(0.05)
                        except Exception as e:
                            print(f"Error reading emotion file (attempt {attempt+1}): {e}")
                            time.sleep(0.05)
                            
                    if time_since_last_update > 10.0 and not queue_data_received:
                        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] WARNING: No emotion updates received for {time_since_last_update:.1f} seconds")
        except Exception as e:
            print(f"Error checking emotion file: {e}")
            
        time.sleep(0.1)
    
    if should_stop:
        print("Audio data receiver thread stopped due to global stop flag")
    else:
        print("Audio data receiver thread stopped due to audio toggle disabled")


# Define the main function for the dashboard application
def main(emotion_queue_param=None, stop_event_param=None, frame_queue_param=None): # Renamed params to avoid conflict
    """Start the Streamlit dashboard using provided IPC queues"""
    global should_stop, video_frame_queue, emotion_data_queue, video_thread, audio_thread, stop_event
    
    video_frame_queue = frame_queue_param
    emotion_data_queue = emotion_queue_param
    
    # Use the global stop_event if no specific one is passed, or the passed one
    current_stop_event = stop_event_param if stop_event_param is not None else stop_event

    restart_video_thread()
    restart_audio_thread(queue=emotion_data_queue, stop_evt=current_stop_event)
    
    while not should_stop:
        update_dashboard()
        update_metrics()
        
        if current_stop_event and current_stop_event.is_set():
            should_stop = True
            print("Stop event detected in dashboard main loop")
            break
            
        time.sleep(0.1)
        
    print("Dashboard loop exited")

# Set up the app first with the functions before defining the UI (Organizational comment)
# This ensures functions are defined before use in callbacks (Organizational comment)

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

def restart_audio_thread(enabled=None, queue=None, stop_evt=None): # Renamed queue to queue_param
    """Safely restart the audio thread based on toggle state"""
    global audio_thread, should_stop, emotion_data_queue
    
    if enabled is None:
        try:
            enabled = st.session_state.enable_audio
        except (AttributeError, KeyError):
            enabled = True
    
    # If no queue provided, use the global one
    current_queue = queue if queue is not None else emotion_data_queue # Use param if provided
    
    # Stop the existing thread if it's running
    if audio_thread and audio_thread.is_alive():
        # The thread will exit on the next loop when checking should_stop
        print("Existing audio thread will terminate on next loop")
    
    # Only start a new thread if enabled and we have a queue
    if enabled and current_queue is not None:
        audio_thread = threading.Thread(
            target=receive_emotion_data_thread,
            args=(current_queue, stop_evt) # Use current_queue
        )
        audio_thread.daemon = True
        audio_thread.start()
        print("Audio thread started after toggle change")
    else:
        if not enabled:
            print("Audio processing disabled, no thread started") 
        else:
            print("No audio queue available, cannot start thread")

# If run directly (for testing)
if __name__ == "__main__":
    detector_running = os.environ.get("AFFECTLINK_DETECTOR_RUNNING") == "1"
    
    print(f"Detector running: {detector_running}")
    
    if detector_running:
        frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")
        emotion_path = os.path.join(tempfile.gettempdir(), "affectlink_emotion.json")
        
        # Create empty queues to pass to main (local to this block)
        main_frame_queue = queue.Queue(maxsize=5)
        main_emotion_queue = queue.Queue(maxsize=10)
        
        print(f"Running in detector-connected mode")
        print(f"Will look for frames at: {frame_path}")
        print(f"Will look for emotions at: {emotion_path}")
        
        main(main_emotion_queue, None, main_frame_queue)
    else:
        print("Starting dashboard in standalone demo mode")
        local_frame_queue = queue.Queue(maxsize=5)
        local_emotion_queue = queue.Queue(maxsize=10)
        
        def demo_data_provider():
            """Generate demo data for testing the dashboard standalone"""
            # Imports for numpy, time, random are at the top level now
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