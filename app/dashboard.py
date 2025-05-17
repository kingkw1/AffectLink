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
    import time  # Make sure time is available throughout the function

    # Process updates from the UI update queue
    update_received = False
    force_update = False
    
    while not ui_update_queue.empty():
        try:
            new_data = ui_update_queue.get_nowait()
            
            # Check if this update should force a refresh
            if new_data.get('force_update', False):
                force_update = True
                
            # Update our data
            latest_data = new_data
            update_received = True
        except queue.Empty:
            pass
            
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
            update_dashboard.last_forced_update = current_time    # Get latest video frame
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
            
            frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                try:
                    # Track file modification time to avoid reloading the same frame repeatedly
                    mod_time = os.path.getmtime(frame_path)
                    current_time = time.time()
                    
                    # Only load if file has been modified since last check or if it's been a while
                    if (not hasattr(update_dashboard, 'last_frame_time') or 
                        mod_time > update_dashboard.last_frame_time or
                        (hasattr(update_dashboard, 'last_frame_check') and 
                         current_time - update_dashboard.last_frame_check > 2)):
                         
                        # Try to read the file, with better retry logic for potential file access issues
                        for attempt in range(5):  # Try up to 5 times (increased from 3)
                            try:
                                # Use IMREAD_COLOR flag explicitly to ensure correct loading
                                frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                                if frame is not None and frame.size > 0:
                                    update_dashboard.last_frame_time = mod_time
                                    
                                    # Log successful frame read occasionally
                                    if not hasattr(update_dashboard, 'read_count'):
                                        update_dashboard.read_count = 0
                                    update_dashboard.read_count += 1
                                    
                                    if update_dashboard.read_count % 20 == 0:
                                        print(f"Successfully read frame #{update_dashboard.read_count} from file, shape={frame.shape}")
                                    
                                    break
                                else:
                                    print(f"Attempt {attempt+1}: Frame read returned None or empty frame")
                                    time.sleep(0.05)  # Brief pause before retry
                            except Exception as retry_err:
                                print(f"Retry {attempt+1} reading frame: {retry_err}")
                                time.sleep(0.05 * (attempt + 1))  # Progressively longer delay
                        
                    update_dashboard.last_frame_check = current_time
                except Exception as e:
                    print(f"Error reading frame file: {e}")
            
    except Exception as e:
        print(f"Error retrieving video frame: {e}")
        pass
        
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
                
                # Track frame update time to detect stale frames
                if not hasattr(update_dashboard, 'last_frame_update_time'):
                    update_dashboard.last_frame_update_time = time.time()
                else:
                    update_dashboard.last_frame_update_time = time.time()
                    
                # Debug info about successful frame update
                if hasattr(update_dashboard, 'frame_update_count'):
                    update_dashboard.frame_update_count += 1
                else:
                    update_dashboard.frame_update_count = 1
                    
                if update_dashboard.frame_update_count % 10 == 0:  # Log every 10 frames
                    print(f"Successfully updated video frame #{update_dashboard.frame_update_count}")
            else:
                print("Received invalid frame for display")
        except Exception as e:
            print(f"Error displaying frame: {e}")
            
            # Fallback to last good frame if available
            if 'last_frame' in st.session_state and st.session_state.last_frame is not None:
                video_container.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
    else:
        # No frame received, display message or last frame if available
        if 'last_frame' in st.session_state and st.session_state.last_frame is not None:
            video_container.image(st.session_state.last_frame, channels="RGB", use_container_width=True)
        else:
            # Display placeholder message
            video_container.markdown("*Waiting for video feed...*")

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
    transcribed_text = latest_data.get('transcribed_text', '')
    
    # Remove timestamp if present (for cleaner display)
    if '[' in transcribed_text and ']' in transcribed_text:
        transcribed_text = transcribed_text.split('[')[0].strip()
    
    # Check if we have a new transcription to show
    if hasattr(update_dashboard, 'last_transcription'):
        if update_dashboard.last_transcription != transcribed_text and transcribed_text:
            print(f"Updated dashboard with new transcription: '{transcribed_text[:20]}...'")
    update_dashboard.last_transcription = transcribed_text
    
    # Display the transcription with special formatting if it's empty
    if not transcribed_text:
        text_container.markdown("**Latest transcription:**  \n*Waiting for speech...*")
    else:
        text_container.markdown(f"**Latest transcription:**  \n{transcribed_text}")

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