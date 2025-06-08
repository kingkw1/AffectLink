import cv2
import os
import random
import shutil
import tempfile
import time
import logging # Added import

# Set up environment for DeepFace model caching
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
deepface_cache_dir = os.path.join(project_root, "models", "deepface_cache")

# Set the DEEPFACE_HOME environment variable
os.environ['DEEPFACE_HOME'] = deepface_cache_dir

# Now import DeepFace, it should use the DEEPFACE_HOME we just set.
from deepface import DeepFace

# Create the DeepFace cache directory if it doesn't exist
os.makedirs(deepface_cache_dir, exist_ok=True)

from constants import FACIAL_TO_UNIFIED

# Initialize a local logger for this module
logger = logging.getLogger(__name__)
logger.info(f"DeepFace models will be cached in: {deepface_cache_dir}")

def init_webcam(preferred_index=0, try_fallbacks=True):
	"""Initialize webcam with fallback options"""
	# Try to set up camera with preferred index first
	cap = cv2.VideoCapture(preferred_index)

	# If that didn't work, try alternate indices
	if not cap.isOpened() and try_fallbacks:
		logger.info(f"Camera index {preferred_index} failed, trying alternates")

		# Try indices 0 through 2
		for idx in range(3):
			if idx == preferred_index:
				continue  # Skip the one we already tried

			logger.info(f"Trying camera index {idx}")
			cap = cv2.VideoCapture(idx)
			if cap.isOpened():
				logger.info(f"Successfully opened camera with index {idx}")
				break

	# If still not open, try different backend APIs
	if not cap.isOpened() and try_fallbacks:
		# Try different backend APIs (DirectShow, etc)
		for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2]:
			logger.info(f"Trying camera index {preferred_index} with backend {backend}")
			cap = cv2.VideoCapture(preferred_index + backend)
			if cap.isOpened():
				logger.info(f"Successfully opened camera with backend {backend}")
				break

	if not cap.isOpened():
		logger.warning("Failed to open any webcam - video analysis disabled")
		return None

	# Configure for reasonable defaults
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
	cap.set(cv2.CAP_PROP_FPS, 20)

	return cap


def get_facial_emotion_from_frame(frame):
	"""Analyzes a single video frame to detect facial emotion."""
	try:
		analysis = DeepFace.analyze(
			frame,
			actions=['emotion'],
			enforce_detection=False,  # Don't error if no face, just return empty
			silent=True  # Suppress DeepFace's own console output
		)

		# DeepFace returns a list of dicts, one for each detected face
		if analysis and isinstance(analysis, list) and len(analysis) > 0:
			first_face = analysis[0]  # Process the first detected face
			face_region = first_face.get('region') # Get the region of the detected face
			# Get all raw scores from DeepFace, normalized to 0-1
			raw_emotion_scores = {k: v / 100.0 for k, v in first_face['emotion'].items()}

			valid_unified_scores = {}
			for raw_emotion, raw_score in raw_emotion_scores.items():
				unified_map_target = FACIAL_TO_UNIFIED.get(raw_emotion)
				# Only consider emotions that map to a valid UNIFIED_EMOTION
				if unified_map_target is not None:
					valid_unified_scores[unified_map_target] = max(valid_unified_scores.get(unified_map_target, 0.0), raw_score)
			
			final_emotion = "unknown"
			final_confidence = 0.0
			if valid_unified_scores and any(s > 0 for s in valid_unified_scores.values()):
				# Find the dominant emotion from the filtered valid_unified_scores
				final_emotion = max(valid_unified_scores, key=valid_unified_scores.get)
				final_confidence = valid_unified_scores[final_emotion]
			
			# logger.info(f"Facial emotion (filtered): {final_emotion} ({final_confidence:.2f})")
			return (final_emotion, final_confidence), raw_emotion_scores, face_region
		else:  # No face detected or analysis list is empty
			logger.debug("No face detected or analysis empty in get_facial_emotion_from_frame.")
			return ("unknown", 0.0), {}, None

	except Exception as e:
		logger.error(f"Error during facial emotion analysis in get_facial_emotion_from_frame: {e}", exc_info=True)
		return ("unknown", 0.0), {}, None


# Updated function signature
def process_video(shared_state, video_lock, video_started_event):
	"""Process video frames for facial emotion detection."""
	# Removed: global shared_state, face_cascade, video_frame_queue

	logger.info("Video processing thread started")

	# Removed face_cascade initialization, as DeepFace handles detection
	# if face_cascade is None:
	#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	# Try to initialize webcam
	cap = init_webcam(preferred_index=int(os.environ.get('WEBCAM_INDEX', '0')))

	if cap is None:
		logger.error("Failed to initialize webcam - skipping video analysis")
		video_started_event.set() # Set event even if webcam fails, so audio can start
		return

	logger.info("Webcam initialized successfully.")
	video_started_event.set() # Signal that video (or attempt) has started

	# Get shared frame queue from shared_state if available
	# (Assuming 'stop_event' key in shared_state might hold more than just the event)
	frame_queue = None
	if isinstance(shared_state.get('stop_event'), dict) and 'shared_frame_data' in shared_state['stop_event']:
		frame_queue = shared_state['stop_event']['shared_frame_data']
		logger.info(f"Found shared frame queue: {frame_queue}")

	# Process frames in a loop
	while cap.isOpened():
		# Check if we need to stop using shared_state['stop_event']
		stop_event_obj = shared_state.get('stop_event')
		if isinstance(stop_event_obj, dict): # Handling old structure if stop_event is a dict
			if stop_event_obj.get('stop', False):
				logger.info("Stop signal received in video processing (via dict)")
				break
		elif hasattr(stop_event_obj, 'is_set') and stop_event_obj.is_set(): # Handling if stop_event is an Event object
			logger.info("Stop event set in video processing")
			break
		elif stop_event_obj is True: # Handling simple boolean flag for stop
			logger.info("Stop signal (boolean) received in video processing")
			break


		# Capture frame
		success, frame = cap.read()
		if not success:
			logger.warning("Failed to read from webcam")
			# Try to re-initialize camera
			cap.release()
			time.sleep(1)
			cap = init_webcam()
			if cap is None:
				logger.error("Failed to reinitialize webcam - exiting video processing")
				break
			continue

		# Store the latest frame in shared_state
		shared_state['latest_frame'] = frame
		  # Share frame with dashboard if we have a queue
		if frame_queue is not None:
			try:
				# Try to add the frame to the queue without blocking
				if hasattr(frame_queue, 'put'):
					# Note: need to copy the frame as it might be modified elsewhere
					frame_queue.put(frame.copy(), block=False)
					logger.debug("Added frame to queue")
			except Exception as e:
				# Queue might be full, that's okay
				logger.debug(f"Couldn't add frame to queue: {e}")

		# Also save the latest frame to a shared file location for dashboard to access
		# This is a fallback/complement to the queue-based sharing
		try:
			# Increment frame counter
			if hasattr(process_video, 'frame_count'):
				process_video.frame_count += 1
			else:
				process_video.frame_count = 0

			# Increase the update frequency - save every 2nd frame instead of every 3rd
			# This provides more frequent updates to the dashboard
			if process_video.frame_count % 1 == 0:  # Changed from 3 to 2
				# Add a small random component to filenames to avoid any caching issues
				random_suffix = random.randint(1000, 9999)

				# First write to a temp file to avoid partial reads by dashboard
				tmp_frame_path = os.path.join(tempfile.gettempdir(), f"affectlink_frame_tmp_{random_suffix}.jpg")
				frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")

				# Analyze face for emotion (and get region) BEFORE drawing and saving
				# This ensures we have the region data if a face is detected
				facial_emotion_data, raw_emotion_scores, face_region = get_facial_emotion_from_frame(frame.copy()) # Operate on a copy for analysis

				# Draw bounding box on the original 'frame' if a face is detected
				if face_region:
					x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Green box, thickness 2

				# Save with higher quality (95) to temp file
				# Convert to RGB before saving to ensure proper color format
				if frame is not None:
					# Make a copy to avoid modifying the original
					frame_to_save = frame.copy() # This frame now potentially has the bounding box
					cv2.imwrite(tmp_frame_path, frame_to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

					# Then move the temp file to the final location (atomic operation)
					shutil.move(tmp_frame_path, frame_path)
					logger.debug(f"Saved frame to {frame_path} (frame #{process_video.frame_count})")
		except Exception as e:
			# Saving to file is optional, so don't stop on errors
			logger.debug(f"Error saving frame to file: {e}")

		# Update shared_state with emotion data (already retrieved above)
		try:
			if facial_emotion_data and facial_emotion_data[0] != "unknown" and facial_emotion_data[0] != "error":
				# Update shared_state
				with video_lock:
					shared_state['facial_emotion'] = facial_emotion_data
					# Store the full RAW (but normalized) scores from DeepFace for potential detailed view or debugging
					shared_state['facial_emotion_full_scores'] = raw_emotion_scores
			else:  # No face detected, or error in get_facial_emotion_from_frame
				with video_lock:
					shared_state['facial_emotion'] = facial_emotion_data if facial_emotion_data else ("unknown", 0.0)
					shared_state['facial_emotion_full_scores'] = raw_emotion_scores if raw_emotion_scores else {}
				if facial_emotion_data and facial_emotion_data[0] == "error":
					logger.debug("Error reported by get_facial_emotion_from_frame.")
				else:
					logger.debug("No face detected or analysis empty via get_facial_emotion_from_frame.")

		except Exception as e:
			with video_lock:
				shared_state['facial_emotion'] = ("error", 0.0)
				shared_state['facial_emotion_full_scores'] = {}
			logger.error(f"Error during facial emotion analysis using get_facial_emotion_from_frame: {e}", exc_info=True)

		# Sleep briefly to avoid maxing out CPU
		time.sleep(0.05)

	# Clean up
	if cap is not None:
		cap.release()
	logger.info("Video processing thread exited")