import cv2
from deepface import DeepFace
import os
import random
import shutil
import tempfile
import time
import logging # Added import

from constants import FACIAL_TO_UNIFIED

# Initialize a local logger for this module
logger = logging.getLogger(__name__)


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
			if process_video.frame_count % 2 == 0:  # Changed from 3 to 2
				# Add a small random component to filenames to avoid any caching issues
				random_suffix = random.randint(1000, 9999)

				# First write to a temp file to avoid partial reads by dashboard
				tmp_frame_path = os.path.join(tempfile.gettempdir(), f"affectlink_frame_tmp_{random_suffix}.jpg")
				frame_path = os.path.join(tempfile.gettempdir(), "affectlink_frame.jpg")

				# Save with higher quality (95) to temp file
				# Convert to RGB before saving to ensure proper color format
				if frame is not None:
					# Make a copy to avoid modifying the original
					frame_to_save = frame.copy()
					cv2.imwrite(tmp_frame_path, frame_to_save, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

					# Then move the temp file to the final location (atomic operation)
					shutil.move(tmp_frame_path, frame_path)
					logger.debug(f"Saved frame to {frame_path} (frame #{process_video.frame_count})")
		except Exception as e:
			# Saving to file is optional, so don't stop on errors
			logger.debug(f"Error saving frame to file: {e}")

		# Analyze face for emotion
		try:
			# Lock before accessing shared_state if necessary, though DeepFace is the dominant time sink here
			analysis = DeepFace.analyze(
				frame,
				actions=['emotion'],
				enforce_detection=False,
				silent=True
			)

			# DeepFace returns a list of dicts, one for each face. We'll take the first.
			if analysis and isinstance(analysis, list) and len(analysis) > 0:
				first_face = analysis[0]
				dominant_emotion = first_face['dominant_emotion']
				confidence = first_face['emotion'][dominant_emotion] / 100.0  # Normalize to 0-1
				all_emotions_scores = {k: v / 100.0 for k, v in first_face['emotion'].items()} # Normalize all

				# Map to unified emotion (optional, but good for consistency)
				unified_emotion = FACIAL_TO_UNIFIED.get(dominant_emotion, "unknown")

				logger.info(f"Facial emotion: {unified_emotion} ({confidence:.2f})")

				# Update shared_state
				with video_lock: # Ensure thread-safe update
					shared_state['facial_emotion'] = (unified_emotion, confidence)
					shared_state['facial_emotion_full_scores'] = all_emotions_scores
					logger.info(f"SHARED_STATE UPDATE: facial_emotion set to ({unified_emotion}, {confidence:.2f})")
					logger.debug(f"SHARED_STATE UPDATE: facial_emotion_full_scores set to {all_emotions_scores}")

			else:
				# No face detected or empty analysis
				with video_lock:
					shared_state['facial_emotion'] = ("unknown", 0.0)
					shared_state['facial_emotion_full_scores'] = {e: 0.0 for e in FACIAL_TO_UNIFIED.keys() if FACIAL_TO_UNIFIED[e]} # Empty scores for all unified
				logger.debug("No face detected or empty analysis from DeepFace.")

		except Exception as e:
			with video_lock:
				shared_state['facial_emotion'] = ("error", 0.0)
				shared_state['facial_emotion_full_scores'] = {}
			logger.error(f"Error during facial emotion analysis: {e}", exc_info=True)

		# Sleep briefly to avoid maxing out CPU
		time.sleep(0.05)

	# Clean up
	if cap is not None:
		cap.release()
	logger.info("Video processing thread exited")