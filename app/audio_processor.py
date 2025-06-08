from collections import deque
import sounddevice as sd
import numpy as np
import soundfile as sf
import hashlib
import os
import random
import time
import traceback
import librosa
import torch
import tempfile
import logging  # Added for local logger
import requests
import base64
import json

from constants import SER_TO_UNIFIED, TEXT_TO_UNIFIED, UNIFIED_EMOTIONS, AUDIO_CHUNK_SIZE, AUDIO_SAMPLE_RATE
# Removed: from main_processor import logger, shared_state

# Local logger for this module
logger = logging.getLogger(__name__)

# Initialize audio buffer with a fixed size
audio_buffer = deque(maxlen=AUDIO_CHUNK_SIZE * AUDIO_SAMPLE_RATE)

'''
THIS MEANS THAT THIS MUST BE SET IN POWERSHELL WITH THE FOLLOWING:

$env:AFFECTLINK_WHISPER_API_URL="https://localhost:9600/invocations"
$env:AFFECTLINK_SER_API_URL="https://localhost:56651/invocations"
$env:AFFECTLINK_SPEECHTEXT_API_URL="https://localhost:54250/invocations"

Together with the os.genv line, this accomplishes the same as the previous hardcoded URL.
    WHISPER_API_URL = "https://localhost:60049/invocations"
'''
WHISPER_API_URL = os.getenv("AFFECTLINK_WHISPER_API_URL", "https://localhost:60049/invocations")
SER_API_URL = os.getenv("AFFECTLINK_SER_API_URL", "https://localhost:55853/invocations") 
TEXT_CLASSIFIER_API_URL = os.getenv("AFFECTLINK_SPEECHTEXT_API_URL", "https://localhost:54250/invocations")

def call_text_classifier_api(text_input: str, api_url: str):
    """
    Calls the deployed text emotion classifier API.
    Expected response format: {"predictions": [{"label": "emotion", "score": value}]}
    """
    payload = {
        "dataframe_records": [
            {"text": text_input}
        ]
    }
    headers = {
        "Content-Type": "application/json",
    }

    logger.debug(f"Calling text classifier API at {api_url} with text: '{text_input[:50]}...'")

    try:
        # Use verify=False for localhost/self-signed certs from local HP AI Studio deployment
        response = requests.post(api_url, headers=headers, json=payload, verify=False, timeout=10)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()

        predictions = response_json.get('predictions', [])
        if predictions and len(predictions) > 0 and isinstance(predictions[0], dict):
            # The API returns the top prediction directly as a dictionary in a list
            # Return the first (and likely only) dictionary of prediction
            return predictions[0] # Returns a single dictionary like {'label': 'joy', 'score': 0.98}
        else:
            logger.warning(f"Text classifier API returned unexpected prediction format: {response_json}")
            return {"label": "unknown", "score": 0.0} # Return a default unknown emotion
    except requests.exceptions.Timeout:
        logger.error(f"Text classifier API call timed out after 10 seconds to {api_url}")
        return {"label": "unknown", "score": 0.0}
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Text classifier API connection error to {api_url}: {e}")
        return {"label": "unknown", "score": 0.0}
    except requests.exceptions.RequestException as e:
        logger.error(f"Text classifier API request error: {e}")
        return {"label": "unknown", "score": 0.0}
    except json.JSONDecodeError as e:
        logger.error(f"Text classifier API response JSON decode error: {e}. Response text: {response.text}")
        return {"label": "unknown", "score": 0.0}
    except Exception as e:
        logger.error(f"An unexpected error occurred during text classifier API call: {e}")
        return {"label": "unknown", "score": 0.0}
    

def transcribe_audio_whisper_api(audio_path, api_url):
    """
    Transcribe audio file using the deployed Whisper API.
    """
    try:
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file missing for API transcription: {audio_path}")
            return None

        # Read audio file and encode as base64
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": {
                "audio_base64": [audio_base64]
            }
        }

        logger.info(f"Sending audio to Whisper API at {api_url}...")
        start_time = time.time()
        # Use verify=False for localhost/self-signed certs during development
        response = requests.post(api_url, headers=headers, json=payload, verify=False)
        response_time = time.time() - start_time

        if response.status_code == 200:
            predictions = response.json().get('predictions', [])
            if predictions and len(predictions) > 0 and 'transcription' in predictions[0]:
                transcribed_text = predictions[0]['transcription'].strip()
                logger.info(f"API Transcription successful in {response_time:.2f}s: '{transcribed_text}'")
                return transcribed_text
            else:
                logger.error(f"API response missing 'transcription' in predictions: {response.json()}")
                return None
        else:
            logger.error(f"Whisper API call failed with status code {response.status_code}: {response.text}")
            return None

    except requests.exceptions.ConnectionError as ce:
        logger.error(f"Connection error to Whisper API at {api_url}: {ce}")
        return None
    except requests.exceptions.Timeout as te:
        logger.error(f"Timeout connecting to Whisper API at {api_url}: {te}")
        return None
    except requests.exceptions.RequestException as re:
        logger.error(f"Request error to Whisper API at {api_url}: {re}")
        return None
    except Exception as e:
        logger.error(f"Error during Whisper API transcription: {e}")
        logger.error(traceback.format_exc())
        return None


def analyze_audio_emotion_ser_api(audio_path, api_url):
    """
    Analyze audio file for emotion using the deployed SER API.
    Returns dominant emotion, its score, and a list of all emotion scores.
    """
    try:
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file missing for SER API analysis: {audio_path}")
            return "unknown", 0.0, []

        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "dataframe_records": [
                {
                    "audio_base64": audio_base64
                }
            ]
        }

        logger.info(f"Sending audio to SER API at {api_url} for emotion analysis...")
        start_time = time.time()
        # Use verify=False for localhost/self-signed certs during development
        response = requests.post(api_url, headers=headers, json=payload, verify=False)
        response_time = time.time() - start_time

        if response.status_code == 200:
            # The API now returns a JSON object with a 'predictions' key,
            # which is a list of dictionaries (one for each input audio).
            # Each dictionary contains 'dominant_audio_emotion', 'dominant_audio_emotion_score',
            # and 'full_audio_emotion_scores'.
            predictions = response.json().get('predictions', [])
            
            if predictions and len(predictions) > 0:
                # Assuming we send one audio chunk at a time, we'll take the first prediction.
                first_prediction = predictions[0]
                
                dominant_emotion = first_prediction.get('dominant_audio_emotion', 'unknown').strip()
                score = first_prediction.get('dominant_audio_emotion_score', 0.0)
                full_results = first_prediction.get('full_audio_emotion_scores', []) # This will be the list of dicts

                logger.info(f"SER API analysis successful in {response_time:.2f}s: Dominant emotion: '{dominant_emotion}' (Score: {score:.4f})")
                return dominant_emotion, score, full_results
            else:
                logger.error(f"SER API response missing valid predictions or empty: {response.json()}")
                return "unknown", 0.0, []
        else:
            logger.error(f"SER API call failed with status code {response.status_code}: {response.text}")
            return "unknown", 0.0, []

    except requests.exceptions.ConnectionError as ce:
        logger.error(f"Connection error to SER API at {api_url}: {ce}")
        return "unknown", 0.0, []
    except requests.exceptions.Timeout as te:
        logger.error(f"Timeout connecting to SER API at {api_url}: {te}")
        return "unknown", 0.0, []
    except requests.exceptions.RequestException as re:
        logger.error(f"Request error to SER API at {api_url}: {re}")
        return "unknown", 0.0, []
    except Exception as e:
        logger.error(f"Error during SER API analysis: {e}")
        logger.error(traceback.format_exc())
        return "unknown", 0.0, []
    
    
def transcribe_audio_whisper(audio_path, whisper_model):
    """
    Transcribe audio file using Whisper locally.
    """
    try:
        # Generate a truly unique ID for this transcription attempt
        # Include more entropy sources to ensure we don't get cached results
        current_time_ms = int(time.time() * 1000)
        random_salt = random.randint(0, 1000000)
        process_id = os.getpid()
        audio_file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        unique_string = f"{audio_path}_{current_time_ms}_{random_salt}_{process_id}_{audio_file_size}"
        transcription_id = hashlib.md5(unique_string.encode()).hexdigest()[:8]

        logger.info(f"[{transcription_id}] Starting local transcription of {audio_path} at {current_time_ms}")

        if not os.path.exists(audio_path):
            logger.warning(f"[{transcription_id}] Audio file missing: {audio_path}")
            return None

        file_size = os.path.getsize(audio_path)
        if file_size < 2000:  # Increased minimum size threshold
            logger.warning(f"[{transcription_id}] Audio file too small: {file_size} bytes")
            return None

        # Check audio content
        audio_data, sample_rate = sf.read(audio_path)
        audio_duration = len(audio_data) / sample_rate
        audio_rms = np.sqrt(np.mean(audio_data**2))
        audio_peak = np.max(np.abs(audio_data))
        audio_std = np.std(audio_data)

        logger.info(f"Audio file stats: path={audio_path}, duration={audio_duration:.2f}s, RMS={audio_rms:.6f}, peak={audio_peak:.6f}, std={audio_std:.6f}, samples={len(audio_data)}")

        # Save detailed audio stats to help diagnose issues
        segments = 5
        segment_size = len(audio_data) // segments
        if segment_size > 0:
            segment_stats = []
            for i in range(segments):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size
                segment = audio_data[start_idx:end_idx]
                segment_rms = np.sqrt(np.mean(segment**2))
                segment_stats.append(f"{segment_rms:.6f}")
            logger.info(f"Audio segment RMS values: {', '.join(segment_stats)}")

        # If audio is essentially silence, don't bother transcribing
        if audio_rms < 0.005:  # Increased RMS threshold for silence detection
            logger.warning(f"[{transcription_id}] Audio appears to be silence (RMS={audio_rms:.6f}), skipping transcription")
            return None

        # Log whisper model device
        if hasattr(whisper_model, 'device'):
            device_info = f"on device {whisper_model.device}"
        else:
            device_info = "(device unknown)"

        logger.info(f"[{transcription_id}] Running Whisper transcription {device_info} on {audio_duration:.2f}s audio")

        # Proceed with transcription for valid audio
        start_time = time.time()

        # Use more aggressive options for better transcription
        try:
            # Use more parameters to improve accuracy and ensure actual speech is detected
            result = whisper_model.transcribe(
                audio_path,
                language="en",  # Force English language
                temperature=0.0,  # More deterministic output
                no_speech_threshold=0.5,  # Slightly less aggressive filtering
                logprob_threshold=-1.0,  # Default, less permissive
                condition_on_previous_text=False,  # Don't condition on previous text to avoid repetition
                initial_prompt="This is a transcription of speech."  # Help the model understand context
            )
        except Exception as e:
            logger.error(f"[{transcription_id}] Error during whisper transcription: {e}")
            logger.error(traceback.format_exc())
            return None

        transcription_time = time.time() - start_time

        # Log the result details
        transcribed_text = result['text'].strip() if result and 'text' in result else ""

        # Extra validation to ensure we're not repeating the same text due to caching
        if hasattr(transcribe_audio_whisper, 'last_transcription') and (
            transcribe_audio_whisper.last_transcription == transcribed_text or
            (not transcribed_text and transcribe_audio_whisper.last_transcription == "")
        ):
            logger.warning(f"[{transcription_id}] Detected repeated transcription: '{transcribed_text or 'empty'}'")

            # Try forcing different text by adding noise parameter to next call
            if not hasattr(transcribe_audio_whisper, 'repeat_count'):
                transcribe_audio_whisper.repeat_count = 0
            transcribe_audio_whisper.repeat_count += 1

            # If empty transcription or same text repeatedly, try forcing buffer reset sooner
            if not transcribed_text or transcribe_audio_whisper.repeat_count > 1:
                logger.warning(f"[{transcription_id}] {'Empty' if not transcribed_text else 'Same'} text repeated, forcing audio buffer reset")

                # Reset the counter and clear the last transcription
                transcribe_audio_whisper.repeat_count = 0
                transcribe_audio_whisper.last_transcription = None

                # Signal to the caller that they should reset the buffer
                return "RESET_BUFFER"
        else:
            # Reset repeat counter when we get different text
            transcribe_audio_whisper.repeat_count = 0             # Store for comparison next time
            transcribe_audio_whisper.last_transcription = transcribed_text

            # Log detailed results
            if transcribed_text:
                logger.info(f"[{transcription_id}] Transcribed in {transcription_time:.2f}s: '{transcribed_text}' (length: {len(transcribed_text)})")

            # Log segments if available for more detailed analysis
            if 'segments' in result and len(result['segments']) > 0:
                segments_info = []
                for i, segment in enumerate(result['segments']):
                    if 'text' in segment:
                        segments_info.append(f"[{i}] {segment.get('text', '')[:30]}...")
                logger.info(f"[{transcription_id}] Segments: {' | '.join(segments_info)}")
            else:
                logger.warning(f"[{transcription_id}] Transcription returned empty result after {transcription_time:.2f}s despite valid audio")

        return transcribed_text
    except Exception as e:
        logger.error(f"Local Transcription error: {e}")
        logger.error(traceback.format_exc())
        return None


def process_audio_chunk_from_file(
    audio_chunk_data,  # np.ndarray (float32)
    audio_sample_rate, # int
    whisper_model,
    text_emotion_classifier, # transformers.Pipeline
    audio_feature_extractor, # transformers.AutoFeatureExtractor
    audio_emotion_classifier # transformers.AutoModelForAudioClassification
):
    """
    Processes a single audio chunk from data for transcription and emotion analysis.
    Returns:
        A tuple: (transcribed_text, text_emotion_data, audio_emotion_data, text_emotion_full_scores, audio_emotion_full_scores)
    """
    transcribed_text = ""
    text_emotion_data = ("unknown", 0.0)
    audio_emotion_data = ("unknown", 0.0)
    text_emotion_full_scores = {}
    audio_emotion_full_scores = {}
    temp_audio_file_name = None

    try:
        # 1. Save audio_chunk_data to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_audio_file_name = tmpfile.name
            sf.write(temp_audio_file_name, audio_chunk_data, audio_sample_rate)
        logger.debug(f"Temporarily saved audio chunk to {temp_audio_file_name}")

        # 2. Transcription
        try:
            raw_transcription = transcribe_audio_whisper(temp_audio_file_name, whisper_model)
            if raw_transcription == "RESET_BUFFER" or raw_transcription is None:
                transcribed_text = ""
            else:
                transcribed_text = raw_transcription
        except Exception as e_transcribe:
            logger.error(f"Error during transcription of {temp_audio_file_name}: {e_transcribe}", exc_info=True)
            transcribed_text = ""


        # 3. Text Emotion Classification
        if transcribed_text and transcribed_text.strip():
            try:
                text_results_outer = text_emotion_classifier(transcribed_text)
                
                if text_results_outer and isinstance(text_results_outer, list) and len(text_results_outer) > 0:
                    text_scores_list = text_results_outer[0] # This is the list of score dicts like [{'label': 'sad', 'score': 0.9}, ...]
                    if isinstance(text_scores_list, list) and len(text_scores_list) > 0:
                        text_emotion_full_scores = {item['label']: item['score'] for item in text_scores_list}
                        
                        best_text_emotion = max(text_scores_list, key=lambda x: x['score'])
                        dominant_text_label_raw = best_text_emotion['label']
                        dominant_text_confidence = best_text_emotion['score']
                        
                        unified_text_emotion = TEXT_TO_UNIFIED.get(dominant_text_label_raw.lower(), "unknown")
                        text_emotion_data = (unified_text_emotion, dominant_text_confidence)
                    else:
                        logger.warning(f"Unexpected format for text_scores_list (expected list of dicts): {text_scores_list}")
                        text_emotion_data = ("unknown", 0.0)
                else:
                    logger.warning(f"Unexpected format for text_results_outer (expected list containing list of dicts): {text_results_outer}")
                    text_emotion_data = ("unknown", 0.0)

            except Exception as e_text:
                logger.error(f"Error during text emotion classification for text: '{transcribed_text}' - {e_text}", exc_info=True)
                text_emotion_data = ("error", 0.0)
        else:
            text_emotion_data = ("unknown", 0.0) # No text or only whitespace

        # 4. Audio Emotion Classification
        try:
            if audio_chunk_data is not None and len(audio_chunk_data) > 0:
                inputs = audio_feature_extractor(
                    audio_chunk_data, 
                    sampling_rate=audio_sample_rate, 
                    return_tensors="pt",
                    padding=True # Add padding for short chunks
                )
                
                # Move inputs to the same device as the model
                model_device = audio_emotion_classifier.device
                inputs_on_device = {k: v.to(model_device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = audio_emotion_classifier(**inputs_on_device).logits
                
                probs = torch.softmax(logits, dim=-1).squeeze()
                
                model_config_labels = audio_emotion_classifier.config.id2label
                
                # Ensure probs is iterable and matches model_config_labels size
                if probs.ndim == 0: # Single score
                     probs = probs.unsqueeze(0)

                if len(probs) == len(model_config_labels):
                    audio_emotion_full_scores = {model_config_labels[i]: probs[i].item() for i in range(len(model_config_labels))}
                    
                    if audio_emotion_full_scores:
                        # Determine dominant raw audio emotion
                        dominant_audio_label_raw = max(audio_emotion_full_scores, key=audio_emotion_full_scores.get)
                        audio_confidence_raw = audio_emotion_full_scores[dominant_audio_label_raw]

                        # Map to unified emotion - this part requires careful handling if multiple raw map to one unified
                        # For simplicity, find the unified emotion corresponding to the most dominant raw emotion
                        unified_audio_emotion_label = SER_TO_UNIFIED.get(dominant_audio_label_raw.lower(), "unknown")
                        
                        # If you need to aggregate scores for unified emotions (e.g. multiple 'anger_x' map to 'anger')
                        # you would iterate through audio_emotion_full_scores and sum them up for unified labels.
                        # For now, using the confidence of the top raw mapped emotion.
                        audio_emotion_data = (unified_audio_emotion_label, audio_confidence_raw)
                    else:
                        audio_emotion_data = ("unknown", 0.0)
                else:
                    logger.warning(f"Mismatch between number of probabilities ({len(probs)}) and model labels ({len(model_config_labels)}). Skipping audio emotion.")
                    audio_emotion_data = ("unknown", 0.0)
            else:
                audio_emotion_data = ("unknown", 0.0) # No audio data to process

        except Exception as e_audio:
            logger.error(f"Error during audio emotion analysis: {e_audio}", exc_info=True)
            audio_emotion_data = ("error", 0.0)

    except Exception as e_main:
        logger.error(f"Error processing audio chunk from file: {e_main}", exc_info=True)
        transcribed_text = ""
        text_emotion_data = ("error", 0.0)
        audio_emotion_data = ("error", 0.0)

    finally:
        # 5. Cleanup temporary file
        if temp_audio_file_name and os.path.exists(temp_audio_file_name):
            try:
                os.remove(temp_audio_file_name)
                logger.debug(f"Successfully deleted temporary audio file {temp_audio_file_name}")
            except Exception as e_delete:
                logger.warning(f"Could not delete temporary audio file {temp_audio_file_name}: {e_delete}")

    return transcribed_text, text_emotion_data, audio_emotion_data, text_emotion_full_scores, audio_emotion_full_scores


def analyze_audio_emotion_full(audio_path, ser_model, ser_processor):
    """
    Get full detailed audio emotion analysis
    """
    try:
        # Load audio (mono, 16kHz)
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Process audio
        inputs = ser_processor(waveform, sampling_rate=16000, return_tensors="pt")

        model_device = next(ser_model.parameters()).device
        logger.debug(f"Model is on device: {model_device}")

        for k in inputs:
            inputs[k] = inputs[k].to(model_device)

        with torch.no_grad():
            logits = ser_model(**inputs).logits

        scores = torch.nn.functional.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Get the actual labels from the model's config
        model_labels_dict = ser_model.config.id2label
        ser_actual_labels = [model_labels_dict[i] for i in sorted(model_labels_dict.keys())]

        all_results = []
        for i, score in enumerate(scores):
            emotion = ser_actual_labels[i] if i < len(ser_actual_labels) else f"unknown-{i}"
            all_results.append({
                "emotion": emotion,
                "score": float(score)
            })

        result_sorted = sorted(all_results, key=lambda x: x["score"], reverse=True)

        if result_sorted:
            top_emotion = result_sorted[0]["emotion"]
            top_score = result_sorted[0]["score"]
            return top_emotion, top_score, result_sorted
        else:
            return "neutral", 0.0, []

    except Exception as e:
        logger.error(f"Audio emotion full analysis error: {e}")
        return "neutral", 0.0, []


def moving_average(scores):
    """
    Calculate moving average for a list of scores
    """
    if not scores:
        return 0
    return sum(scores) / len(scores)


def audio_processing_loop(shared_state, audio_lock, whisper_model, text_emotion_classifier, ser_model, ser_processor, device, video_started_event, use_whisper_api: bool = True, use_ser_api: bool = True, use_text_classifier_api: bool = True):
    """Process audio for speech-to-text and emotion analysis"""
    global audio_buffer  # Only audio_buffer is global now
    logger.info("Audio processing thread started")

    last_transcription_change_time = time.time()
    last_transcription = ""

    try:
        logger.info("Waiting for video processing to start...")
        video_started_event.wait()
        logger.info("Video processing started, beginning audio processing")

        chunk_duration = 5
        smoothing_window = 3
        emotion_window = deque(maxlen=smoothing_window)
        score_window = deque(maxlen=smoothing_window)
        audio_emotion_window = deque(maxlen=smoothing_window)
        audio_score_window = deque(maxlen=smoothing_window)

        stop_event_obj = shared_state.get('stop_event')
        while not (isinstance(stop_event_obj, dict) and stop_event_obj.get('stop', False)) and not (hasattr(stop_event_obj, 'is_set') and stop_event_obj.is_set()):
            try:
                # Initialize variables at the start of each loop iteration
                unified_text_emotion = "unknown"
                text_score = 0.0
                raw_audio_emotion = "unknown" # Also initialize raw_audio_emotion
                audio_score = 0.0 # Also initialize audio_score
                audio_emotions_full_results = [] # Initialize full audio results

                # Use the audio buffer collected in record_audio
                if shared_state['latest_audio'] is None or len(shared_state['latest_audio']) < 1000:
                    logger.info("No audio data available yet, waiting...")
                    time.sleep(1)
                    continue

                # Save the latest audio to a temporary file for processing
                temp_wav = os.path.join(tempfile.gettempdir(), f"affectlink_temp_audio_{int(time.time()*1000)}.wav")
                audio_data = shared_state['latest_audio'].copy()  # Create a copy to avoid race conditions
                logger.debug(f"Saving audio chunk with {len(audio_data)} samples for processing")

                # Calculate some basic audio stats for diagnostics
                audio_mean = np.mean(np.abs(audio_data))
                audio_max = np.max(np.abs(audio_data))
                logger.info(f"Audio data stats: mean={audio_mean:.6f}, max={audio_max:.6f}, samples={len(audio_data)}")

                # Resample to 16kHz for Whisper - audio_data is already 16kHz if audio_sample_rate is 16000
                sf.write(temp_wav, audio_data, 16000)

                # ADDED: Detailed logging of audio data before transcription
                if audio_data is not None and audio_data.size > 0:
                    logger.info(f"APL_PRE_TRANSCRIPTION_AUDIO: samples={len(audio_data)}, duration={len(audio_data)/16000.0:.2f}s, dtype={audio_data.dtype}, min={np.min(audio_data):.4f}, max={np.max(audio_data):.4f}, mean_abs={np.mean(np.abs(audio_data)):.4f}, sum_abs={np.sum(np.abs(audio_data)):.4f}")
                else:
                    logger.warning("APL_PRE_TRANSCRIPTION_AUDIO: No audio data or empty audio data before transcription attempt.")

                # Transcribe audio to text based on the toggle
                logger.debug("Transcribing audio...")
                if use_whisper_api: # <--- CONDITIONAL LOGIC FOR WHISPER API
                    logger.info("Using Whisper API for transcription.")
                    text = transcribe_audio_whisper_api(temp_wav, WHISPER_API_URL)
                else:
                    logger.info("Using local Whisper model for transcription.")
                    text = transcribe_audio_whisper(temp_wav, whisper_model) # Existing local call

                logger.info(f"Whisper output: '{text}'") # Added log

                # Handle transcription results and buffer resets
                PLACEHOLDER_TEXT = "Waiting for audio transcription..."
                if text == "RESET_BUFFER":
                    logger.warning("Whisper requested buffer reset. Clearing audio buffer.")
                    with audio_lock:
                        audio_buffer.clear()
                        shared_state['latest_audio'] = None
                        # Only set placeholder if current is not a fresh, valid transcription
                        if shared_state['transcribed_text'] == PLACEHOLDER_TEXT or not shared_state['transcribed_text'].strip():
                            shared_state['transcribed_text'] = PLACEHOLDER_TEXT
                        shared_state['audio_reset_time'] = time.time()  # Signal update
                    if hasattr(audio_processing_loop, 'buffer_reset_count'):
                        audio_processing_loop.buffer_reset_count = 0
                    if hasattr(audio_processing_loop, 'last_texts'):
                        audio_processing_loop.last_texts = []
                    time.sleep(0.5)  # Brief pause after reset
                    continue  # Skip the rest of this iteration
                elif text and text.strip() and text.strip() != PLACEHOLDER_TEXT:
                    shared_state['transcribed_text'] = text
                # If text is None, empty, or the placeholder, do NOT update shared_state['transcribed_text']
                # This preserves the last real transcription.

                # Use a simple counter for buffer management
                if not hasattr(audio_processing_loop, 'buffer_reset_count'):
                    audio_processing_loop.buffer_reset_count = 0
                audio_processing_loop.buffer_reset_count += 1

                # Periodic buffer management - reset more frequently
                # This should ideally happen BEFORE transcription of the current chunk if it's based on staleness
                # For now, ensure it doesn't overwrite a fresh transcription from THIS iteration.
                if audio_processing_loop.buffer_reset_count >= 5:
                    logger.info("Periodically clearing audio buffer to ensure fresh audio")
                    with audio_lock:
                        audio_buffer.clear()
                        shared_state['latest_audio'] = None
                        # Only set placeholder if the current text isn't the one we just got
                        if not (text and text.strip() and text.strip() != PLACEHOLDER_TEXT):
                            shared_state['transcribed_text'] = PLACEHOLDER_TEXT
                        shared_state['audio_reset_time'] = time.time()
                    random_sleep = random.uniform(0.2, 0.7)
                    time.sleep(random_sleep)
                    audio_processing_loop.buffer_reset_count = 0
                    logger.warning(f"Audio buffer reset performed at {time.time():.3f} with {random_sleep:.3f}s sleep")

                # Track the last few transcriptions to detect if we're stuck in a loop
                if not hasattr(audio_processing_loop, 'last_texts'):
                    audio_processing_loop.last_texts = []
                    audio_processing_loop.no_new_text_counter = 0

                if text and text.strip() and text.strip() != PLACEHOLDER_TEXT:
                    logger.info(f"Transcribed text: {text[:50]}...")
                    text_with_timestamp = f"{text} [ts:{time.time():.2f}]"
                    audio_processing_loop.last_texts.append(text)
                    if len(audio_processing_loop.last_texts) > 5:
                        audio_processing_loop.last_texts.pop(0)
                    if len(audio_processing_loop.last_texts) >= 2:
                        if audio_processing_loop.last_texts[-1] == audio_processing_loop.last_texts[-2]:
                            logger.warning(f"Detected same transcription twice in a row - force clearing buffer: '{text}'")
                            with audio_lock:
                                audio_buffer.clear()
                                shared_state['latest_audio'] = None
                                shared_state['transcribed_text'] = PLACEHOLDER_TEXT  # Explicit placeholder
                            random_sleep = random.uniform(0.1, 0.5)
                            time.sleep(random_sleep)
                            audio_processing_loop.last_texts = []
                            audio_processing_loop.buffer_reset_count = 0
                elif not text or not text.strip():
                    logger.warning("No transcription generated - will try again with fresh audio")

                # Get all text emotion scores
                # Ensure text is valid before calling the text classifier
                if text and text.strip() and text.strip() != PLACEHOLDER_TEXT:
                    logger.info(f"Attempting text emotion classification for: '{text[:100]}'")
                    # --- NEW CONDITIONAL LOGIC FOR TEXT CLASSIFIER API ---
                    if use_text_classifier_api:
                        logger.info("Using Text Classifier API for emotion analysis.")
                        text_emotion_scores_raw = call_text_classifier_api(text, TEXT_CLASSIFIER_API_URL)
                    else:
                        logger.info("Using local Text Classifier model for emotion analysis.")
                        # The local model likely returns a list of lists of dicts, e.g., [[{'label': 'joy', 'score': 0.9}]]
                        # This will need to be handled differently if the local model's output changes.
                        text_emotion_scores_raw = text_emotion_classifier(text) # Existing local call
                    # --- END NEW CONDITIONAL LOGIC ---

                    logger.info(f"Raw text emotion scores: {text_emotion_scores_raw}")

                    # --- MODIFIED LOGIC HERE ---
                    text_emotion_data = None
                    if isinstance(text_emotion_scores_raw, dict) and 'label' in text_emotion_scores_raw and 'score' in text_emotion_scores_raw:
                        # This is the expected format from your API call (single dict)
                        text_emotion_data = text_emotion_scores_raw
                    elif isinstance(text_emotion_scores_raw, list) and \
                         len(text_emotion_scores_raw) > 0 and isinstance(text_emotion_scores_raw[0], list) and \
                         len(text_emotion_scores_raw[0]) > 0 and isinstance(text_emotion_scores_raw[0][0], dict) and \
                         'label' in text_emotion_scores_raw[0][0] and 'score' in text_emotion_scores_raw[0][0]:
                        # This handles the original HuggingFace pipeline output format [[{...}]]
                        # where the local model might still return it this way.
                        text_emotion_data = text_emotion_scores_raw[0][0]
                    elif isinstance(text_emotion_scores_raw, list) and \
                         len(text_emotion_scores_raw) > 0 and isinstance(text_emotion_scores_raw[0], dict) and \
                         'label' in text_emotion_scores_raw[0] and 'score' in text_emotion_scores_raw[0]:
                        # This handles if the local model sometimes returns [{'label': 'joy', 'score': 0.9}]
                        text_emotion_data = text_emotion_scores_raw[0]

                    if text_emotion_data:
                        # Map raw emotion to unified emotion
                        unified_text_emotion = TEXT_TO_UNIFIED.get(text_emotion_data['label'], "unknown")
                        text_score = text_emotion_data['score']

                        # Construct unified_text_scores dictionary for consistency
                        # This assumes you want a dict with all unified emotions, with the dominant one having its score
                        unified_text_scores = {emotion: 0.0 for emotion in UNIFIED_EMOTIONS}
                        unified_text_scores[unified_text_emotion] = text_score

                        logger.info(f"Unified text scores: {unified_text_scores}")
                        shared_state['text_emotion_unified_scores'] = unified_text_scores # Store for main_processor

                        shared_state['text_emotion_history'].append({
                            'timestamp': time.time(),
                            'scores': unified_text_scores
                        })

                        # Determine dominant text emotion from unified scores for current display
                        if unified_text_scores and any(s > 0 for s in unified_text_scores.values()):
                            dominant_text_emotion = max(unified_text_scores, key=unified_text_scores.get)
                            confidence = unified_text_scores[dominant_text_emotion]
                            shared_state['text_emotion'] = (dominant_text_emotion, confidence)
                            logger.info(f"SHARED_STATE UPDATE: text_emotion set to ({dominant_text_emotion}, {confidence:.2f})")
                            unified_text_emotion = dominant_text_emotion # Update for smoothing
                            text_score = confidence                       # Update for smoothing
                        else:
                            logger.warning(f"Unified text scores are empty, all zero, or invalid after normalization: {unified_text_scores}")
                            shared_state['text_emotion'] = ("unknown", 0.0)
                            unified_text_emotion = "unknown"
                            text_score = 0.0
                    else:
                        logger.warning(f"Text classifier returned no valid scores or unexpected format after all checks: {text_emotion_scores_raw}")
                        shared_state['text_emotion'] = ("unknown", 0.0)
                        unified_text_emotion = "unknown"
                        text_score = 0.0
                else: # if text is invalid or placeholder
                    logger.warning(f"Skipping text emotion classification because text is invalid or placeholder: '{text}'")
                    shared_state['text_emotion'] = ("unknown", 0.0)
                    unified_text_emotion = "unknown"
                    text_score = 0.0

                # Get audio emotion scores
                logger.debug("Analyzing audio emotion...")
                try:
                    if use_ser_api: # <--- NEW CONDITIONAL LOGIC FOR SER API
                        logger.info("Using SER API for audio emotion analysis.")
                        raw_audio_emotion, audio_score, audio_emotions_full_results = analyze_audio_emotion_ser_api(temp_wav, SER_API_URL)
                    else:
                        logger.info("Using local SER model for audio emotion analysis.")
                        raw_audio_emotion, audio_score, audio_emotions_full_results = analyze_audio_emotion_full(temp_wav, ser_model, ser_processor)

                    # Initialize unified_audio_emotion for broader scope
                    unified_audio_emotion = "unknown"

                    if raw_audio_emotion and audio_score is not None:
                        # Map raw SER emotion to unified emotion
                        unified_audio_emotion = SER_TO_UNIFIED.get(raw_audio_emotion, "unknown")

                        logger.info(f"Audio emotion: {unified_audio_emotion} ({audio_score:.2f})")
                        # Update the shared state directly for dashboard access
                        shared_state['audio_emotion'] = (unified_audio_emotion, audio_score)
                        if audio_emotions_full_results:
                            shared_state['audio_emotion_full_scores'] = audio_emotions_full_results
                        else:
                            shared_state['audio_emotion_full_scores'] = []

                        logger.info(f"SHARED_STATE UPDATE: audio_emotion set to ({unified_audio_emotion}, {audio_score:.2f})")
                    else:
                        logger.debug("Audio emotion analysis returned no valid results or score was None")
                        # Ensure shared_state reflects no valid audio emotion if analysis fails
                        shared_state['audio_emotion'] = ("unknown", 0.0)
                        shared_state['audio_emotion_full_scores'] = []

                except Exception as audio_err:
                    logger.error(f"Error analyzing audio emotion: {audio_err}")
                    logger.error(traceback.format_exc()) # Add traceback
                    shared_state['audio_emotion'] = ("unknown", 0.0)
                    shared_state['audio_emotion_full_scores'] = []

                # Clean up temp file
                if isinstance(temp_wav, str) and os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                        logger.debug(f"Successfully removed temp audio file: {temp_wav}")
                    except OSError as e:
                        logger.warning(f"Error removing temp audio file {temp_wav}: {e}")

                # Smoothing text emotions - similar to original but with better error handling
                if unified_text_emotion != "unknown" and text_score is not None: # Use unified_text_emotion and check text_score
                    emotion_window.append(unified_text_emotion) # Use unified_text_emotion
                    score_window.append(text_score)

                    if len(emotion_window) == smoothing_window:
                        most_frequent_emotion = max(set(emotion_window), key=list(emotion_window).count)
                        relevant_scores = [s for e, s in zip(emotion_window, score_window) if e == most_frequent_emotion]
                        smoothed_score = moving_average(relevant_scores) if relevant_scores else 0.0
                        shared_state['text_emotion_smoothed'] = (most_frequent_emotion, smoothed_score)
                        logger.info(f"SHARED_STATE UPDATE: text_emotion_smoothed set to ({most_frequent_emotion}, {smoothed_score:.2f}) after smoothing")
                    else:
                        shared_state['text_emotion_smoothed'] = (unified_text_emotion, text_score)
                        logger.info(f"SHARED_STATE UPDATE: text_emotion_smoothed set to ({unified_text_emotion}, {text_score:.2f}) (insufficient window)")
                else:
                    shared_state['text_emotion_smoothed'] = ("unknown", 0.0)
                    logger.info(f"SHARED_STATE UPDATE: text_emotion_smoothed set to ('unknown', 0.0) due to invalid current text emotion")

                # Smoothing audio emotions
                if unified_audio_emotion != "unknown" and audio_score is not None:
                    audio_emotion_window.append(unified_audio_emotion)
                    audio_score_window.append(audio_score)

                    if len(audio_emotion_window) == smoothing_window:
                        most_frequent_audio_emotion = max(set(audio_emotion_window), key=list(audio_emotion_window).count)
                        relevant_audio_scores = [s for e, s in zip(audio_emotion_window, audio_score_window) if e == most_frequent_audio_emotion]
                        smoothed_audio_score = moving_average(relevant_audio_scores) if relevant_audio_scores else 0.0
                        shared_state['audio_emotion_smoothed'] = (most_frequent_audio_emotion, smoothed_audio_score)
                        logger.info(f"SHARED_STATE UPDATE: audio_emotion_smoothed set to ({most_frequent_audio_emotion}, {smoothed_audio_score:.2f}) after smoothing")
                    else:
                        shared_state['audio_emotion_smoothed'] = (unified_audio_emotion, audio_score)
                        logger.info(f"SHARED_STATE UPDATE: audio_emotion_smoothed set to ({unified_audio_emotion}, {audio_score:.2f}) (insufficient window)")
                else:
                    shared_state['audio_emotion_smoothed'] = ("unknown", 0.0)
                    logger.info(f"SHARED_STATE UPDATE: audio_emotion_smoothed set to ('unknown', 0.0) due to invalid current audio emotion")

            except Exception as e:
                logger.error(f"Error in audio processing loop iteration: {e}")
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Fatal error in audio processing thread: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Audio processing thread stopped")


def record_audio(shared_state):
    """Record audio continuously and store in shared state"""
    global audio_buffer

    # Track audio stats
    audio_level_tracker = deque(maxlen=20)  # Track last 20 audio chunks
    last_stats_time = time.time()

    def audio_callback(indata, frames, time_, status):
        """Callback for sounddevice to continuously capture audio"""
        nonlocal audio_level_tracker, last_stats_time

        if status:
            logger.info(f"Audio recording status: {status}")

        # Add the new audio data to our buffer
        audio_data = indata[:, 0]  # Use first channel

        # Track audio levels for diagnostics
        audio_rms = np.sqrt(np.mean(audio_data**2))
        audio_level_tracker.append(audio_rms)

        # Periodic diagnostics about audio levels
        current_time = time.time()
        if current_time - last_stats_time > 5:  # Every 5 seconds
            avg_level = sum(audio_level_tracker) / len(audio_level_tracker) if audio_level_tracker else 0
            peak_level = max(audio_level_tracker) if audio_level_tracker else 0
            logger.info(f"Audio levels: current={audio_rms:.6f}, avg={avg_level:.6f}, peak={peak_level:.6f}, buffer_size={len(audio_buffer)}")

            # Detailed buffer diagnostics
            if audio_buffer:
                buffer_array = np.array(list(audio_buffer)) # Convert deque to list then numpy array for stats
                if buffer_array.size > 0: # Ensure buffer_array is not empty
                    buffer_min = np.min(buffer_array)
                    buffer_max = np.max(buffer_array)
                    buffer_std = np.std(buffer_array)
                    logger.info(f"Buffer stats: min={buffer_min:.6f}, max={buffer_max:.6f}, std={buffer_std:.6f}")
                else:
                    logger.info("Buffer stats: audio_buffer is currently empty for stats calculation.")

            if avg_level < 0.001: # This threshold is for warning, not for stopping transcription
                logger.warning("Audio levels are very low. Is your microphone working and unmuted?")
            last_stats_time = current_time

        # Add new audio data to buffer
        for sample in audio_data:
            audio_buffer.append(sample)

        # Store the latest audio for processing
        try:
            # Ensure audio_buffer is not empty before converting to numpy array
            if audio_buffer:
                audio_array = np.array(list(audio_buffer)) # Convert deque to list then numpy array
                shared_state['latest_audio'] = audio_array
            else:
                # If buffer is empty, ensure latest_audio reflects that
                shared_state['latest_audio'] = np.array([])

        except Exception as e:
            logger.error(f"Error updating latest audio: {e}")

        # Track silence periods
        if not hasattr(audio_callback, 'silence_tracker'):
            audio_callback.silence_tracker = 0

        # Check if this is essentially silence - be more aggressive with buffer clearing
        if audio_rms < 0.001:
            audio_callback.silence_tracker += 1
            # Reset buffer after 15 seconds of silence to avoid "I'm glad I found you" syndrome
            # Much more aggressive than before (15 seconds instead of 30)
            if audio_callback.silence_tracker > 150:  # ~15 seconds if called every 100ms
                logger.warning("Detected extended silence - resetting audio buffer")
                audio_buffer.clear()
                shared_state['latest_audio'] = np.array([])  # Clear latest audio as well
                audio_callback.silence_tracker = 0
        else:
            audio_callback.silence_tracker = 0  # Reset on non-silent audio

        # Add a timestamp to track when we last got audio data
        if not hasattr(audio_callback, 'last_audio_time'):
            audio_callback.last_audio_time = time.time()
        else:
            audio_callback.last_audio_time = time.time()

    try:
        # Set up the audio stream
        stream = sd.InputStream(
            callback=audio_callback,
            channels=1,
            samplerate=AUDIO_SAMPLE_RATE,
            blocksize=int(AUDIO_SAMPLE_RATE * 0.1)  # 100ms blocks
        )

        # Start recording
        with stream:
            logger.info(f"Audio recording started with sample rate {AUDIO_SAMPLE_RATE}Hz")

            # Keep recording until stop flag is set
            while True:
                stop_event_obj = shared_state.get('stop_event')
                if isinstance(stop_event_obj, dict):
                    if stop_event_obj.get('stop', False):
                        logger.info("Stop signal received in audio recording")
                        break
                elif stop_event_obj and stop_event_obj:
                    logger.info("Stop event detected in audio recording")
                    break

                # Sleep briefly to avoid burning CPU
                time.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in audio recording: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Audio recording stopped")