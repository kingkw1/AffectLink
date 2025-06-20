import os
import sys
import time
import mlflow
import cv2
import librosa
import numpy as np
from transformers import pipeline, AutoModelForAudioClassification, AutoFeatureExtractor
import logging

# Add the project root to sys.path to ensure local modules are found
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir, '..')
if project_root not in sys.path:
    sys.path.append(project_root)
DATA_DIR = os.path.join(project_root, "data")


# Add relevant paths if not already in sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
# If audio_emotion_processor.py is directly in 'app' and 'constants.py' is also in 'app'
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import local modules
from references.debug_scripts.test_audio_processor import process_audio_chunk_from_file
from constants import (
    FACIAL_TO_UNIFIED, SER_TO_UNIFIED, UNIFIED_EMOTIONS, TEXT_TO_UNIFIED,
    AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE,
    TEXT_CLASSIFIER_MODEL_ID, SER_MODEL_ID, SER_TO_UNIFIED, TEXT_TO_UNIFIED
)
from video_processor import get_facial_emotion_from_frame
from main_processor import create_unified_emotion_dict, calculate_average_multimodal_similarity, load_models
from audio_processor import logger

# Set up environment for DeepFace model caching
deepface_cache_dir = os.path.join(project_root, "models", "deepface_cache")
os.environ['DEEPFACE_HOME'] = deepface_cache_dir
os.makedirs(deepface_cache_dir, exist_ok=True)

# Configure logging for batch_analyzer.py
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress DeepFace logging for cleaner console output during analysis
logging.getLogger('deepface').setLevel(logging.ERROR)

# --- Model Loading (will be done once in process_media_file) ---
whisper_model = None
text_emotion_classifier = None
audio_feature_extractor = None
audio_emotion_classifier = None


def process_audio(input_file_path):
    logger.info(f"Processing audio from file: {input_file_path}")
    audio_results = []
    # Load entire audio file
    audio_data, current_audio_sample_rate = librosa.load(input_file_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
    
    total_samples = len(audio_data)
    samples_per_chunk = int(AUDIO_CHUNK_SIZE * current_audio_sample_rate)
    
    for i in range(0, total_samples, samples_per_chunk):
        audio_chunk = audio_data[i : i + samples_per_chunk]
        
        if len(audio_chunk) < samples_per_chunk and i + samples_per_chunk < total_samples:
            # Optional: Pad the last chunk if it's shorter and not the very end of the file
            # This might be useful if the model expects fixed-size inputs, but Whisper handles variable lengths.
            # padding_needed = samples_per_chunk - len(audio_chunk)
            # audio_chunk = np.pad(audio_chunk, (0, padding_needed), mode='constant')
            pass # Whisper handles variable length, so padding might not be strictly necessary
        elif len(audio_chunk) == 0:
            continue

        # Call the updated function that now returns 5 values
        transcribed_text, text_emotion_data, audio_emotion_data, text_full_scores, audio_full_scores = \
            process_audio_chunk_from_file(
                audio_chunk, current_audio_sample_rate, 
                whisper_model, text_emotion_classifier, 
                audio_feature_extractor, audio_emotion_classifier
            )

        start_time_chunk_sec = i / current_audio_sample_rate
        end_time_chunk_sec = (i + len(audio_chunk)) / current_audio_sample_rate

        audio_results.append({
            'start_time_sec': start_time_chunk_sec,
            'end_time_sec': end_time_chunk_sec,
            'transcribed_text': transcribed_text,
            'text_emotion': text_emotion_data,
            'audio_emotion': audio_emotion_data,
            'text_emotion_full_scores': text_full_scores, # Now correctly populated
            'audio_emotion_full_scores': audio_full_scores  # Now correctly populated
        })
    
    logger.info(f"Finished processing audio file: {input_file_path}. Collected {len(audio_results)} audio segments.")
    return audio_results

def process_video(input_file_path, frame_processing_rate=1):
    logger.info(f"Processing video from file: {input_file_path} with frame_processing_rate={frame_processing_rate}")
    video_results = []
    cap = cv2.VideoCapture(input_file_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open video file {input_file_path}")
        return video_results

    if frame_processing_rate <= 0:
        logger.warning("frame_processing_rate must be a positive integer. Defaulting to 1.")
        frame_processing_rate = 1

    frame_idx = 0
    processed_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if frame_idx % frame_processing_rate == 0:
            current_facial_emotion = ("unknown", 0.0)
            current_facial_scores = {}
            
            try:                  
                facial_emotion_data, raw_emotion_scores = get_facial_emotion_from_frame(frame)

                if facial_emotion_data and facial_emotion_data[0] != "unknown" and facial_emotion_data[0] != "error":
                    current_facial_emotion = facial_emotion_data
                    current_facial_scores = raw_emotion_scores
                    logger.info(f"[{timestamp_sec:.2f}s] Frame {frame_idx} (Processed): Facial emotion: {facial_emotion_data[0]} ({facial_emotion_data[1]:.2f})")
                elif facial_emotion_data: 
                    current_facial_emotion = facial_emotion_data
                    current_facial_scores = raw_emotion_scores if raw_emotion_scores else {}
                # If facial_emotion_data is None, defaults remain ("unknown", 0.0) and {}

            except Exception as e:
                logger.debug(f"Error in facial analysis for frame {frame_idx} at {timestamp_sec:.2f}s: {e}")
                # Defaults current_facial_emotion = ("unknown", 0.0), current_facial_scores = {} are kept

            video_results.append({
                'timestamp_sec': timestamp_sec,
                'facial_emotion': current_facial_emotion,
                'facial_emotion_full_scores': current_facial_scores
            })
            processed_frame_count += 1
        
        frame_idx += 1
    
    cap.release()
    logger.info(f"Finished processing video file: {input_file_path}. Read {frame_idx} total frames, processed {processed_frame_count} frames for emotion.")
    return video_results


def _convert_scores_to_unified_vector(scores_dict, unified_emotions_list=UNIFIED_EMOTIONS):
    """
    Converts a dictionary of emotion scores to a consistent vector (list of floats)
    based on a predefined order of unified emotions. Missing emotions get a score of 0.0.
    """
    vector = [0.0] * len(unified_emotions_list)
    if not scores_dict:
        return vector

    # DeepFace returns np.float32, ensure conversion to float for consistency
    for i, emotion in enumerate(unified_emotions_list):
        # Ensure we only use emotions present in the scores_dict and handle np.float32
        if emotion in scores_dict:
            score = scores_dict[emotion]
            # Convert numpy types to native Python floats if necessary
            if isinstance(score, np.float32):
                vector[i] = float(score)
            else:
                vector[i] = score
    return vector

def prepare_data_for_similarity_calculation(video_results, audio_results):
    """
    Aligns video and audio/text emotion results by time and prepares
    them into vectors suitable for cosine similarity calculation.
    
    Args:
        video_results (list): List of dictionaries from process_video.
        audio_results (list): List of dictionaries from process_audio.
        
    Returns:
        list: A list of dictionaries, where each dictionary contains:
              'timestamp_start_sec': start time of the chunk
              'timestamp_end_sec': end time of the chunk
              'facial_vector': emotion vector for facial data in this chunk
              'audio_vector': emotion vector for audio data in this chunk
              'text_vector': emotion vector for text data in this chunk
              (and potentially other relevant data like transcribed_text)
    """
    aligned_data = []

    for audio_chunk_result in audio_results:
        audio_chunk_start_time = audio_chunk_result['start_time_sec']
        audio_chunk_end_time = audio_chunk_result['end_time_sec']

        # 1. Aggregate facial emotion scores for frames within this audio chunk
        facial_scores_in_chunk = []
        for frame_result in video_results:
            frame_timestamp = frame_result['timestamp_sec']
            if audio_chunk_start_time <= frame_timestamp < audio_chunk_end_time:
                # Only add if full scores are available and not empty
                if frame_result.get('facial_emotion_full_scores'):
                    facial_scores_in_chunk.append(frame_result['facial_emotion_full_scores'])
        
        # Calculate average facial emotion scores for the chunk
        # Initialize a dictionary for sum of scores for each unified emotion
        avg_facial_scores = {emotion: 0.0 for emotion in UNIFIED_EMOTIONS}
        
        if facial_scores_in_chunk:
            for scores_dict in facial_scores_in_chunk:
                for emotion, score in scores_dict.items():
                    unified_emotion = FACIAL_TO_UNIFIED.get(emotion)
                    if unified_emotion:
                        # Sum scores, ensuring numpy types are converted to float
                        avg_facial_scores[unified_emotion] += float(score)
            
            # Divide by number of frames to get average
            num_frames = len(facial_scores_in_chunk)
            for emotion in avg_facial_scores:
                avg_facial_scores[emotion] /= num_frames
        
        # 2. Convert scores to unified vectors
        unified_facial_scores = create_unified_emotion_dict(avg_facial_scores, FACIAL_TO_UNIFIED)
        unified_audio_scores = create_unified_emotion_dict(audio_chunk_result.get('audio_emotion_full_scores', {}), SER_TO_UNIFIED)
        unified_text_scores = create_unified_emotion_dict(audio_chunk_result.get('text_emotion_full_scores', {}), TEXT_TO_UNIFIED)

        # 3. Convert aggregated scores and chunk-based scores into unified vectors
        facial_vector = _convert_scores_to_unified_vector(unified_facial_scores, UNIFIED_EMOTIONS)
        audio_vector = _convert_scores_to_unified_vector(unified_audio_scores, UNIFIED_EMOTIONS)
        text_vector = _convert_scores_to_unified_vector(unified_text_scores, UNIFIED_EMOTIONS)

        aligned_data.append({
            'timestamp_start_sec': audio_chunk_start_time,
            'timestamp_end_sec': audio_chunk_end_time,
            'facial_vector': facial_vector,
            'audio_vector': audio_vector,
            'text_vector': text_vector,
            'transcribed_text': audio_chunk_result['transcribed_text']
        })
    
    return aligned_data


def process_media_file(input_file_path, frame_processing_rate=1):
    """
    Processes a video or audio file for emotion analysis and prints results.
    """
    file_extension = os.path.splitext(input_file_path)[1].lower()
    
    # load_models() # Ensure models are loaded
    
    video_results = []
    audio_results = []

    if file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        video_results = process_video(input_file_path, frame_processing_rate=frame_processing_rate)

        # Attempt to load audio from the video file to check if it exists
        # We don't need to store the data here if process_audio will load it again,
        # but it's a quick check. Alternatively, process_audio could take raw data.
        temp_audio_data, _ = librosa.load(input_file_path, sr=AUDIO_SAMPLE_RATE, mono=True, dtype=np.float32)
        if len(temp_audio_data) > 0:
            logger.info(f"Video file {input_file_path} contains audio. Processing audio component.")
            audio_results = process_audio(input_file_path) # Process the same file for audio
        else:
            logger.info(f"Video file {input_file_path} does not contain a significant audio component.")

    elif file_extension in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        audio_results = process_audio(input_file_path)

    else:
        logger.error(f"Unsupported file type: {file_extension} for {input_file_path}")
        return [], []

    logger.info(f"Media processing complete for {input_file_path}.")
    if video_results:
        logger.info(f"Collected {len(video_results)} video analysis results.")
    if audio_results:
        logger.info(f"Collected {len(audio_results)} audio analysis results.")
    
    # For now, just returning them. Further processing (like consistency analysis) would happen here or be passed on.
    return video_results, audio_results

def run_batch_analysis(media_file_path, frame_processing_rate=1):
    """
    Runs multimodal emotion analysis on a given media file (video/audio).
    Logs parameters and metrics to MLflow.
    """
    logger.info(f"Starting batch analysis for {media_file_path}")
    
    # Initialize MLflow run
    # The run name will be clearly visible in the MLflow UI
    with mlflow.start_run(run_name=f"Batch Analysis - {os.path.basename(media_file_path)}") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        start_time = time.time()
        
        if not os.path.exists(media_file_path):
            logger.error(f"Error: Input media file '{media_file_path}' does not exist. Please update the path.")
            return
        
        # --- Log Parameters ---
        mlflow.log_param("input_file_path", media_file_path)
        mlflow.log_param("frame_processing_rate", frame_processing_rate)
        mlflow.log_param("deepface_cache_dir", deepface_cache_dir)
        mlflow.log_param("text_classifier_model_id", TEXT_CLASSIFIER_MODEL_ID)
        mlflow.log_param("ser_model_id", SER_MODEL_ID)
        logger.info("Logged run parameters to MLflow.")

        # Ensure models are loaded before starting the analysis
        global whisper_model, text_emotion_classifier, audio_feature_extractor, audio_emotion_classifier
        whisper_model, text_emotion_classifier, audio_feature_extractor, audio_emotion_classifier, device = load_models()

        logger.info(f"Starting batch analysis for file: {media_file_path} with frame_processing_rate={frame_processing_rate}")
        
        video_results, audio_results = process_media_file(media_file_path, frame_processing_rate=frame_processing_rate)

        # --- Data Preparation for Similarity Calculation ---
        if video_results and audio_results:
            aligned_for_similarity = prepare_data_for_similarity_calculation(video_results, audio_results)
            logger.info(f"Prepared {len(aligned_for_similarity)} aligned data chunks for similarity calculation.")
            
            # Calculate average similarity across all chunks
            total_similarity = 0.0
            total_chunks_with_similarity = 0
            
            # Example of how you would then use this data for calculation
            for chunk in aligned_for_similarity:
                avg_similarity = calculate_average_multimodal_similarity(
                    chunk['facial_vector'],
                    chunk['audio_vector'],
                    chunk['text_vector']
                )
                # Add to running total for average calculation
                if avg_similarity is not None:
                    total_similarity += avg_similarity
                    total_chunks_with_similarity += 1
                    
                logger.info(f"Chunk [{chunk['timestamp_start_sec']:.2f}s - {chunk['timestamp_end_sec']:.2f}s]: Average Similarity = {avg_similarity:.2f}")
                logger.info(f"  Facial Vector: {chunk['facial_vector']}")
                logger.info(f"  Audio Vector: {chunk['audio_vector']}")
                logger.info(f"  Text Vector: {chunk['text_vector']}")
                logger.info(f"  Transcribed Text: {chunk['transcribed_text']}")
            
            # Calculate and log average cosine similarity if we have valid data
            if total_chunks_with_similarity > 0:
                average_cosine_similarity = total_similarity / total_chunks_with_similarity
                mlflow.log_metric("average_cosine_similarity", average_cosine_similarity)
                logger.info(f"Logged average_cosine_similarity: {average_cosine_similarity:.4f}")
            else:
                logger.warning("No valid cosine similarities to average. Skipping average_cosine_similarity metric.")

        elif video_results:
            logger.info("Only video results available. Cannot perform multimodal consistency calculation.")
        elif audio_results:
            logger.info("Only audio results available. Cannot perform multimodal consistency calculation.")
        else:
            logger.warning("No video or audio results found for consistency calculation.")

        # For now, we just log that the data is available.
        if video_results:
            logger.info(f"Main: Received {len(video_results)} video results for further analysis.")
        if audio_results:
            logger.info(f"Main: Received {len(audio_results)} audio results for further analysis.")

        end_time = time.time()
        duration = end_time - start_time
        mlflow.log_metric("processing_duration_seconds", duration)
        logger.info(f"Batch analysis completed in {duration:.2f} seconds.")
        logger.info(f"MLflow run logged. View with: 'mlflow ui'")

def main(video_file_path, frame_processing_rate=1): # Renamed parameter for clarity
    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name="Batch_Multimodal_Analysis")
    
    # Call the run_batch_analysis function that handles the actual processing and MLflow logging
    run_batch_analysis(video_file_path, frame_processing_rate)

if __name__ == '__main__':
    media_file_path = os.path.join(DATA_DIR, "sample_video.mp4")
    desired_frame_processing_rate = 20 # Process every 5th frame

    main(media_file_path, frame_processing_rate=desired_frame_processing_rate)