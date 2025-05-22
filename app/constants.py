UNIFIED_EMOTIONS = ['neutral', 'happy', 'sad', 'angry']
TEXT_TO_UNIFIED = {
    'neutral': 'neutral',
    'joy': 'happy',
    'sadness': 'sad',
    'anger': 'angry',
    'fear': None,
    'surprise': None,
    'disgust': None
}
SER_TO_UNIFIED = {
    'neu': 'neutral',
    'hap': 'happy',
    'sad': 'sad',
    'ang': 'angry'
}
FACIAL_TO_UNIFIED = {
    'neutral': 'neutral',
    'happy': 'happy',
    'sad': 'sad',
    'angry': 'angry',
    'fear': None,
    'surprise': None,
    'disgust': None
}
# Constants
VIDEO_WINDOW_DURATION = 5  # seconds
AUDIO_WINDOW_DURATION = 5  # seconds
CAMERA_INDEX = 0  # Default camera index, can be overridden
# Output verbosity control
VERBOSE_OUTPUT = False
# Model IDs
TEXT_CLASSIFIER_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"
SER_MODEL_ID = "superb/hubert-large-superb-er"