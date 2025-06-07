import requests
import json
import base64
import os
import time
import numpy as np
from dotenv import load_dotenv

# Load environment variables (if you have them)
load_dotenv()

# --- Configuration ---
# Update this to your SER model's deployed URL
# Make sure this is ONLY the base URL (e.g., "https://localhost:55853" or "https://your-server:port")
MLFLOW_SERVING_URL = os.getenv("AFFECTLINK_SER_API_URL", "https://localhost:55853") # REMOVED '/invocations' from default
ENDPOINT = "/invocations" # This is the standard MLflow prediction endpoint
FULL_URL = f"{MLFLOW_SERVING_URL}"

# Path to your local sample audio file that you want to send
# This is the path on YOUR LOCAL MACHINE where you run this test script.
# Assumes 'data' folder is in the same directory as this script, or adapt as needed.
LOCAL_SAMPLE_AUDIO_PATH = "data/sample_audio.wav" 

# --- Main Execution ---
if __name__ == "__main__":
    # Check if the audio file exists
    if not os.path.exists(LOCAL_SAMPLE_AUDIO_PATH):
        print(f"Error: Audio file not found at {LOCAL_SAMPLE_AUDIO_PATH}")
        print("Please ensure 'sample_audio.wav' is in the 'data/' directory relative to your AffectLink root.")
        exit(1)

    # Read audio file and encode to base64
    print(f"Using local audio file path: {LOCAL_SAMPLE_AUDIO_PATH}")
    try:
        with open(LOCAL_SAMPLE_AUDIO_PATH, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_base64_string = base64.b64encode(audio_bytes).decode('utf-8')
        print("Audio file successfully read and base64 encoded.")
    except Exception as e:
        print(f"Error reading or encoding audio file: {e}")
        exit(1)

    # Construct payload
    payload = {
        "dataframe_records": [
            {
                "audio_base64": audio_base64_string
            }
        ]
    }

    print(f"Sending request to: {FULL_URL}")
    base64_snippet = audio_base64_string[:50] + "..." + audio_base64_string[-10:]
    print(f"Payload (audio_base64 snippet): {base64_snippet}")

    headers = {
        "Content-Type": "application/json",
    }

    try:
        start_time = time.time()
        response = requests.post(FULL_URL, headers=headers, data=json.dumps(payload), verify=False)
        end_time = time.time()

        print("\n--- Model Invocation Successful ---")
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        response_json = response.json()
        
        # --- NEW: Print the full response JSON ---
        print("\nFull Response JSON:")
        print(json.dumps(response_json, indent=2))

        if response.status_code == 200:
            if "predictions" in response_json and len(response_json["predictions"]) > 0:
                prediction_data = response_json["predictions"]
                
                if isinstance(prediction_data, list) and len(prediction_data) > 0:
                    first_prediction = prediction_data[0] # Assuming one audio sample per request
                    if isinstance(first_prediction, dict):
                        dominant_emotion = first_prediction.get("dominant_audio_emotion")
                        dominant_score = first_prediction.get("dominant_audio_emotion_score")
                        full_scores = first_prediction.get("full_audio_emotion_scores")
                        raw_logits = first_prediction.get("raw_logits") # NEW
                        softmax_probs = first_prediction.get("softmax_probabilities") # NEW

                        print(f"\n--- Detailed Prediction Results ---")
                        print(f"Dominant Audio Emotion: {dominant_emotion}")
                        print(f"Dominant Audio Emotion Score: {dominant_score:.4f}")
                        print(f"Full Audio Emotion Scores: {json.dumps(full_scores, indent=2)}")
                        
                        if raw_logits is not None:
                            print(f"Raw Logits: {raw_logits}") # Print as is, already a list
                        if softmax_probs is not None:
                            print(f"Softmax Probabilities: {softmax_probs}") # Print as is, already a list
                        
                    else:
                        print(f"Unexpected prediction format in first item: {first_prediction}")
                else:
                    print("Response does not contain valid predictions list or it is empty.")
            else:
                print("Response does not contain 'predictions' key or predictions are empty.")
        else:
            print(f"Error: Model invocation failed with status code {response.status_code}")
            if "error_code" in response_json:
                print(f"Error Code: {response_json['error_code']}")
            if "message" in response_json:
                print(f"Error Message: {response_json['message']}")

    except requests.exceptions.ConnectionError as e:
        print(f"\nError: Could not connect to the MLflow serving URL. Is the server running at {FULL_URL}?")
        print(f"Details: {e}")
    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred during the request: {e}")
    except json.JSONDecodeError as e:
        print(f"\nError decoding JSON response: {e}")
        print(f"Raw response content: {response.text}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")