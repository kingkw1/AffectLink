import requests
import json
import base64
import os
import time
from dotenv import load_dotenv

# Load environment variables (if you have them)
load_dotenv()

# Configuration
MLFLOW_SERVING_URL = os.getenv("MLFLOW_SERVING_URL", "https://localhost:60049")
ENDPOINT = "/invocations"
FULL_URL = f"{MLFLOW_SERVING_URL}{ENDPOINT}"

# Path to your local sample audio file that you want to send
# This is the path on YOUR LOCAL MACHINE where you run this test script.
LOCAL_SAMPLE_AUDIO_PATH = "data/sample_audio.wav" 

# Check if the audio file exists
if not os.path.exists(LOCAL_SAMPLE_AUDIO_PATH):
    print(f"Error: Audio file not found at {LOCAL_SAMPLE_AUDIO_PATH}")
    print("Please ensure 'sample_audio.wav' is in the 'data/' directory relative to this script.")
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
# The key for the input data should match the 'name' in your input_schema
# which is now 'audio_base64'.
payload = {
    "inputs": {
        "audio_base64": [
            audio_base64_string
        ]
    }
}

print(f"Sending request to: {FULL_URL}")
print(f"Payload (first 100 chars of base64 for brevity): {str(payload)[:100]}...")

headers = {
    "Content-Type": "application/json",
}

try:
    start_time = time.time()
    # Use verify=False if you are using self-signed certificates (common for localhost HTTPS)
    response = requests.post(FULL_URL, headers=headers, data=json.dumps(payload), verify=False)
    end_time = time.time()

    print("\n--- Model Invocation Successful ---")
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Time: {end_time - start_time:.2f} seconds")
    print("Response JSON:")
    response_json = response.json()
    print(json.dumps(response_json, indent=2))

    if response.status_code == 200:
        if "predictions" in response_json and len(response_json["predictions"]) > 0:
            prediction = response_json["predictions"][0]
            if "transcription" in prediction:
                print(f"\nTranscription: {prediction['transcription']}")
            else:
                print("Response does not contain 'transcription' key in prediction.")
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