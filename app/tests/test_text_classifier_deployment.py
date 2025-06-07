# app/tests/test_text_classifier_deployment.py

import requests
import json
import time
import os
import sys

# Add the parent directory to the Python path to allow importing shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# Define the API URL for your deployed Text Emotion Classifier
TEXT_CLASSIFIER_API_URL = os.getenv("AFFECTLINK_SPEECHTEXT_API_URL", "https://localhost:54250/invocations") 

# A warning for clarity if the env var isn't set
if not os.getenv("AFFECTLINK_SPEECHTEXT_API_URL"):
    print("WARNING: AFFECTLINK_SPEECHTEXT_API_URL environment variable not set. Using default: https://localhost:54250/invocations")


def test_text_classifier_api(api_url):
    """
    Sends a test text to the deployed text emotion classifier API and prints the response.
    """
    test_text = "I am so happy today, this is fantastic news!"

    payload = {
        "dataframe_records": [
            {"text": test_text}
        ]
    }

    headers = {
        "Content-Type": "application/json",
    }

    print(f"Sending request to: {api_url}")
    print(f"Payload (text snippet): '{test_text[:50]}...'")

    try:
        start_time = time.time()
        # RE-ADDING verify=False for local HTTPS deployments with self-signed certificates.
        # This is necessary because your local AI Studio deployment uses self-signed certs.
        response = requests.post(api_url, headers=headers, json=payload, verify=False)
        response_time = time.time() - start_time

        print("\n--- Model Invocation Successful ---")
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Time: {response_time:.2f} seconds")

        if response.status_code == 200:
            response_json = response.json()
            print("Response JSON:")
            print(json.dumps(response_json, indent=2))

            predictions = response_json.get('predictions', [])
            if predictions and len(predictions) > 0:
                # The response JSON shows 'predictions' as a list containing a single dictionary
                # with the top emotion. We can directly access it.
                dominant_emotion_data = predictions[0] 
                
                # Ensure 'label' and 'score' keys exist
                if 'label' in dominant_emotion_data and 'score' in dominant_emotion_data:
                    dominant_emotion = dominant_emotion_data['label']
                    dominant_score = dominant_emotion_data['score']
                    print(f"\nDominant Emotion: {dominant_emotion} (Score: {dominant_score:.4f})")
                else:
                    print("Unexpected prediction format: 'label' or 'score' not found in the dominant emotion data.")
            else:
                print("No predictions found in the response.")
        else:
            print(f"Error: API call failed. Response: {response.text}")

    except requests.exceptions.ConnectionError as ce:
        print(f"Error: Connection error to {api_url}: {ce}")
    except requests.exceptions.Timeout as te:
        print(f"Error: Timeout connecting to {api_url}: {te}")
    except requests.exceptions.RequestException as re:
        print(f"Error: Request error to {api_url}: {re}")
    except json.JSONDecodeError as jde:
        print(f"Error: Could not decode JSON response: {jde}. Response text: {response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("Testing Text Emotion Classifier Deployment...")
    test_text_classifier_api(TEXT_CLASSIFIER_API_URL)