# app/tests/test_text_classifier_deployment.py

import requests
import json
import time
import os
import sys

# Add the parent directory to the Python path to allow importing shared modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# Define the API URL for your deployed Text Emotion Classifier
TEXT_CLASSIFIER_API_URL = "https://localhost:63398/invocations" # Keep this for local testing

def test_text_classifier_api(api_url):
    """
    Sends a test text to the deployed text emotion classifier API and prints the response.
    """
    test_texts = [
        "I am so happy today, this is fantastic news!",
        "I feel terrible and sad about what happened yesterday.",
        "This is a very neutral statement.",
        "I'm extremely angry about the delay!"
    ]

    # --- MODIFIED PAYLOAD ---
    # Use 'dataframe_split' format which is robustly handled by MLflow servers
    # This structure directly maps to a Pandas DataFrame
    payload = {
        "dataframe_split": {
            "columns": ["text"], # Your DataFrame will have a column named 'text'
            "data": [[t] for t in test_texts] # Each inner list is a row; here, each row has one element
        }
    }
    # --- END MODIFIED PAYLOAD ---


    headers = {
        "Content-Type": "application/json",
    }

    print(f"Sending request to: {api_url}")
    print(f"Payload (first text snippet): '{test_texts[0][:50]}...'")

    try:
        start_time = time.time()
        # Ensure verify=False for self-signed certificates in local deployment
        response = requests.post(api_url, headers=headers, json=payload, verify=False) 
        response_time = time.time() - start_time

        print("\n--- Model Invocation Successful ---")
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Time: {response_time:.2f} seconds")

        if response.status_code == 200:
            response_json = response.json()
            print("Full Response JSON:")
            print(json.dumps(response_json, indent=2)) 

            predictions = response_json.get('predictions', [])
            if predictions:
                print("\n--- Detailed Prediction Results ---")
                for pred_idx, prediction_data in enumerate(predictions):
                    print(f"\n--- Result for Input {pred_idx + 1} ('{prediction_data.get('text_input', 'N/A')}') ---")
                    
                    dominant_emotion = prediction_data.get('dominant_emotion', 'N/A')
                    dominant_score = prediction_data.get('dominant_emotion_score', 0.0)
                    full_scores = prediction_data.get('full_emotion_scores', [])

                    print(f"Dominant Emotion: {dominant_emotion} (Score: {dominant_score:.4f})")
                    print("All Emotion Scores:")
                    if isinstance(full_scores, list):
                        for item in full_scores:
                            if isinstance(item, dict) and 'emotion' in item and 'score' in item:
                                print(f"  - {item['emotion']}: {item['score']:.4f}")
                            else:
                                print(f"  - Unexpected format: {item}")
                    else:
                        print(f"  Unexpected full_emotion_scores format: {full_scores}")

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