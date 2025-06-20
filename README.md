# AffectLink: Multimodal Emotion Consistency Tracking for Telehealth AffectLink: Multimodal Emotion Consistency Tracking for Telehealth

<p align="center">
  <img src="assets/thumbnail.png" alt="AffectLink Thumbnail" width="800"/>
</p>

**AffectLink** is an AI-powered system designed to bridge the empathy gap in tele-healthcare. It provides real-time, local-first multimodal emotional analysis of patients during virtual sessions by comparing facial expressions, vocal tone, and spoken content. Its core innovation, the **Emotional Consistency Index (ECI)**, highlights discrepancies between verbal and non-verbal cues, offering clinicians deeper insights into unexpressed emotions, fatigue, or stress. All processing occurs securely and locally on the HP workstation, ensuring patient privacy and HIPAA compliance.

## 🔍 What It Does

  - 🎙️ **Real-time Multimodal Analysis:** Processes live audio and video streams for facial expressions, speech emotion, and text emotion.
  - 🧩 **Emotional Consistency Index (ECI):** Synthesizes emotional streams to identify inconsistencies between spoken words and non-verbal cues.
  - 📈 **Privacy-Preserving:** All AI inference and data processing occur locally on the HP workstation, ensuring patient data never leaves the device.
  - 🧑‍⚕️ **Clinician-Focused Dashboard:** Provides an intuitive Streamlit UI for real-time visualization and insights.

## 🚀 Features

  - Multimodal emotional analysis (audio, video, text).
  - Real-time and local-first data processing.
  - Seamless integration with HP AI Studio for model deployment and tracking.
  - Secure, on-device inference without cloud dependency for sensitive data.

## 🛠️ Built With

  - **Python**
  - **HP AI Studio:** For local model deployment (MLflow & Swagger API) and experiment tracking.
  - **DeepFace:** For local facial emotion analysis.
  - **Whisper (OpenAI):** For Speech-to-Text (ASR).
  - **Hugging Face Transformers:** For Speech Emotion Recognition (SER) and Text Emotion Classification.
  - **Streamlit:** For the interactive web application user interface.
  - **OpenCV, Sounddevice, SciPy, NumPy, Librosa:** For media processing.

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## 🚀 Getting Started (For Judges & Developers)

This section provides a step-by-step guide to get AffectLink up and running, including deploying the necessary AI models via HP AI Studio.

### Prerequisites

Before you begin, ensure you have the following:

  * **HP AI Studio:** Installed and configured on your workstation with an NVIDIA GPU. AffectLink heavily leverages HP AI Studio's local model deployment capabilities.
  * **Git:** For cloning the repository.
  * **Python 3.10:** Recommended version.
  * **Webcam & Microphone:** Required for live audio and video input.
  * **Internet Connection:** Needed initially to download models (DeepFace, Whisper, Hugging Face models) the first time they are used or registered.

### Step-by-Step Setup & Running AffectLink

Follow these instructions meticulously to set up and run AffectLink locally, demonstrating its full capabilities as intended for judging.

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/kingkw1/AffectLink.git
    cd AffectLink
    ```

2.  **Set Up and Install Python Dependencies for the Local Application:**
    It's highly recommended to use a Python virtual environment to manage dependencies. This ensures that AffectLink's libraries don't conflict with other Python projects on your system.

      * **A. Create a Virtual Environment (Recommended Python 3.10):**
        It's highly recommended to use Python 3.10 for this project.

          * **For Windows Users:**
            If `python3.10` is not directly recognized as a command, you have a few options:
              * **Option 1 (Recommended for most Windows users):** Use the `py` launcher if Python is installed via the official installer and `py` is in your PATH.
                ```bash
                py -3.10 -m venv .venv
                ```
              * **Option 2 (If `py` launcher is not available):** Use the generic `python` command, then verify its version in the next step.
                ```bash
                python -m venv .venv
                ```
              * **Option 3 (Direct Path - if previous options fail):** Specify the full path to your Python 3.10 executable. Replace `YOUR_USERNAME` with your actual Windows username.
                ```bash
                "C:\Users\YOUR_USERNAME\AppData\Local\Programs\Python\Python310\python.exe" -m venv .venv
                ```
          * **For Linux/macOS Users:**
            ```bash
            python3.10 -m venv .venv
            # If 'python3.10' is not found, try 'python3 -m venv .venv' or 'python -m venv .venv'
            ```

      * **B. Activate the Virtual Environment:**

          * **For Windows (Command Prompt):**
            ```bash
            .venv\Scripts\activate
            ```
          * **For Windows (PowerShell):**
            ```powershell
            .venv\Scripts\Activate.ps1
            ```
          * **For Linux/macOS (Bash/Zsh):**
            ```bash
            source .venv/bin/activate
            ```

      * **C. Verify Python Version (Crucial Step):**
        After activating your virtual environment, it is **critical** to confirm that the correct Python version (ideally 3.10.x) is active before installing dependencies.

        ```bash
        python --version
        ```

        *Ensure this command outputs `Python 3.10.x` or similar.* If it does not, deactivate the venv (e.g., by closing and reopening your terminal or typing `deactivate`) and retry creating it using a different method from Step 2A that correctly points to Python 3.10.

      * **D. Install Required Python Packages:**
        Before installing, **please edit your `requirements.txt` file.** 

        With your virtual environment activated, install the remaining necessary libraries from the *modified* `requirements.txt`:

        ```bash
        pip install -r requirements.txt
        ```

        **Then, install PyTorch and Torchaudio separately, based on your system's GPU (CUDA) setup:**

          * **Option 1: For systems with NVIDIA GPU and CUDA 12.x installed (most common for HP AI Studio users):**
            To find your exact CUDA version, on Windows, check the NVIDIA Control Panel or run `nvidia-smi` in PowerShell/CMD. On Linux, run `nvcc --version`. Replace `cu12x` with your specific CUDA version (e.g., `cu121` for CUDA 12.1, `cu122` for CUDA 12.2, etc.). For CUDA 12.8, use `cu128`.
            ```bash
            pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
            # IMPORTANT: Make sure to replace 'cu128' with your actual CUDA version if different.
            # Example for CUDA 12.1: pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu121
            ```
          * **Option 2: For systems without an NVIDIA GPU, or if you prefer a CPU-only installation:**
            ```bash
            pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
            ```

        *Note: The `torch` and `torchaudio` versions (`2.7.0` in these commands) should match the base version from your original `requirements.txt`.*

      * **E. Verify GPU/CUDA Setup (Highly Recommended):**
        After installing all packages, it's crucial to confirm that PyTorch and TensorFlow (used by DeepFace) are correctly configured to use your NVIDIA GPU with CUDA. Open your Python interpreter within the activated virtual environment and run the following checks:

          * **For PyTorch:**

            ```python
            import torch
            print(f"PyTorch CUDA available: {torch.cuda.is_is_available()}")
            if torch.cuda.is_available():
                print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
                print(f"PyTorch current CUDA device: {torch.cuda.current_device()}")
                print(f"PyTorch device name: {torch.cuda.get_device_name(0)}")
            ```

            *Expected output for success:* `PyTorch CUDA available: True` and details about your GPU.

          * **For TensorFlow (used by DeepFace):**

            ```python
            import tensorflow as tf
            print(f"TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
            ```

            *Expected output for success:* A list containing your GPU device (e.g., `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`).

          * **Troubleshooting:**

              * **PyTorch CUDA Issues:** If PyTorch CUDA is not available:

                  * Ensure you used the correct `cu12x` version in the `pip install torch` command (Step 2D Option 1).
                  * Verify that the NVIDIA CUDA Toolkit and CuDNN are correctly installed and configured on your system, and that their versions are compatible with the PyTorch version you're trying to use.
                  * Check NVIDIA driver paths and ensure they are correctly added to your system's PATH.
                  * Consult PyTorch's official installation guides for more detailed troubleshooting.

              * **TensorFlow GPU Issues (TensorFlow GPU available: []):**
                This is a common challenge, even with CUDA installed. If TensorFlow is not detecting your GPU, it means the installed TensorFlow package is likely not configured for GPU, or there's a version mismatch with your CUDA/CuDNN.

                **Recommendation for Hackathon (Prioritize Functionality):**
                For the hackathon, if you cannot quickly resolve the GPU detection for TensorFlow, it is often best to ensure the project runs by forcing TensorFlow to use the CPU.

                1.  **Uninstall** your current `tensorflow` and `tf-keras` packages:
                    ```bash
                    pip uninstall tensorflow tf-keras -y
                    ```
                2.  **Install** the CPU-only versions:
                    ```bash
                    pip install tensorflow-cpu==2.19.0 tf-keras==2.19.0
                    ```
                    *(Note: Ensure the versions here match the base versions from your original `requirements.txt` to maintain overall compatibility with other packages.)*
                3.  **Re-run** the `tf.config.list_physical_devices('GPU')` check. It should still show `[]`, but now you're certain it's using the CPU version, which will work.

                **Advanced GPU Troubleshooting (If time allows):** If you wish to troubleshoot TensorFlow's GPU further, ensure your `tensorflow` and `tf_keras` packages are designed for GPU (e.g., by checking their PyPI pages for specific GPU wheels). Sometimes, specific environment variables or different wheel files are needed. Since PyTorch is working, your CUDA installation is likely fine; the issue is specific to TensorFlow's integration. Refer to the official TensorFlow GPU installation guide for your specific version.

              * **General Installation Issues:** If other issues persist, ensure you have stable internet connection for downloads.

3.  **Download Pre-trained Models (First Run):**
    AI Studio will need internet access to download some dependencies, and to download the pre-trained DeepFace, Whisper, and Hugging Face models. Ensure you have an internet connection for this step. These models will be cached locally.

4.  **Set Up and Install Dependencies for HP AI Studio:**

      * **A. Setup Project**

          * **1. Project Details**
              * Enter a name and (optionally) a brief description of the project
              * Git Repository URL: `https://github.com/kingkw1/AffectLink.git`
              * Git Local Folder: Select the git folder where you cloned the repo in the previous step.
          * **2. Continue without "Connecting Your Data"**
          * **3. Setup Workspace**
              * Create a new workspace. Select `Deeplearning GPU` from the images catalog.
              * Choose a workspace name and setup with GPU configuration.
              * Select `Add Custom Libraries` and select "hp_ai_studio_requirements.txt" from inside of "AffectLink/deployment".
              * Select `Validate` and wait for confirmation that validation is complete.
              * Create Workspace.

      * **B. Install System-Level Dependencies (Linux):**
        In addition to the Python packages, AffectLink requires certain system-level libraries. If you are running a Debian/Ubuntu-based Linux system (like within HP AI Studio's workspace), you can install these with:

        ```bash
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx portaudio19-dev ffmpeg
        ```

        These install dependencies for OpenCV (facial analysis), SoundDevice (audio input), and FFmpeg (audio/video processing).

5.  **Deploy AI Models via HP AI Studio (CRITICAL FOR JUDGING):**
    AffectLink's core AI models (Whisper for ASR, Hugging Face model for SER, and Hugging Face model for Text Emotion) are designed to be deployed as local API endpoints using HP AI Studio's Swagger functionality. This demonstrates our "local-first" and secure inference approach.

      * **A. Register Models to MLflow:**

          * Start the workspace you just created and open a terminal from inside the JupyterLab interface (under the "Other" section).
          * Use the following commands to activate the correct environment and register the models:
            ```bash
            conda activate aistudio
            cd AffectLink/deployment/registration
            python register_ser_classifier.py
            python register_text_classifier.py
            python register_whisper.py
            ```
          * These scripts are designed to:
              * Trigger the necessary model downloads (if not already cached).
              * Initialize MLflow tracking.
              * Register the Whisper, SER, and Text Emotion models to your local MLflow instance within HP AI Studio. You should see entries for these models in the "Models" section of the HP AI Studio UI.
          * **Confirm successful model registration:**
              * In the navigation links under the project title, select `Monitor`
              * Within the `Experiments` tab, you should see `AffectLink_Model_Registration`.
              * Within the `Models` tab, you should see the 3 models that we registered.

      * **B. Deploy Models to Swagger API Endpoints:**

          * In the HP AI Studio UI, go to the "Deployments" tab.
          * Click "New Service."
          * For each of the registered models (Whisper, SER, Text Emotion), create a new deployment:
              * `Service name`: Assign a unique and descriptive name for each deployment (e.g., `whisper-api`, `ser-model`, `text-classifier-endpoint`).
              * `Select your model`: Select the respective model name and its latest version from the dropdowns.
              * `Choose model version`: Select the model version. A new version is created every time the registration script is run, so use the latest successful deployment (`1` if registration completed without any problems the first time).
              * `GPU configuration`: Choose `With GPU` if applicable for better performance.
              * `Choose workspace`: Select the name of the workspace you just created.
              * Click `Deploy`.
          * Once deployed, navigate back to the "Deployments" tab. For each deployment, click the "Play" button to spin up the local API endpoint.
          * **Obtain Endpoint URLs:** After each model is running, click on the **link icon** next to its deployment name. This will open a new browser tab with the Swagger UI for that specific model's endpoint. **Copy the full URL from your browser's address bar for each model.**

6.  **Configure AffectLink with Deployed Model Endpoints (CRITICAL):**
    Since your Streamlit application (`dashboard.py`) runs as a separate process and connects to these dynamically deployed models, you need to provide their dynamically assigned API endpoint URLs via environment variables.

      * **Set Environment Variables:** Before running `run_affectlink.py`, set the following environment variables with the actual URLs you obtained from HP AI Studio's Swagger UI for each respective model.
          * **For Windows (PowerShell):**
            ```powershell
            $env:AFFECTLINK_WHISPER_API_URL="https://localhost:XXXXX/invocations"
            $env:AFFECTLINK_SPEECHTEXT_API_URL="https://localhost:XXXXX/invocations"
            $env:AFFECTLINK_SER_API_URL="https://localhost:XXXXX/invocations"
            # Replace XXXXX with the respective dynamic port numbers.
            ```
          * **For Linux/macOS (Bash/Zsh):**
            ```bash
            export AFFECTLINK_WHISPER_API_URL="https://localhost:XXXXX/invocations"
            export AFFECTLINK_SPEECHTEXT_API_URL="https://localhost:XXXXX/invocations"
            export AFFECTLINK_SER_API_URL="https://localhost:XXXXX/invocations"
            # Replace XXXXX with the respective dynamic port numbers
            ```
          * *Remember to open a new terminal or PowerShell window for these variables to take effect if you set them permanently.*

7.  **Run the AffectLink Application ("Run All" for Judging):**
    Now that your models are deployed via HP AI Studio and your application is configured with their endpoints, you can launch AffectLink from within your machine's local terminal (not AI Studio).

    ```bash
    python .\app\run_affectlink.py
    ```

    This script will:

      * Start the `main_processor.py` for audio and video capture, and emotion analysis.
      * Launch the Streamlit web application, which will automatically open in your default browser.
      * The Streamlit UI will then connect to your locally deployed AI Studio model endpoints for Whisper, SER, and Text Emotion (using the URLs from your environment variables), and use the local DeepFace library for facial analysis.

    *Troubleshooting:* If the Streamlit application doesn't open automatically, look for a message in your terminal indicating the local URL (e.g., `http://localhost:8501`).

### Expected Behavior

  * The Streamlit application will display live video feedback.
  * Emotion metrics (Facial, Audio, Text) will update in real-time.
  * The Emotional Consistency Index (ECI) will be displayed.
  * Transcribed text will appear as you speak.
  * All data processing and AI inference, including calls to your HP AI Studio deployed models, will occur locally on your workstation.

## 📚 Project Architecture & Models Used

### System Architecture Diagram

```mermaid
graph TD
    A["User / Webcam / Mic 🎤📹"] -->|Raw Frames/Audio Stream| C["main_processor.py (Orchestrator)"]

    subgraph "Local Workstation / HP AI Studio Environment 🔒"
        C -->|Video Frames| E["Video Emotion Processor (DeepFace) 🖼️"]
        E -->|Facial Emotions| C

        C -->|Audio Chunks| D["Audio Pre-processor 🎵"]
        
        %% Consolidated Swagger API for all deployed models
        D -->|Audio for ASR/SER & Transcribed Text for Text Emo| G["Swagger API (Models: Whisper, SER, Text Emo deployed via HP AI Studio) 🚀"]
        G -->|Transcriptions, Audio Emotions, Text Emotions| D

        D -->|Processed Modality Data| C

        C -->|Aggregated Results & Consistency Index| TempFiles["Temp Files (affectlink_emotion.json, affectlink_frame.jpg) 📄"]
        TempFiles -->|Read for Display| B["AffectLink Streamlit UI 🖥️"]

        C -->|Log Metrics & Artifacts| F["MLflow (Tracking & Models) 📊"]
    end
    
    style A fill:#d9e8fb,stroke:#333,stroke-width:2px,color:#1a1a1a
    style B fill:#d0f0c0,stroke:#333,color:#1a1a1a
    style C fill:#fef3bd,stroke:#333,color:#1a1a1a
    style D fill:#e6e6fa,stroke:#333,color:#1a1a1a
    style E fill:#e6e6fa,stroke:#333,color:#1a1a1a
    style F fill:#cceeff,stroke:#333,color:#1a1a1a
    style G fill:#ffcccc,stroke:#333,color:#1a1a1a
    style TempFiles fill:#f8f8f8,stroke:#999,stroke-dasharray: 5 5,color:#1a1a1a
```

### Explanation of Models and Methods

  * **DeepFace (Facial Emotion Analysis):**
      * **Method:** This library is used for real-time facial expression detection and classification. It leverages pre-trained convolutional neural networks to identify emotions from video frames.
      * **Deployment:** Currently runs directly on the local machine via its Python library due to integration complexities encountered with HP AI Studio's model deployment for this specific library within the hackathon timeframe. It processes frames streamed from the webcam.
  * **Whisper (ASR - Automatic Speech Recognition):**
      * **Method:** OpenAI's robust Whisper model is used for high-accuracy speech-to-text transcription. It processes audio chunks to convert spoken language into text.
      * **Deployment:** Registered with MLflow and deployed as a local API endpoint via HP AI Studio's Swagger functionality. The `main_processor.py` sends audio data to this locally hosted endpoint for inference.
  * **Speech Emotion Recognition (SER) Model (Hugging Face Transformers):**
      * **Method:** A fine-tuned Hugging Face transformer model is used to analyze vocal tone and classify speech into various emotion categories (e.g., happiness, sadness, anger).
      * **Deployment:** Registered with MLflow and deployed as a local API endpoint via HP AI Studio's Swagger functionality. It processes audio data sent to its locally hosted endpoint.
  * **Text Emotion Model (Hugging Face Transformers):**
      * **Method:** A pre-trained or fine-tuned Hugging Face transformer model is used to analyze the sentiment and emotional tone of the transcribed text (from Whisper).
      * **Deployment:** Registered with MLflow and deployed as a local API endpoint via HP AI Studio's Swagger functionality. It processes transcribed text data sent to its locally hosted endpoint.

### Security Note

  * **No API Keys Embedded:** This project is designed to run locally. If future integrations require third-party API keys, ensure they are managed securely using environment variables (e.g., `os.environ`), not embedded directly in the code.

## 🎥 Demo Video

[Hackathon Submission Video](https://youtu.be/rzp9CGChHJ4)

## ✨ Future Work

  - **Enhanced Visualization for Temporal Analysis:** Implement real-time, scrolling line plots of emotional consistency (ECI) and individual emotion scores over time. This will allow clinicians to observe trends and dynamic shifts in emotional states throughout a session.
  - **Asynchronous Model Inference**: Implement asynchronous API calls for the deployed models (Whisper, SER, Text Emotion) to allow parallel processing of audio and text data. This would further improve real-time performance and responsiveness of the application, especially under varying network conditions or model loads.
  - **Longitudinal Emotional Tracking & Reporting:** Extend the system to track and analyze emotional patterns across multiple sessions for the same individual. This would enable clinicians to identify long-term trends, monitor progress, and gain a holistic view of a patient's emotional journey over time.
  - **Develop Methods for Analyzing Interpersonal Emotional Synchrony:** Investigate the temporal alignment and co-regulation of emotional states between the patient and the provider. This would involve exploring techniques to assess emotional "jiving" and interaction dynamics, providing insights into rapport, empathy, and the bidirectional emotional flow within therapeutic sessions.
  - **Optimized Emotion Classification Models:** Investigate and integrate more lightweight transformer models for Speech Emotion Recognition (SER) and Text Emotion classification. The goal is to further reduce the computational load on the local machine while maintaining high accuracy, ensuring even smoother real-time performance.
  - **Further DeepFace Integration with HP AI Studio:** Continue to investigate and resolve technical challenges to achieve full integration of the DeepFace model within HP AI Studio's model deployment pipeline. This would centralize all AI inference management within the HP AI Studio environment.
  - **Configurable Alerting & Thresholds:** Allow clinicians to set customizable alerts or thresholds for specific emotional states or ECI values. This would proactively flag moments of potential distress or inconsistency, guiding the clinician's attention.
  - **Extension to Multi-Participant Sessions:** Explore the capability to process and analyze interactions in sessions involving more than two participants (e.g., group therapy or family sessions), focusing on identifying overall group emotional dynamics and individual contributions.