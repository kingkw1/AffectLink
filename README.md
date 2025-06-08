# AffectLink: Multimodal Emotion Consistency Tracking for Telehealth

**AffectLink** is an AI-powered system that analyzes a patient’s emotional state by comparing facial expressions with vocal tone. It helps identify inconsistencies, fatigue, or stress — providing clinicians with real-time and retrospective emotional insights during virtual sessions.

## 🔍 What It Does
- 🎙️ Detects speech-based emotional tone using Whisper + emotion classification.
- 🎥 Detects facial emotion using real-time video feed.
- 🧩 Compares the two to assess emotional consistency or mismatch.
- 📈 Tracks mood and stress markers over time to surface potential concerns.
- 🧑‍⚕️ Built for clinicians and therapists using telehealth platforms.

## 🚀 Features
- Multimodal emotional analysis (audio + video).
- Session-level emotional summary reports.
- Local-first privacy-preserving deployment.
- Optional integration with MLflow and Swagger.

## 🛠️ Built With
- Python
- OpenCV, DeepFace, Whisper
- Streamlit / Flask
- MLflow (optional)
- HP AI Studio or local NVIDIA hardware

## 🧪 Setup Instructions
```bash
git clone https://github.com/kingkw1/AffectLink.git
cd AffectLink
pip install -r requirements.txt
python app/main.py
```

## 📄 Demo Video
- [Link to Demo Video Here]

## 📚 Documentation

### Architecture
```mermaid
graph TD
    A["User / Webcam / Mic 🎤📹"] -->|Input Stream| B["AffectLink Streamlit UI 🖥️"]
    
    subgraph "Local Workstation / HP AI Studio 🔒"
        B -->|Data & Commands| C["main_processor.py (Orchestrator)"]
        
        C -->|Audio Frames| D["Audio Emotion Processor 🎵"]
        C -->|Video Frames| E["Video Emotion Processor 🖼️"]
        
        D -->|Audio Emotions & Transcriptions| C
        E -->|Facial Emotions| C
        
        C -->|Log Metrics & Artifacts| F["MLflow (Tracking & Models) 📊"]
        C -->|Text Inference Request| G["Swagger API (Local Model Deployment) 🚀"]
        G -->|Text Emotion Results| C
        
        C -->|Display Results & Consistency Index| B
    end
    
    style A fill:#d9e8fb,stroke:#333,stroke-width:2px,color:#1a1a1a
    style B fill:#d0f0c0,stroke:#333,color:#1a1a1a
    style C fill:#fef3bd,stroke:#333,color:#1a1a1a
    style D fill:#e6e6fa,stroke:#333,color:#1a1a1a
    style E fill:#e6e6fa,stroke:#333,color:#1a1a1a
    style F fill:#cceeff,stroke:#333,color:#1a1a1a
    style G fill:#ffcccc,stroke:#333,color:#1a1a1a
```

- [Architecture Diagram](docs/architecture.png)
- [Demo Scenario Script](docs/demo_script.md)

## ✨ Future Work
- Add sentiment summarization of conversation.
- Improve emotion classification with lightweight transformers.
- Extend to triadic (3+ participants) sessions.

## 📜 License
MIT License

---
Built with 💬 by Kevin King for the HP AI Studio & NVIDIA Developer Challenge.