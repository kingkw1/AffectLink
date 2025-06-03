## AffectLink Environment Setup: System Dependencies

To ensure AffectLink runs correctly, especially in headless or containerized environments like Azure AI Studio, you need to install certain system-level libraries that aren't handled by Python's `pip` package manager.

These instructions assume a Debian/Ubuntu-based Linux environment, which is common for many cloud and Docker deployments.

---

### 1. OpenCV (`libGL.so.1`) Dependency

**Problem:**
You might hit an `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`. This means OpenCV, a dependency of DeepFace, needs a graphics rendering library (`libGL.so.1`, part of OpenGL) that's missing from the system. Keep in mind that `pip install opencv-python` only gets you the Python bindings, not this underlying system library.

**Solution:**
Install the `libgl1-mesa-glx` package; it provides the essential `libGL.so.1` library.

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```

---

### 2. SoundDevice (PortAudio) Dependency

**Problem:**
Similar to OpenCV, `sounddevice` is a Python library that wraps the PortAudio library, a versatile audio I/O library. Just like before, `pip install sounddevice` only installs the Python wrapper, not the core C library. Without PortAudio, `sounddevice` can't do its job.

**Solution:**
Install the PortAudio development library (`portaudio19-dev`) on your system.

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
```

---

### 3. FFmpeg Dependency (for Audio/Video Processing)

**Problem:**
When `librosa` tries to extract audio from video files (like MP4s), you might see errors such as `soundfile.LibsndfileError: Format not recognised` or `audioread.exceptions.NoBackendError`. This happens because the audio processing libraries (`soundfile`, `audioread`) don't have the right codecs or tools, specifically FFmpeg, to handle the video format.

**Solution:**
Install **FFmpeg**, a robust multimedia framework that's crucial for decoding and encoding a wide range of audio and video formats.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

---

### Summary of System-Level Installations

To fully prepare your environment for AffectLink, run all these commands in your terminal (or include them in your Dockerfile/setup script):

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx portaudio19-dev ffmpeg
```

After installing these system dependencies, make sure all your Python dependencies (from your `requirements.txt` file) are also installed:

```bash
pip install -r requirements.txt
```

These steps should now completely resolve all common dependency-related errors, allowing your AffectLink application to run smoothly and process both facial emotions and audio.