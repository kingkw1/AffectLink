## AffectLink Environment Setup: System Dependencies

To ensure AffectLink runs correctly, especially in headless or containerized environments like Azure AI Studio, you need to install certain system-level libraries that are not handled by Python's `pip` package manager.

These instructions assume a Debian/Ubuntu-based Linux environment, which is common for many cloud and Docker deployments.

### 1. OpenCV (libGL.so.1) Dependency

**Problem:**
You might encounter an `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`. This error indicates that OpenCV, which DeepFace relies on, requires a graphics rendering library (libGL.so.1, part of OpenGL) that is missing from the system. `pip install opencv-python` only installs the Python bindings, not the underlying system library.

**Solution:**
Install the `libgl1-mesa-glx` package, which provides the necessary `libGL.so.1` library.

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx
```

### 2. SoundDevice (PortAudio) Dependency

**Problem:**
Similar to OpenCV, `sounddevice` is a Python library that provides bindings for the PortAudio library, a cross-platform audio I/O library. `pip install sounddevice` only installs the Python wrapper, not the underlying C library. Without PortAudio, `sounddevice` cannot function.

**Solution:**
Install the PortAudio development library (`portaudio19-dev`) on your system.

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev
```

### Summary of System-Level Installations

To prepare your environment for AffectLink, run both of these commands in your terminal or include them in your Dockerfile/setup script:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx portaudio19-dev
```

After installing these system dependencies, ensure all your Python dependencies (from a `requirements.txt` file) are also installed:

```bash
pip install -r requirements.txt
```

These steps should resolve the common `libGL.so.1` and `sounddevice` related errors, allowing your AffectLink application to run smoothly.