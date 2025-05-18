#!/usr/bin/env python3
"""
start_realtime.py - Helper script to start AffectLink in real-time processing mode

This script sets up the environment variables and launches run_app.py with real-time
webcam and microphone processing. It handles device detection and initialization.

Usage:
    python start_realtime.py

Note:
    - You must have a webcam and/or microphone connected to your system
    - Both video and audio processing are enabled by default but can be toggled in the UI
    - The dashboard will open in your default web browser automatically
"""

import os
import sys
import subprocess
import time

def check_webcam():
    """Check if webcam is available"""
    print("Checking webcam availability...")
    try:
        # Import the camera utilities module
        from camera_utils import find_available_camera
        
        camera_idx, backend, cap = find_available_camera()
        
        if cap is not None:
            print(f"✅ Webcam found: Camera #{camera_idx} with {backend} backend.")
            cap.release()
            return True, camera_idx, backend
        else:
            print("❌ No webcam detected.")
            return False, None, None
    except Exception as e:
        print(f"❌ Error checking webcam: {e}")
        return False, None, None

def check_microphone():
    """Check if microphone is available"""
    print("Checking microphone availability...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        
        if default_input:
            print(f"✅ Microphone found: {default_input.get('name', 'Unknown')}")
            return True, default_input.get('name')
        else:
            print("❌ No microphone detected.")
            return False, None
    except Exception as e:
        print(f"❌ Error checking microphone: {e}")
        return False, None

def main():
    """Run the real-time AffectLink system"""
    print("\n=== AffectLink Real-time Processing Mode ===")
    
    # Check for required devices
    has_webcam, webcam_idx, webcam_backend = check_webcam()
    has_microphone, mic_name = check_microphone()
    
    if not has_webcam and not has_microphone:
        print("\n❌ ERROR: Neither webcam nor microphone detected.")
        print("AffectLink requires at least one of these devices to function properly.")
        input("\nPress Enter to exit...")
        return
    
    # Set environment variables for device information
    if has_webcam:
        os.environ["WEBCAM_INDEX"] = str(webcam_idx)
        if webcam_backend == "directshow":
            os.environ["WEBCAM_BACKEND"] = "directshow"
    
    print("\n🔄 Starting AffectLink in real-time processing mode...")
    
    # Get the path to the run_app.py script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_app_path = os.path.join(current_dir, "run_app.py")
    
    # Set environment variable to force detector mode
    os.environ["AFFECTLINK_DETECTOR_RUNNING"] = "1"
    
    # Start the run_app.py script
    process = subprocess.Popen([sys.executable, run_app_path])
    
    print("\n✅ AffectLink started!")
    print("• A browser window should open automatically with the dashboard.")
    print("• You can toggle video and audio processing in the dashboard.")
    print("• Press Ctrl+C in this window to stop AffectLink completely.\n")
    
    try:
        # Keep this script running so the user can Ctrl+C to stop
        while True:
            time.sleep(1)
            # Check if the subprocess is still alive
            if process.poll() is not None:
                print("❌ AffectLink process has terminated unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\n⏹️ Keyboard interrupt received. Shutting down AffectLink...")
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            # Force kill if needed
            print("Forcing process termination...")
            process.kill()
    
    print("✅ AffectLink shutdown complete.")

if __name__ == "__main__":
    main()
