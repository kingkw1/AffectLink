
"""
Testing camera functionality with our new camera utilities.
"""

import os
import sys
import cv2

# Add the parent directory to sys.path to import app modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from app.run_app import find_available_camera

def test_webcam():
    print("Testing webcam availability with camera_utils.find_available_camera...")
    
    # Try to use camera_utils to find a camera
    camera_idx, backend, cap = find_available_camera()
    
    if cap is None:
        print("Failed to find any working camera")
        return False
    
    # Store the camera info in environment variables
    os.environ['WEBCAM_INDEX'] = str(camera_idx)
    os.environ['WEBCAM_BACKEND'] = backend
    
    print(f"Successfully found camera {camera_idx} with {backend} backend")
    
    # Try to read a frame and display it
    if cap.isOpened():
        print("Camera is open, trying to read a frame...")
        ret, frame = cap.read()
        if ret:
            print("Successfully read a frame from the camera")
            
            # Show the frame for verification
            cv2.imshow('Camera Test', frame)
            print("Showing camera frame. Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Clean up
            cap.release()
            return True
        else:
            print("Failed to read a frame from the camera")
            cap.release()
            return False
    else:
        print("Camera is not open")
        return False

if __name__ == "__main__":
    # Test the camera
    success = test_webcam()
    
    if success:
        print("\nCamera test successful! The webcam is working correctly.")
        print(f"Camera info: Index={os.environ.get('WEBCAM_INDEX')}, Backend={os.environ.get('WEBCAM_BACKEND')}")
    else:
        print("\nCamera test failed. Please check your camera connection and permissions.")
