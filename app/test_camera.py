import cv2
import time

def test_camera_indices():
    """Test all camera indices from 0-10 to find available cameras"""
    for idx in range(11):
        print(f"Testing camera index: {idx}")
        cap = cv2.VideoCapture(idx)
        
        if not cap.isOpened():
            print(f"  Camera {idx} cannot be opened")
            continue
            
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"  Camera {idx} opened but cannot read frame")
            cap.release()
            continue
            
        print(f"  SUCCESS: Camera {idx} is accessible and working")
        print(f"  Frame dimensions: {frame.shape}")
        
        # Display the frame for 2 seconds
        window_name = f"Camera {idx} Test"
        cv2.imshow(window_name, frame)
        cv2.waitKey(2000)  # Wait for 2 seconds
        cv2.destroyWindow(window_name)
        
        cap.release()
        
    print("Camera testing completed")

if __name__ == "__main__":
    test_camera_indices()
