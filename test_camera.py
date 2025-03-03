import cv2
import time

def test_camera():
    print("Testing camera access...")
    
    # Try different camera indices
    for idx in range(3):
        print(f"\nTrying camera index {idx}")
        cap = cv2.VideoCapture(idx)
        
        if not cap.isOpened():
            print(f"Failed to open camera with index {idx}")
            cap.release()
            continue
            
        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame from camera {idx}")
            cap.release()
            continue
            
        print(f"Successfully accessed camera {idx}")
        print(f"Frame shape: {frame.shape}")
        print(f"Backend being used: {cap.getBackendName()}")
        
        # Release this camera before trying next one
        cap.release()
        time.sleep(1)  # Give some time for camera to properly close

if __name__ == '__main__':
    test_camera()