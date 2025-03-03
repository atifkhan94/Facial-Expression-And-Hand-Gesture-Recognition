from ultralytics import YOLO
import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Initialize YOLOv8 model for face detection
        self.model = YOLO('yolov8n-face.pt')
        
    def detect_faces(self, frame):
        # Run YOLOv8 inference on the frame
        results = self.model(frame)
        
        # Initialize list to store face detections
        faces = []
        
        # Process detection results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                
                # Only keep high confidence detections
                if confidence > 0.5:
                    faces.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence
                    })
        
        return faces
    
    def draw_faces(self, frame, faces):
        # Draw bounding boxes and confidence scores for each detected face
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            conf = face['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f'Face: {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame