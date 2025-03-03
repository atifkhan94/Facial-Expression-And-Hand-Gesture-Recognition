import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms

class ExpressionNet(nn.Module):
    def __init__(self):
        super(ExpressionNet, self).__init__()
        self.fc1 = nn.Linear(468 * 3, 128)  # 468 landmarks with x,y,z
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 7)  # 7 expressions

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=1)

class GestureRecognizer:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face.FaceMesh()
        self.hands = self.mp_hands.Hands()

        # Initialize deep learning models
        self.expression_model = ExpressionNet()
        self.expression_model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def get_facial_expression(self, face_landmarks):
        # Extract all landmarks for deep learning model
        landmarks = np.array([[l.x, l.y, l.z] for l in face_landmarks.landmark])
        landmarks = landmarks.flatten()
        
        # Convert to tensor and get prediction
        with torch.no_grad():
            tensor_input = torch.FloatTensor(landmarks).unsqueeze(0)
            output = self.expression_model(tensor_input)
            pred_idx = torch.argmax(output).item()

        # Map prediction to expression
        expressions = ["Neutral 😐", "Happy 😃", "Sad 😢", "Surprised 😮", 
                      "Angry 😠", "Fearful 😨", "Disgusted 🤢"]
        return expressions[pred_idx]

    def get_facial_expression(self, face_landmarks):
        # Get key facial landmarks for expression recognition using more accurate points
        upper_lip = face_landmarks.landmark[13]  # Upper lip
        lower_lip = face_landmarks.landmark[14]  # Lower lip
        left_eye_top = face_landmarks.landmark[159]  # Left eye top
        left_eye_bottom = face_landmarks.landmark[145]  # Left eye bottom
        right_eye_top = face_landmarks.landmark[386]  # Right eye top
        right_eye_bottom = face_landmarks.landmark[374]  # Right eye bottom
        left_eyebrow = face_landmarks.landmark[107]  # Left eyebrow
        right_eyebrow = face_landmarks.landmark[336]  # Right eyebrow
        mouth_left = face_landmarks.landmark[61]  # Mouth corner left
        mouth_right = face_landmarks.landmark[291]  # Mouth corner right

        # Calculate normalized distances with improved accuracy
        mouth_vertical = abs(upper_lip.y - lower_lip.y)
        mouth_horizontal = abs(mouth_left.x - mouth_right.x)
        smile_ratio = mouth_horizontal / mouth_vertical if mouth_vertical > 0 else 0

        left_eye_openness = abs(left_eye_top.y - left_eye_bottom.y)
        right_eye_openness = abs(right_eye_top.y - right_eye_bottom.y)
        avg_eye_openness = (left_eye_openness + right_eye_openness) / 2

        eyebrow_height = (abs(left_eyebrow.y - left_eye_top.y) + 
                         abs(right_eyebrow.y - right_eye_top.y)) / 2

        # Enhanced expression detection with improved thresholds and multiple features
        if smile_ratio > 4.0 and mouth_vertical > 0.03:
            return "Smiling 😊"
        elif eyebrow_height > 0.04:
            return "Surprised 😮"
        elif avg_eye_openness < 0.015 and mouth_vertical < 0.02:
            return "Blinking 😉"
        elif mouth_vertical > 0.04 and eyebrow_height < 0.025:
            return "Happy 😃"
        elif mouth_vertical < 0.015 and eyebrow_height < 0.03:
            return "Sad 😢"
        return "Neutral 😐"

    def get_hand_gesture(self, hand_landmarks):
        # Get finger landmarks with base joints
        thumb_tip = hand_landmarks.landmark[4]
        thumb_base = hand_landmarks.landmark[2]
        index_tip = hand_landmarks.landmark[8]
        index_base = hand_landmarks.landmark[5]
        middle_tip = hand_landmarks.landmark[12]
        middle_base = hand_landmarks.landmark[9]
        ring_tip = hand_landmarks.landmark[16]
        ring_base = hand_landmarks.landmark[13]
        pinky_tip = hand_landmarks.landmark[20]
        pinky_base = hand_landmarks.landmark[17]
        wrist = hand_landmarks.landmark[0]

        # Calculate finger states with improved accuracy
        fingers_up = []
        finger_pairs = [
            (index_tip, index_base),
            (middle_tip, middle_base),
            (ring_tip, ring_base),
            (pinky_tip, pinky_base)
        ]

        for tip, base in finger_pairs:
            # Consider both vertical position and distance from base
            if (tip.y < base.y - 0.03) and (abs(tip.y - base.y) > 0.07):
                fingers_up.append(True)
            else:
                fingers_up.append(False)

        # Improved thumb detection using relative position
        thumb_up = False
        if hand_landmarks.landmark[17].x < wrist.x:  # Left hand
            thumb_up = thumb_tip.x < thumb_base.x - 0.04
        else:  # Right hand
            thumb_up = thumb_tip.x > thumb_base.x + 0.04

        # Enhanced gesture recognition with more precise conditions
        if all(fingers_up) and not thumb_up:
            return "Open Palm ✋"
        elif not any(fingers_up) and not thumb_up:
            return "Closed Fist ✊"
        elif fingers_up[0] and not any(fingers_up[1:]) and not thumb_up:
            return "Pointing 👆"
        elif fingers_up[0] and fingers_up[1] and not any(fingers_up[2:]) and not thumb_up:
            return "Peace Sign ✌️"
        elif thumb_up and not any(fingers_up):
            return "Thumbs Up 👍"
        return "Other"

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face landmarks
        face_results = self.face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Calculate bounding box for face
                face_points = np.array([[l.x * frame.shape[1], l.y * frame.shape[0]] for l in face_landmarks.landmark])
                x_min, y_min = np.min(face_points, axis=0).astype(int)
                x_max, y_max = np.max(face_points, axis=0).astype(int)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                # Get and display facial expression
                expression = self.get_facial_expression(face_landmarks)
                # Remove question mark from expression if present
                expression = expression.replace('?', '') if expression else expression
                cv2.putText(frame, f"Expression: {expression}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Process hand landmarks
        hand_results = self.hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                # Get and display hand gesture
                gesture = self.get_hand_gesture(hand_landmarks)
                # Remove question mark from gesture if present
                gesture = gesture.replace('?', '') if gesture else gesture
                cv2.putText(frame, f"Gesture: {gesture}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

def main():
    # Try different camera indices if default camera (0) fails
    camera_indices = list(range(4))  # Try indices 0-3
    cap = None
    selected_index = None

    print("Searching for available cameras...")
    for idx in camera_indices:
        try:
            print(f"Attempting to initialize camera {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap is not None and cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    selected_index = idx
                    print(f"Successfully initialized camera with index {idx}")
                    print(f"Frame size: {test_frame.shape}")
                    break
                else:
                    print(f"Camera {idx} opened but failed to read frame")
            else:
                print(f"Failed to open camera {idx}")
        except Exception as e:
            print(f"Error initializing camera {idx}: {str(e)}")
        if cap is not None:
            cap.release()

    if selected_index is None:
        print("Error: Could not initialize any camera. Please check if:")
        print("1. Your webcam is properly connected")
        print("2. No other application is using the webcam")
        print("3. You have given camera access permissions to this application")
        return

    try:
        recognizer = GestureRecognizer()
        print("\nStarting facial expression and hand gesture recognition...")
        print("Press 'q' to quit the application")

        cap = cv2.VideoCapture(selected_index)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame. Attempting to reconnect...")
                cap.release()
                cap = cv2.VideoCapture(selected_index)
                if not cap.isOpened():
                    print("Could not reconnect to camera. Exiting...")
                    break
                continue

            try:
                processed_frame = recognizer.process_frame(frame)
                cv2.imshow('Facial Expression & Hand Gesture Recognition', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                break

    except Exception as e:
        print(f"\nError initializing the recognition system: {str(e)}")
        print("Please make sure all required libraries are installed correctly.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()