# Facial Expression and Gesture Recognition System

A comprehensive computer vision system that combines facial expression analysis, hand gesture recognition, and face detection using state-of-the-art deep learning models and computer vision techniques.

## Features

### 1. Face Detection
- Utilizes YOLOv8 for robust face detection
- High-confidence detection threshold (>0.5)
- Bounding box visualization with confidence scores

### 2. Facial Expression Recognition
- Real-time facial expression analysis
- Detects multiple expressions:
  - Neutral 😐
  - Happy 😃
  - Sad 😢
  - Surprised 😮
  - Angry 😠
  - Fearful 😨
  - Disgusted 🤢
  - Smiling 😊
  - Blinking 😉
- Uses MediaPipe Face Mesh for precise facial landmark detection
- Custom neural network for expression classification

### 3. Hand Gesture Recognition
- Real-time hand gesture detection and classification
- Supports multiple gestures:
  - Open Palm ✋
  - Closed Fist ✊
  - Pointing 👆
  - Peace Sign ✌️
  - Thumbs Up 👍
- Utilizes MediaPipe Hands for accurate hand landmark detection

## Technologies Used

### Deep Learning & Computer Vision
- **YOLOv8**: State-of-the-art object detection for face detection
- **MediaPipe**: Framework for face mesh and hand landmark detection
- **OpenCV (cv2)**: Image processing and visualization
- **PyTorch**: Deep learning framework for expression recognition
- **NumPy**: Numerical computing and array operations

### Development Tools
- **Python**: Primary programming language
- **Virtual Environment**: Python venv for dependency management

## Project Structure

```
├── face_detection.py    # YOLOv8-based face detection implementation
├── main.py             # Main application with gesture recognition
├── requirements.txt    # Project dependencies
└── test_camera.py     # Camera testing utility
```

## Setup and Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## System Requirements

- Python 3.9+
- Webcam or compatible camera device
- CUDA-compatible GPU (recommended for optimal performance)

## Key Components

### Face Detection Module
- Implements YOLOv8 for accurate face detection
- Handles multiple face detection in a single frame
- Provides confidence scores for each detection

### Gesture Recognition System
- Real-time hand landmark detection
- Advanced gesture classification algorithm
- Support for multiple simultaneous hand tracking

### Expression Analysis
- Uses 468 facial landmarks for precise expression detection
- Custom neural network architecture for expression classification
- Real-time processing and visualization

## Performance Optimization

- Efficient frame processing using NumPy operations
- Optimized model inference
- Multi-threaded camera capture and processing
- Automatic camera device detection and initialization

## Usage

- Run the application using `python main.py`
- Press 'q' to quit the application
- The system will automatically detect and use the best available camera
- Real-time visualization of detected faces, expressions, and gestures

## Error Handling

- Robust camera initialization with multiple fallback options
- Graceful error handling for camera disconnection
- Automatic recovery from frame capture failures

## Future Improvements

- Additional gesture recognition patterns
- Enhanced expression recognition accuracy
- Performance optimizations for low-end devices
- Support for multiple camera streams
- Integration with external applications