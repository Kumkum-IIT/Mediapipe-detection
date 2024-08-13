# Hand Gesture Detection using OpenCV and Mediapipe

This project detects hand gestures such as "Thumbs Up," "Fist," "Peace," and "OK" using OpenCV and Mediapipe in Python. The code captures video from a webcam, processes the frames to identify hand landmarks, and then classifies the gesture based on predefined rules.

## Features

- Real-time hand gesture detection using webcam.
- Gesture recognition includes:
  - Thumbs Up
  - Fist
  - Peace
  - OK
- Visual feedback with the detected gesture displayed on the video feed.

## Requirements

- Python 3.x
- OpenCV
- Mediapipe

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/hand-gesture-detection.git
   cd hand-gesture-detection

2. Install the required dependencies:
   ```bash
   pip install opencv-python mediapipe

## Usage

1. Run the main.py script:
    ```bash
    python main.py

2. The webcam feed will open, and the program will start detecting hand gestures in real-time. Press q to exit the program.

## Code Overview
- main.py: Contains the main logic for hand gesture detection.
- Gesture Detection Functions:
  1. is_thumbs_up(hand_landmarks): Detects if the gesture is a "Thumbs Up."
  2. is_fist(hand_landmarks): Detects if the gesture is a "Fist."
  3. is_peace(hand_landmarks): Detects if the gesture is "Peace."
  4. is_ok(hand_landmarks): Detects if the gesture is "OK."
  5. resize_and_show(image): Resizes the image for display.
Main Loop: Captures frames from the webcam, processes them to detect gestures, and displays the results.
