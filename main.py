import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Specify the path to your video file
vidpath = 'video.mov'

# Initialize video capture
vidcap = cv2.VideoCapture(0)

# Set the desired window width and height
winwidth = 960
winheight = 540

# Download the hand landmarker model if not present
model_path = 'hand_landmarker.task'

# Create HandLandmarker
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

detector = vision.HandLandmarker.create_from_options(options)

# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]

# Function to draw landmarks
def draw_landmarks_on_image(bgr_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    h, w, c = bgr_image.shape
    annotated_image = np.copy(bgr_image)
    
    for hand_landmarks in hand_landmarks_list:
        # Draw connections
        for connection in HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
    
    return annotated_image

# Process video
while vidcap.isOpened():
    ret, frame = vidcap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Process the frame for hand tracking
    detection_result = detector.detect(mp_image)
    
    # Draw landmarks on the frame
    if detection_result.hand_landmarks:
        frame = draw_landmarks_on_image(frame, detection_result)

    # Draw landmarks on the frame
    if detection_result.hand_landmarks:
        frame = draw_landmarks_on_image(frame, detection_result)

    # Resize the frame to the desired window size
    resized_frame = cv2.resize(frame, (winwidth, winheight))

    # Display the resized frame
    cv2.imshow('Hand Tracking', resized_frame)

    # Exit loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
vidcap.release()
cv2.destroyAllWindows()