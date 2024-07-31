import cv2
import mediapipe as mp
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

# Initialize Mediapipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(0)

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    cv2.imshow('img', img)

# Define gesture detection functions

def is_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    
    # Check if all fingertips are below their respective PIP joints
    fingers_curled = all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    )

    if (thumb_tip.y < thumb_mcp.y):
        thumb_up = thumb_tip.x < thumb_mcp.x
    thumb_up = thumb_tip.x > thumb_tip.y and thumb_mcp.x > thumb_mcp.y
    index_finger_pip = hand_landmarks.landmark[finger_pips[0]]
    if index_finger_pip.x > thumb_tip.y:
        result = fingers_curled 

        return thumb_up and result
    else:
        return False and False

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)

def is_fist(hand_landmarks):
    finger_tips = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    finger_pips = [
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    thumb_cmc = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_CMC]
    thumb_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_MCP]
    
    # Check if all fingertips are below their respective PIP joints
    fingers_curled_y = all(
        hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    )

    thumb_curled = calculate_distance(thumb_tip, thumb_cmc) < calculate_distance(thumb_tip, thumb_mcp)

    # # Debugging output
    # print(f"Fingers curled: {fingers_curled}")
    # # print(f"Thumb curled: {thumb_curled}")
    # print(f"Final result (Fist gesture): {result}")
    return fingers_curled_y

def is_peace(hand_landmarks):
    # Get relevant landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Check if index and middle fingers are extended
    index_extended = index_tip.y < index_pip.y < wrist.y
    middle_extended = middle_tip.y < middle_pip.y < wrist.y

    # Check if ring and pinky fingers are curled
    ring_curled = ring_tip.y > ring_pip.y
    pinky_curled = pinky_tip.y > pinky_pip.y

    # Check if index and middle fingers are separated
    fingers_separated = abs(index_tip.x - middle_tip.x) > abs(index_pip.x - middle_pip.x)

    return index_extended and middle_extended and ring_curled and pinky_curled and fingers_separated


def is_ok(hand_landmarks):
    # Get relevant landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Calculate distances
    def distance(p1, p2):
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5

    # Check if thumb and index finger are close (forming a circle)
    thumb_index_close = distance(thumb_tip, index_tip) < 0.1

    # Check if other fingers are extended
    def is_finger_extended(tip, mcp):
        return tip.y < mcp.y
    
    middle_extended = is_finger_extended(middle_tip, middle_mcp)
    ring_extended = is_finger_extended(ring_tip, ring_mcp)
    pinky_extended = is_finger_extended(pinky_tip, pinky_mcp)

    result = thumb_index_close and middle_extended and ring_extended and pinky_extended

    # Debugging output
    # print(f"Thumb-index close: {thumb_index_close}")
    # print(f"Middle extended: {middle_extended}")
    # print(f"Ring extended: {ring_extended}")
    # print(f"Pinky extended: {pinky_extended}")
    # print(f"Final result (OK gesture): {result}")

    return result



# Function to process uploaded image
# def process_image(image_path):
while camera.isOpened():
    # Load image
    _, image = camera.read()
    image = cv2.flip(image, 1)
    # image = cv2.imread(image_path)
    # if image is None:
    #     print("Error: Could not load image.")
    #     return

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

        # Process the image and detect hands
        results = hands.process(image_rgb)

        # Convert back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                
                # Detect gestures
                # cv2.putText(image, text, org, font, font_scale, color, thickness, line_type)

                if is_fist(hand_landmarks):
                    cv2.putText(image_bgr, 'Fist', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif is_thumbs_up(hand_landmarks):
                    cv2.putText(image_bgr, 'Thumbs Up', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif is_peace(hand_landmarks):
                    cv2.putText(image_bgr, 'Peace', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif is_ok(hand_landmarks):
                    cv2.putText(image_bgr, 'OK', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image_bgr, 'Unknown Gesture', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting image
        cv2.imshow('Hand Gesture Detection', image_bgr)
        # resize_and_show(image_bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

# Example usage
image_path = 'thumbsup.jpg'  # Replace with your image path
# process_image(image_path)

