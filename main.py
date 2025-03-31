import cv2
import mediapipe as mp
import numpy as np
import math
import platform
import subprocess
import time

# For volume control
if platform.system() == "Windows":
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
elif platform.system() == "Darwin":  # macOS
    import osascript
elif platform.system() == "Linux":
    pass  # Will use subprocess for Linux

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection

# Initialize detectors
hands = mp_hands.Hands(
    model_complexity=1,  # Increased model complexity for better accuracy
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=2)  # Allow detection of both hands
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize volume control based on platform
volume_control = None
min_volume = 0
max_volume = 100
current_volume = 50  # Default starting volume

if platform.system() == "Windows":
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_control = cast(interface, POINTER(IAudioEndpointVolume))
    min_volume = 0
    max_volume = 100
    # Get initial volume
    current_volume = int(volume_control.GetMasterVolumeLevelScalar() * 100)

# Function to get current system volume
def get_system_volume():
    if platform.system() == "Windows":
        return int(volume_control.GetMasterVolumeLevelScalar() * 100)
    elif platform.system() == "Darwin":  # macOS
        volume_script = "output volume of (get volume settings)"
        result = osascript.osascript(volume_script)
        if result[0] == 0:  # Success
            return int(result[1])
    elif platform.system() == "Linux":
        try:
            result = subprocess.check_output(["amixer", "-D", "pulse", "sget", "Master"])
            volume = int(result.decode().strip().split("[")[1].split("%")[0])
            return volume
        except:
            pass
    return 50  # Default fallback

# Function to set system volume
def set_volume(volume_percentage):
    volume_percentage = max(min_volume, min(max_volume, volume_percentage))
    if platform.system() == "Windows":
        # Windows: Convert percentage to scalar (0-1)
        volume_control.SetMasterVolumeLevelScalar(volume_percentage / 100, None)
    elif platform.system() == "Darwin":  # macOS
        osascript.osascript(f"set volume output volume {volume_percentage}")
    elif platform.system() == "Linux":
        subprocess.call(["amixer", "-D", "pulse", "sset", "Master", f"{volume_percentage}%"])
    return volume_percentage

# Function to count fingers
def count_fingers(hand_landmarks, image_width):
    # For a flipped image, we need to adjust the x-coordinate logic
    # Get coordinates of key landmarks
    tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],    # Thumb tip
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],   # Index tip
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],  # Middle tip
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],    # Ring tip
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]     # Pinky tip
    ]
    
    # Get coordinates for the base of each finger
    pip = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP],  # Thumb IP joint (different for thumb)
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],   # Index PIP joint
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],  # Middle PIP joint
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],    # Ring PIP joint
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]     # Pinky PIP joint
    ]
    
    # Get wrist position for thumb calculation
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Count extended fingers
    extended_fingers = 0
    
    # Special case for thumb
    # Thumb is extended if the tip is to the left/right of the IP joint (depending on which hand)
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # For flipped image, the logic is reversed
    is_right_hand = thumb_mcp.x > wrist.x  # Reversed for flipped image
    
    if (is_right_hand and tips[0].x > pip[0].x) or (not is_right_hand and tips[0].x < pip[0].x):
        extended_fingers += 1
    
    # For other fingers, check if the tip is higher than the PIP joint
    for i in range(1, 5):
        if tips[i].y < pip[i].y:  # If fingertip is higher than PIP joint
            extended_fingers += 1
            
    return extended_fingers

# Function to detect pinch gesture and calculate volume
def detect_pinch(hand_landmarks, is_right_hand):
    if not is_right_hand:
        return False, 0, 0, 0
    
    # Get landmark positions
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Calculate distance between thumb and index finger (Euclidean distance in 3D space)
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x)**2 + 
        (thumb_tip.y - index_tip.y)**2 + 
        (thumb_tip.z - index_tip.z)**2
    )
    
    # Check distances to other fingers to ensure it's specifically a thumb-index pinch
    # and not a general hand closure
    index_middle_dist = math.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
    middle_ring_dist = math.sqrt((middle_tip.x - ring_tip.x)**2 + (middle_tip.y - ring_tip.y)**2)
    ring_pinky_dist = math.sqrt((ring_tip.x - pinky_tip.x)**2 + (ring_tip.y - pinky_tip.y)**2)
    
    # Define pinch threshold (adjust based on your needs)
    # Smaller value = more sensitive
    pinch_threshold = 0.05
    
    # Ensure other fingers are spread apart to confirm intentional pinch
    other_fingers_spread = (index_middle_dist > 0.03 and middle_ring_dist > 0.03 and ring_pinky_dist > 0.03)
    
    # Check if fingers are pinched
    is_pinching = distance < pinch_threshold and other_fingers_spread
    
    # Calculate volume based on vertical position
    # Lower y value means higher position on screen
    pinch_y = (thumb_tip.y + index_tip.y) / 2
    # Calculate height relative to wrist position (normalize)
    height_ratio = max(0, min(1, (wrist.y - pinch_y) * 1.5))  # Adjusted multiplier for better range
    height_percent = height_ratio * 100
    
    # Calculate actual 3D positions for visualization
    thumb_x, thumb_y = thumb_tip.x, thumb_tip.y
    index_x, index_y = index_tip.x, index_tip.y
    
    return is_pinching, height_percent, (thumb_x, thumb_y), (index_x, index_y)

# Function to determine if hand is left or right
def identify_hand_type(hand_landmarks):
    # MediaPipe actually provides this information, but as backup:
    # Check the position of the thumb relative to the pinky
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Since the image is flipped, right hand will have thumb to the right of pinky
    is_right_hand = thumb_tip.x > pinky_tip.x
    
    return "Right" if is_right_hand else "Left"

# Video capture
cap = cv2.VideoCapture(0)

# Initialize variables for tracking pinch state
was_pinching = False
pinch_start_volume = current_volume
volume_change_smoothing = 5  # Frames to average for smoothing
last_positions = []
last_volume_change_time = time.time()
volume_change_interval = 0.05  # seconds between volume adjustments

# Get initial system volume if possible
try:
    current_volume = get_system_volume()
    pinch_start_volume = current_volume
except:
    current_volume = 50  # Default if can't get current volume
    pinch_start_volume = current_volume

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
        
    # To improve performance, optionally mark the image as not writeable
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process hands
    results_hands = hands.process(image)
    
    # Process face
    results_face = face_detection.process(image)
    
    # Draw the hand annotations on the image
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    finger_count = 0
    is_pinching = False
    height_percent = 0
    thumb_pos = (0, 0)
    index_pos = (0, 0)
    
    # Check if it's time to process volume change
    current_time = time.time()
    can_change_volume = current_time - last_volume_change_time >= volume_change_interval
    
    # Draw hand landmarks and count fingers
    if results_hands.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # Get hand type information
            if results_hands.multi_handedness:
                # Use MediaPipe's classification
                hand_type = results_hands.multi_handedness[idx].classification[0].label
            else:
                # Fallback to our function
                hand_type = identify_hand_type(hand_landmarks)
                
            is_right_hand = (hand_type == "Right")
            
            # Draw landmarks
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Count fingers
            hand_finger_count = count_fingers(hand_landmarks, image.shape[1])
            finger_count += hand_finger_count
            
            # Get hand position for text
            hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image.shape[1])
            hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0])
            
            # Display hand type and count near the hand
            cv2.putText(image, f"{hand_type} Hand: {hand_finger_count} fingers", 
                       (hand_x - 80, hand_y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Check for pinch gesture on right hand
            if is_right_hand:
                current_pinch, curr_height, thumb_pos, index_pos = detect_pinch(hand_landmarks, is_right_hand)
                
                if current_pinch:
                    is_pinching = True
                    height_percent = curr_height
                    
                    # Convert normalized positions to pixel coordinates
                    thumb_pixel = (int(thumb_pos[0] * image.shape[1]), int(thumb_pos[1] * image.shape[0]))
                    index_pixel = (int(index_pos[0] * image.shape[1]), int(index_pos[1] * image.shape[0]))
                    
                    # Store position for smoothing
                    last_positions.append(height_percent)
                    if len(last_positions) > volume_change_smoothing:
                        last_positions.pop(0)
                    
                    # Calculate smoothed height
                    height_percent = sum(last_positions) / len(last_positions)
                    
                    # Visual feedback for pinch
                    cv2.line(image, thumb_pixel, index_pixel, (0, 0, 255), 3)
                    cv2.circle(image, thumb_pixel, 10, (255, 0, 0), -1)
                    cv2.circle(image, index_pixel, 10, (255, 0, 0), -1)
                    
                    midpoint = ((thumb_pixel[0] + index_pixel[0]) // 2, (thumb_pixel[1] + index_pixel[1]) // 2)
                    cv2.putText(image, f"Height: {int(height_percent)}%", 
                               (midpoint[0] - 50, midpoint[1] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Handle volume control
    if is_pinching:
        if not was_pinching:
            # Just started pinching, store current volume
            pinch_start_volume = current_volume
            last_positions = [height_percent] * volume_change_smoothing  # Initialize smoothing
            was_pinching = True
        
        if can_change_volume:
            # Calculate new volume - map height to volume (start from current level)
            # Scale so small hand movements make reasonable volume changes
            new_volume = int(height_percent)
            
            if new_volume != current_volume:
                try:
                    current_volume = set_volume(new_volume)
                    last_volume_change_time = current_time
                except Exception as e:
                    print(f"Volume control error: {e}")
    elif was_pinching and not is_pinching:
        # Just stopped pinching
        last_positions = []
        was_pinching = False
    
    # Draw face detections
    if results_face.detections:
        for detection in results_face.detections:
            mp_drawing.draw_detection(image, detection)
    
    # Display finger count in top-left corner
    cv2.putText(image, f"Total Fingers: {finger_count}", 
               (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display volume level and a visual volume bar
    cv2.putText(image, f"Volume: {current_volume}%", 
               (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display volume bar
    bar_x = 10
    bar_y = 120
    bar_width = 200
    bar_height = 20
    filled_width = int(bar_width * current_volume / 100)
    
    # Draw background rectangle
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
    # Draw filled rectangle
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 0, 255), -1)
    # Draw border
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 2)
    
    # Display pinch status
    status_text = "Pinch: Detected" if is_pinching else "Pinch: None"
    cv2.putText(image, status_text, 
               (10, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow('MediaPipe Hands and Face Detection', image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()