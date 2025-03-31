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
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=2)
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
    # Get coordinates of key landmarks
    tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]
    
    # Get coordinates for the base of each finger
    pip = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    ]
    
    # Get wrist position for thumb calculation
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    
    # Count extended fingers
    extended_fingers = 0
    
    # Special case for thumb
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    is_right_hand = thumb_mcp.x > wrist.x  # Reversed for flipped image
    
    if (is_right_hand and tips[0].x > pip[0].x) or (not is_right_hand and tips[0].x < pip[0].x):
        extended_fingers += 1
    
    # For other fingers, check if the tip is higher than the PIP joint
    for i in range(1, 5):
        if tips[i].y < pip[i].y:  # If fingertip is higher than PIP joint
            extended_fingers += 1
            
    return extended_fingers

# Function to detect pinch gesture
def detect_pinch(hand_landmarks, is_right_hand):
    if not is_right_hand:
        return False, 0, 0, 0
    
    # Get landmark positions
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Calculate distance between thumb and index finger
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x)**2 + 
        (thumb_tip.y - index_tip.y)**2 + 
        (thumb_tip.z - index_tip.z)**2
    )
    
    # Check distances to other fingers to ensure it's specifically a thumb-index pinch
    index_middle_dist = math.sqrt((index_tip.x - middle_tip.x)**2 + (index_tip.y - middle_tip.y)**2)
    middle_ring_dist = math.sqrt((middle_tip.x - ring_tip.x)**2 + (middle_tip.y - ring_tip.y)**2)
    ring_pinky_dist = math.sqrt((ring_tip.x - pinky_tip.x)**2 + (ring_tip.y - pinky_tip.y)**2)
    
    # Define pinch threshold
    pinch_threshold = 0.05
    
    # Ensure other fingers are spread apart to confirm intentional pinch
    other_fingers_spread = (index_middle_dist > 0.03 and middle_ring_dist > 0.03 and ring_pinky_dist > 0.03)
    
    # Check if fingers are pinched
    is_pinching = distance < pinch_threshold and other_fingers_spread
    
    # Calculate pinch position (midpoint between thumb and index)
    pinch_x = (thumb_tip.x + index_tip.x) / 2
    pinch_y = (thumb_tip.y + index_tip.y) / 2
    
    return is_pinching, (pinch_x, pinch_y), (thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y)

# Function to determine if hand is left or right
def identify_hand_type(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Since the image is flipped, right hand will have thumb to the right of pinky
    is_right_hand = thumb_tip.x > pinky_tip.x
    
    return "Right" if is_right_hand else "Left"

# Video capture
cap = cv2.VideoCapture(0)

# Initialize state variables
pinch_active = False
initial_pinch_pos = (0, 0)
pinch_history = []  # Store recent pinch positions
volume_history = []  # For smoothing volume changes
scroll_start_time = 0
scroll_direction = None  # 'up' or 'down'
prev_y = 0
initial_volume = current_volume

# Gesture control parameters
history_length = 5  # Number of frames to track for movement detection
scroll_activation_delay = 0.3  # Seconds to wait after pinch before scrolling activates
scroll_sensitivity = 2.0  # Higher = more sensitive to movement
volume_change_interval = 0.05  # Seconds between volume updates
last_volume_change_time = time.time()

# Get initial system volume if possible
try:
    current_volume = get_system_volume()
except:
    current_volume = 50  # Default if can't get current volume

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    
    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
        
    # To improve performance, mark the image as not writeable
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process hands and face
    results_hands = hands.process(image)
    results_face = face_detection.process(image)
    
    # Make the image writeable again for drawing
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Variables to track hand activity in this frame
    finger_count = 0
    is_pinching_now = False
    current_pinch_pos = None
    pinch_display_color = (0, 255, 255)  # Yellow by default
    
    # Check if it's time to process volume change
    current_time = time.time()
    can_change_volume = current_time - last_volume_change_time >= volume_change_interval
    
    # Process hand landmarks if detected
    if results_hands.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            # Get hand type
            if results_hands.multi_handedness:
                hand_type = results_hands.multi_handedness[idx].classification[0].label
            else:
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
                is_pinch, pinch_pos, thumb_pos, index_pos = detect_pinch(hand_landmarks, is_right_hand)
                
                if is_pinch:
                    is_pinching_now = True
                    current_pinch_pos = pinch_pos
                    
                    # Convert normalized positions to pixel coordinates
                    thumb_pixel = (int(thumb_pos[0] * image.shape[1]), int(thumb_pos[1] * image.shape[0]))
                    index_pixel = (int(index_pos[0] * image.shape[1]), int(index_pos[1] * image.shape[0]))
                    pinch_pixel = (int(pinch_pos[0] * image.shape[1]), int(pinch_pos[1] * image.shape[0]))
                    
                    # Draw pinch visualization
                    cv2.line(image, thumb_pixel, index_pixel, (0, 0, 255), 3)
                    cv2.circle(image, thumb_pixel, 10, (255, 0, 0), -1)
                    cv2.circle(image, index_pixel, 10, (255, 0, 0), -1)
                    
                    # Handle pinch state management
                    if not pinch_active:
                        # New pinch detected
                        pinch_active = True
                        initial_pinch_pos = pinch_pos
                        pinch_history = [pinch_pos]  # Reset history
                        initial_volume = current_volume  # Remember starting volume
                        scroll_start_time = current_time
                        scroll_direction = None
                        prev_y = pinch_pos[1]
                        pinch_display_color = (0, 255, 255)  # Yellow when first pinched
                    else:
                        # Continue existing pinch
                        pinch_history.append(pinch_pos)
                        if len(pinch_history) > history_length:
                            pinch_history.pop(0)
                        
                        # Check if we've been pinching long enough to activate scrolling
                        if current_time - scroll_start_time >= scroll_activation_delay:
                            # Calculate movement trend over last few frames
                            if len(pinch_history) >= 3:  # Need enough history to detect trend
                                # Calculate vertical movement (y-axis)
                                first_y = pinch_history[0][1]
                                last_y = pinch_history[-1][1]
                                y_diff = first_y - last_y  # Positive when moving up
                                
                                # Determine if movement is consistent enough to be considered scrolling
                                movement_consistent = True
                                for i in range(1, len(pinch_history)):
                                    # Check if all movements are in the same direction
                                    curr_diff = pinch_history[i-1][1] - pinch_history[i][1]
                                    if (curr_diff > 0 and y_diff < 0) or (curr_diff < 0 and y_diff > 0):
                                        movement_consistent = False
                                        break
                                
                                # Update scroll direction if movement is significant and consistent
                                if abs(y_diff) > 0.01 and movement_consistent:
                                    new_direction = "up" if y_diff > 0 else "down"
                                    
                                    # Only change direction if it's significant or direction changes
                                    if scroll_direction != new_direction or abs(y_diff) > 0.03:
                                        scroll_direction = new_direction
                                
                                # Apply volume change if we have a scroll direction
                                if scroll_direction and can_change_volume:
                                    # Calculate scroll magnitude (how far we've moved)
                                    magnitude = abs(y_diff) * scroll_sensitivity
                                    
                                    # Apply volume change based on direction
                                    volume_change = magnitude * 100
                                    if scroll_direction == "down":
                                        volume_change = -volume_change
                                    
                                    # Calculate new volume
                                    new_volume = int(current_volume + volume_change)
                                    new_volume = max(min_volume, min(max_volume, new_volume))
                                    
                                    # Apply volume change if significant
                                    if new_volume != current_volume:
                                        try:
                                            current_volume = set_volume(new_volume)
                                            last_volume_change_time = current_time
                                            # Reset history to avoid over-scrolling
                                            pinch_history = pinch_history[-2:]
                                        except Exception as e:
                                            print(f"Volume control error: {e}")
                                    
                                    # Change pinch visualization color based on scroll direction
                                    if scroll_direction == "up":
                                        pinch_display_color = (0, 255, 0)  # Green for volume up
                                    else:
                                        pinch_display_color = (0, 0, 255)  # Red for volume down
                            
                            # Display scroll status
                            status_text = f"Scrolling: {scroll_direction.upper() if scroll_direction else 'NONE'}"
                            cv2.putText(image, status_text, (pinch_pixel[0] - 80, pinch_pixel[1] - 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, pinch_display_color, 2)
    
    # Reset pinch state if no pinch is currently detected
    if pinch_active and not is_pinching_now:
        pinch_active = False
        pinch_history = []
        scroll_direction = None
    
    # Draw face detections
    if results_face.detections:
        for detection in results_face.detections:
            mp_drawing.draw_detection(image, detection)
    
    # Display finger count
    cv2.putText(image, f"Total Fingers: {finger_count}", 
               (10, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display volume level
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
    status_text = "Pinch: Detected" if is_pinching_now else "Pinch: None"
    cv2.putText(image, status_text, 
               (10, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, pinch_display_color if is_pinching_now else (0, 255, 255), 2)
    
    # Display scroll instruction when pinching but not scrolling yet
    if pinch_active and not scroll_direction and current_time - scroll_start_time < scroll_activation_delay:
        cv2.putText(image, "Hold steady, then move to scroll", 
                   (10, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
    
    cv2.imshow('MediaPipe Hands and Face Detection', image)
    
    if cv2.waitKey(5) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()