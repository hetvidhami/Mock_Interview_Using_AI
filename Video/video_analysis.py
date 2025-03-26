import cv2
import numpy as np
import mediapipe as mp
import dlib
from scipy.spatial import distance as dist
import argparse
import math
import time
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Initialize MediaPipe Face Mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# Load emotion detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_classifier = load_model('model.h5')  # Make sure this model file exists
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Constants for EAR (Eye Aspect Ratio)
EAR_THRESHOLD = 0.25 
BLINK_COUNT = 0
LONG_EYE_CLOSURE_COUNT = 0
EYE_CLOSED = False  # Track if eyes are currently closed
BLINK_COOLDOWN = 5  # Cooldown frames between blinks
cooldown_counter = 0  # Counter for cooldown
LONG_EYE_CLOSURE_THRESHOLD = 90
eye_closed_frames = 0

# Constants for no face detection
NO_FACE_THRESHOLD = 90 
no_face_frames = 0 
total_no_face_count = 0  

# Add normal blink rate parameters
NORMAL_BLINK_RATE = 15  # Normal blink rate per minute
BLINK_RATE_PENALTY_THRESHOLD = NORMAL_BLINK_RATE * 2 
BLINK_RATE_PENALTY = 0.5 

# Head pose tracking
POSE_HOLD_THRESHOLD = 60 
current_pose = None
pose_frames = 0

# Pose counters
left_look_count = 0
right_look_count = 0
up_look_count = 0
down_look_count = 0
left_lean_count = 0
right_lean_count = 0

# Emotion tracking
emotion_count = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Neutral": 0, "Sad": 0, "Surprise": 0}

# Timing variables
start_time = None
total_frames = 0

# Function to calculate EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate face angles (roll, yaw, pitch)
def calculate_face_angle(mesh):
    if not mesh:
        return {}

    # Extract required landmarks
    # MediaPipe Face Mesh landmark indices
    nose_tip = mesh[4]       # Nose tip
    forehead = mesh[10]      # Forehead
    left_eye_inner = mesh[133]  # Left eye inner corner
    right_eye_inner = mesh[362] # Right eye inner corner
    chin = mesh[152]         # Chin

    # Calculate vectors for orientation
    # Horizontal (left to right eye)
    h_vec = np.array([right_eye_inner.x - left_eye_inner.x,
                     right_eye_inner.y - left_eye_inner.y,
                     right_eye_inner.z - left_eye_inner.z])
    
    # Vertical (forehead to chin)
    v_vec = np.array([forehead.x - chin.x,
                     forehead.y - chin.y,
                     forehead.z - chin.z])
    
    # Calculate angles
    # Roll (head tilt side to side)
    roll = np.arctan2(h_vec[1], h_vec[0])
    
    # Yaw (head turning left/right)
    yaw = np.arctan2(h_vec[2], h_vec[0])
    
    # Pitch (head up/down)
    pitch = np.arctan2(v_vec[2], v_vec[1])
    
    return {
        "roll": roll,
        "yaw": yaw,
        "pitch": pitch
    }

# Function to classify head pose based on angles
def classify_head_pose(angles):
    roll = angles["roll"]
    yaw = angles["yaw"]
    pitch = angles["pitch"]

    # Classify roll (leaning left/right)
    if roll < -0.1:
        roll_status = "Leaning Right"
    elif roll > 0.1:
        roll_status = "Leaning Left"
    else:
        roll_status = "Straight (No Lean)"

    # Classify yaw (turning left/right)
    if yaw < -0.1:
        yaw_status = "Looking Right"
    elif yaw > 0.1:
        yaw_status = "Looking Left"
    else:
        yaw_status = "Straight (No Turn)"

    # Classify pitch (looking up/down)
    if pitch < -0.1:
        pitch_status = "Looking Down"
    elif pitch > 0.1:
        pitch_status = "Looking Up"
    else:
        pitch_status = "Straight (No Tilt)"

    return roll_status, yaw_status, pitch_status

# Function to update pose counters
def update_pose_counters(roll_status, yaw_status, pitch_status):
    global current_pose, pose_frames, left_look_count, right_look_count
    global up_look_count, down_look_count, left_lean_count, right_lean_count
    
    # Determine current pose
    new_pose = (roll_status, yaw_status, pitch_status)
    
    if new_pose == current_pose:
        pose_frames += 1
        if pose_frames == POSE_HOLD_THRESHOLD:
            # Count the sustained poses
            if current_pose[1] == "Looking Left":
                left_look_count += 1
                print("Looking Left for 3 seconds - Count:", left_look_count)
            elif current_pose[1] == "Looking Right":
                right_look_count += 1
                print("Looking Right for 3 seconds - Count:", right_look_count)
                
            if current_pose[2] == "Looking Up":
                up_look_count += 1
                print("Looking Up for 3 seconds - Count:", up_look_count)
            elif current_pose[2] == "Looking Down":
                down_look_count += 1
                print("Looking Down for 3 seconds - Count:", down_look_count)
                
            if current_pose[0] == "Leaning Left":
                left_lean_count += 1
                print("Leaning Left for 3 seconds - Count:", left_lean_count)
            elif current_pose[0] == "Leaning Right":
                right_lean_count += 1
                print("Leaning Right for 3 seconds - Count:", right_lean_count)
    else:
        current_pose = new_pose
        pose_frames = 1

def calculate_additional_metrics(duration_seconds):
    global emotion_count, left_look_count, right_look_count, up_look_count, down_look_count
    global left_lean_count, right_lean_count, LONG_EYE_CLOSURE_COUNT, total_no_face_count, BLINK_COUNT
    
    # Calculate blink rate
    blink_rate = (BLINK_COUNT / duration_seconds) * 60 if duration_seconds > 0 else 0
    
    # Calculate positive emotions percentage
    positive_emotions = emotion_count["Happy"] + emotion_count["Surprise"] + emotion_count["Neutral"]
    total_emotions = sum(emotion_count.values())
    positive_percentage = (positive_emotions / total_emotions) * 100 if total_emotions > 0 else 0

    # Calculate looking/leaning away counts
    total_looking_away = left_look_count + right_look_count + up_look_count + down_look_count
    total_leaning_away = left_lean_count + right_lean_count

    # Calculate interview score (base 10/10)
    interview_score = 10.0
    
    # Emotion percentage deductions
    if positive_percentage >= 80:
        # No deduction for good emotional engagement
        pass
    elif 70 <= positive_percentage < 80:
        interview_score -= 0.3
    elif 50 <= positive_percentage < 70:
        # Scale deduction between 0.5-0.7 based on how close to 50%
        deduction = 0.5 + (0.2 * ((70 - positive_percentage) / 20))
        interview_score -= deduction
    else:
        interview_score -= 1.0
    
    # Other deductions
    interview_score -= LONG_EYE_CLOSURE_COUNT * 0.5
    interview_score -= total_no_face_count * 0.5    
    interview_score -= total_looking_away * 0.5    
    interview_score -= total_leaning_away * 0.3
    
    # Add blink rate deduction if blinking too fast
    if blink_rate > BLINK_RATE_PENALTY_THRESHOLD:
        interview_score -= BLINK_RATE_PENALTY
        print(f"⚠️ Excessive blinking detected: {blink_rate:.1f} blinks/min (normal is {NORMAL_BLINK_RATE})")
    
    # Ensure score doesn't go below 0
    interview_score = max(0, interview_score)

    return positive_percentage, total_looking_away, total_leaning_away, interview_score, blink_rate

# Update the display_final_statistics function to show these metrics
def display_final_statistics(duration_seconds):
    global BLINK_COUNT, LONG_EYE_CLOSURE_COUNT, total_no_face_count
    global left_look_count, right_look_count, up_look_count, down_look_count
    global left_lean_count, right_lean_count, emotion_count
    
    # Calculate additional metrics
    positive_percentage, total_looking_away, total_leaning_away, interview_score, blink_rate = calculate_additional_metrics(duration_seconds)
    
    # Calculate blink rate per minute
    if duration_seconds > 0:
        blink_rate = (BLINK_COUNT / duration_seconds) * 60
    else:
        blink_rate = 0
    
    
    print("\n===== FINAL STATISTICS =====")
    print(f"Total duration: {duration_seconds:.2f} seconds")
    print(f"Total blinks: {BLINK_COUNT}")
    print(f"Blink rate: {blink_rate:.2f} blinks per minute")
    print(f"Long eye closures (>=3s): {LONG_EYE_CLOSURE_COUNT}")
    print(f"Face not detected (>=3s): {total_no_face_count}")
    print("\nHead Movement Counts:")
    print(f"Looking left: {left_look_count}")
    print(f"Looking right: {right_look_count}")
    print(f"Looking up: {up_look_count}")
    print(f"Looking down: {down_look_count}")
    print(f"Leaning left: {left_lean_count}")
    print(f"Leaning right: {right_lean_count}")
    print("\nEmotion Detection Counts:")
    for emotion, count in emotion_count.items():
        print(f"{emotion}: {count}")
    
    # New metrics display
    print("\n===== PERFORMANCE ANALYSIS =====")
    print(f"Positive emotions percentage: {positive_percentage:.2f}%")
    print(f"Total times looking away: {total_looking_away}")
    print(f"Total times leaning away: {total_leaning_away}")
    print(f"\nINTERVIEW SCORE: {interview_score:.1f}/10")
    
    # Feedback
    print("\n===== FEEDBACK =====")
    if positive_percentage >= 70:
        print("👍 Excellent emotional engagement!")
    else:
        print("👎 Try to maintain more positive facial expressions")
    
    if total_looking_away > 2:
        print("👎 Reduce looking away from camera")
    elif total_looking_away > 0:
        print("⚠️ Try to maintain better eye contact")
    else:
        print("👍 Excellent eye contact maintained")
    
    if LONG_EYE_CLOSURE_COUNT > 0:
        print(f"👎 Avoid long eye closures (had {LONG_EYE_CLOSURE_COUNT})")
    
    if total_no_face_count > 0:
        print(f"👎 Stay centered in frame (missed face {total_no_face_count} times)")

# Function to classify facial expression
def classify_facial_expression(frame, face_rect):
    # Convert dlib rectangle to coordinates
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    
    # Extract ROI and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+h, x:x+w]
    
    try:
        # Resize to 48x48 as expected by most emotion recognition models
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) == 0:
            return "No Face"
            
        # Normalize and prepare for prediction
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Make prediction
        prediction = emotion_classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        
        # Update emotion count
        emotion_count[label] += 1
        
        return label
        
    except Exception as e:
        print(f"Expression detection error: {e}")
        return "Neutral"

# Function to process a video file or live webcam feed
def process_video(input_source):
    global BLINK_COUNT, LONG_EYE_CLOSURE_COUNT, EYE_CLOSED, cooldown_counter, eye_closed_frames, no_face_frames
    global total_no_face_count, start_time, total_frames

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        total_frames += 1
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Detect facial landmarks using dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            no_face_frames += 1
            if no_face_frames == NO_FACE_THRESHOLD:
                total_no_face_count += 1
                cv2.putText(frame, "WARNING: No face detected for 3 seconds!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            no_face_frames = 0  # Reset counter if face is detected

        for face in faces:
            landmarks = predictor(gray, face)

            # Extract eye landmarks for EAR calculation
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Detect eye blink
            if avg_ear < EAR_THRESHOLD:
                if not EYE_CLOSED and cooldown_counter == 0:
                    BLINK_COUNT += 1
                    EYE_CLOSED = True
                    cooldown_counter = BLINK_COOLDOWN  # Start cooldown
                eye_closed_frames += 1
                if eye_closed_frames == LONG_EYE_CLOSURE_THRESHOLD:
                    LONG_EYE_CLOSURE_COUNT += 1
            else:
                EYE_CLOSED = False
                if eye_closed_frames > 0:
                    eye_closed_frames = 0  # Reset counter only when eyes are open

            # Decrement cooldown counter
            if cooldown_counter > 0:
                cooldown_counter -= 1

            # Classify facial expression
            expression = classify_facial_expression(frame, face)
            
            # Calculate face angles using MediaPipe Face Mesh
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mesh = face_landmarks.landmark
                    angles = calculate_face_angle(mesh)
                    roll_status, yaw_status, pitch_status = classify_head_pose(angles)
                    update_pose_counters(roll_status, yaw_status, pitch_status)

                    # Display results
                    cv2.putText(frame, f"Blinks: {BLINK_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Long Eye Closures: {LONG_EYE_CLOSURE_COUNT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Expression: {expression}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Roll: {roll_status}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Yaw: {yaw_status}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Pitch: {pitch_status}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Video Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Calculate duration and display final statistics
    duration_seconds = time.time() - start_time
    cap.release()
    cv2.destroyAllWindows()
    display_final_statistics(duration_seconds)

# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze head movements, eye blinking, and facial expressions.")
    parser.add_argument("--input", type=str, default="webcam", help="Path to video file or 'webcam' for live recording.")
    args = parser.parse_args()

    # Determine input source
    if args.input == "webcam":
        print("Starting live webcam recording...")
        process_video(0)  # 0 for default webcam
    else:
        print(f"Analyzing video file: {args.input}")
        process_video(args.input)