import cv2
import dlib
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import argparse

# Initialize MediaPipe Face Mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file

# Constants for EAR (Eye Aspect Ratio)
EAR_THRESHOLD = 0.20  # Threshold for eye blink detection
BLINK_COUNT = 0
LONG_EYE_CLOSURE_COUNT = 0
EYE_CLOSED = False  # Track if eyes are currently closed
BLINK_COOLDOWN = 10  # Cooldown frames between blinks
cooldown_counter = 0  # Counter for cooldown
LONG_EYE_CLOSURE_THRESHOLD = 30  # Minimum frames for long eye closure
eye_closed_frames = 0  # Track consecutive frames with eyes closed

# Function to calculate EAR
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to estimate head pose
def estimate_head_pose(landmarks, frame):
    if landmarks is None or len(landmarks) != 6:
        print("Invalid or insufficient landmarks detected.")
        return None

    # 3D model points (generic head model)
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])

    # Camera intrinsic parameters (assumed)
    focal_length = frame.shape[1]
    center = (frame.shape[1] / 2, frame.shape[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    # Solve PnP to estimate head pose
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    try:
        # Ensure landmarks are in the correct format (float32 or float64)
        landmarks = np.array(landmarks, dtype=np.float32)
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, landmarks, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            print("Failed to solve PnP.")
            return None
    except Exception as e:
        print(f"Error in solvePnP: {e}")
        return None

    # Calculate head angle
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    head_angle = angles[1]  # Y-axis rotation (left/right tilt)

    return head_angle

# Function to classify facial expression (dummy implementation)
def classify_facial_expression(face_roi):
    # Placeholder for a real facial expression model
    # Replace this with a pre-trained model (e.g., using TensorFlow or PyTorch)
    return "Neutral"  # Replace with actual expression detection

# Function to process a video file or live webcam feed
def process_video(input_source):
    global BLINK_COUNT, LONG_EYE_CLOSURE_COUNT, EYE_CLOSED, cooldown_counter, eye_closed_frames

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Detect facial landmarks using dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
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
                if eye_closed_frames >= LONG_EYE_CLOSURE_THRESHOLD:
                    LONG_EYE_CLOSURE_COUNT += 1
                    eye_closed_frames = 0  # Reset counter after counting long closure
            else:
                EYE_CLOSED = False
                eye_closed_frames = 0  # Reset counter if eyes are open

            # Decrement cooldown counter
            if cooldown_counter > 0:
                cooldown_counter -= 1

            # Use specific landmarks for head pose estimation
            landmark_indices = [30, 8, 36, 45, 48, 54]  # Nose tip, chin, left eye, right eye, left mouth, right mouth
            landmarks_for_pose = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_indices])

            # Estimate head pose
            head_angle = estimate_head_pose(landmarks_for_pose, frame)

            # Classify facial expression
            face_roi = frame[face.top():face.bottom(), face.left():face.right()]
            expression = classify_facial_expression(face_roi)

            # Display results
            if head_angle is not None:
                cv2.putText(frame, f"Head Angle: {head_angle:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Head Angle: N/A", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks: {BLINK_COUNT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Long Eye Closures: {LONG_EYE_CLOSURE_COUNT}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Expression: {expression}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Video Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or LONG_EYE_CLOSURE_COUNT > 0:
            print("Stopping video.")
            break

    cap.release()
    cv2.destroyAllWindows()

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