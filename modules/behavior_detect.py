import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def get_wrist_trajectory(video_path):
    cap = cv2.VideoCapture(video_path)
    wrist_y_positions = []

    with mp_pose.Pose(static_image_mode=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                # Right wrist landmark = index 16
                wrist = result.pose_landmarks.landmark[16]
                wrist_y_positions.append(wrist.y)

    cap.release()
    return np.array(wrist_y_positions)
def detect_repetitive_behavior(wrist_y_positions):
    if len(wrist_y_positions) < 10:
        return 0.0  # Not enough data

    # Calculate changes in movement
    diffs = np.diff(wrist_y_positions)
    movement_strength = np.std(diffs)

    # Heuristic: strong and frequent up-down motion → repetitive
    if movement_strength > 0.01:
        return 1.0
    return 0.0
def analyze(video_path):
    wrist_traj = get_wrist_trajectory(video_path)
    rep_score = detect_repetitive_behavior(wrist_traj)
    return rep_score
