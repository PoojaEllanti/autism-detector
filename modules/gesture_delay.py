import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def analyze(video_path):
    cap = cv2.VideoCapture(video_path)
    left_wrist_movements = []
    right_wrist_movements = []

    prev_left = None
    prev_right = None
    frame_count = 0
    gesture_count = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                lw = (lm[15].x, lm[15].y)  # Left wrist
                rw = (lm[16].x, lm[16].y)  # Right wrist

                if prev_left and prev_right:
                    # Euclidean movement of both wrists
                    l_move = np.linalg.norm(np.array(lw) - np.array(prev_left))
                    r_move = np.linalg.norm(np.array(rw) - np.array(prev_right))

                    if l_move > 0.02 or r_move > 0.02:
                        gesture_count += 1

                prev_left = lw
                prev_right = rw

    cap.release()

    gesture_rate = gesture_count / frame_count if frame_count else 0
    return round(gesture_rate, 3)
