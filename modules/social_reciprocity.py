import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def analyze(video_path):
    cap = cv2.VideoCapture(video_path)
    response_detected = 0
    total_prompts = 0

    with mp_pose.Pose(static_image_mode=False) as pose:
        prev_head = None
        prev_time = None
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        still_counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                nose = (lm[0].x, lm[0].y)  # Nose as proxy for head

                if prev_head:
                    movement = np.linalg.norm(np.array(nose) - np.array(prev_head))

                    if movement < 0.005:
                        still_counter += 1
                    else:
                        if 10 < still_counter < 40:  # If still for ~0.5–1.5 sec
                            total_prompts += 1
                            if movement > 0.02:
                                response_detected += 1
                        still_counter = 0

                prev_head = nose

    cap.release()
    if total_prompts == 0:
        return 0.0
    return round(response_detected / total_prompts, 2)
