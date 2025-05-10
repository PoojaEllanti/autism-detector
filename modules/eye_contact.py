import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def get_eye_landmarks(face_landmarks, image_shape):
    h, w, _ = image_shape
    left_eye_idx = [33, 133]  # Approx corners of left eye
    right_eye_idx = [362, 263]  # Approx corners of right eye

    left = [(int(face_landmarks.landmark[i].x * w),
             int(face_landmarks.landmark[i].y * h)) for i in left_eye_idx]
    right = [(int(face_landmarks.landmark[i].x * w),
              int(face_landmarks.landmark[i].y * h)) for i in right_eye_idx]
    return left, right

def analyze(video_path):
    cap = cv2.VideoCapture(video_path)
    eye_contact_frames = 0
    total_frames = 0

    with mp_face_mesh.FaceMesh(static_image_mode=False) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    left_eye, right_eye = get_eye_landmarks(face_landmarks, frame.shape)

                    # Simple center approximation of gaze direction
                    eye_center_x = (left_eye[0][0] + right_eye[1][0]) // 2
                    screen_center_x = frame.shape[1] // 2

                    if abs(eye_center_x - screen_center_x) < frame.shape[1] * 0.2:
                        eye_contact_frames += 1

            if total_frames % 10 == 0:
                print(f"Processed {total_frames} frames...")

    cap.release()
    eye_contact_ratio = eye_contact_frames / total_frames if total_frames else 0
    return round(eye_contact_ratio, 2)
