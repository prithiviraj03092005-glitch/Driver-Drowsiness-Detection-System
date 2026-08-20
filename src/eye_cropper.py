import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh

def get_eye_landmarks(frame):
    with mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark

        # Left & Right eye indexes (MediaPipe)
        LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

        left_eye = [(landmarks[i].x, landmarks[i].y) for i in LEFT_EYE_IDX]
        right_eye = [(landmarks[i].x, landmarks[i].y) for i in RIGHT_EYE_IDX]

        return left_eye, right_eye