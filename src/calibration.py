import cv2
import mediapipe as mp
import numpy as np
import json
import os

CALIB_PATH = "calibration.json" # Match main.py's file

# 3D model points for head pose estimation (consistent with main.py)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

def get_head_pose(landmarks, frame_size):
    """Returns yaw, pitch, roll using Mediapipe landmarks"""
    h, w = frame_size
    img_pts = np.array([
        (landmarks[1].x * w,   landmarks[1].y * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[33].x * w,  landmarks[33].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[61].x * w,  landmarks[61].y * h),
        (landmarks[291].x * w, landmarks[291].y * h)
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))
    success, rv, tv = cv2.solvePnP(MODEL_POINTS, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    rot_mat, _ = cv2.Rodrigues(rv)
    pose_mat = cv2.hconcat((rot_mat, tv))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()

    return yaw, pitch, roll

def calibrate_head_pose(video_index=0):
    """Calibrates head pose and saves to calibration.json"""
    cap = cv2.VideoCapture(video_index)
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

    print("LOOK STRAIGHT AND PRESS 'C' TO CALIBRATE")

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            yaw, pitch, roll = get_head_pose(lm, frame.shape[:2])

            cv2.putText(frame, "PRESS C TO CALIBRATE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f} Pitch: {pitch:.1f} Roll: {roll:.1f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if cv2.waitKey(1) == ord('c'):
                data = {"yaw0": float(yaw), "pitch0": float(pitch)}
                with open(CALIB_PATH, "w") as f:
                    json.dump(data, f, indent=4)
                print("CALIBRATION SAVED:", data)
                break

        cv2.imshow("Mediapipe Calibration", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    calibrate_head_pose()