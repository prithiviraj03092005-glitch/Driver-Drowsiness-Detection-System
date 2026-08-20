# main.py – Hybrid Drowsiness + PnP Head Pose + Calibration + Alarm + Dashboard + Light CPU Optimization

import os
import time
import threading
import json
from collections import deque, defaultdict
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

MODEL_PATH = "model/hybrid_eye_model.h5"
ALARM_PATH = "sounds/alarm.mp3"
CALIB_FILE = "calibration.json"
LOGFILE = "logs/drowsiness.log"
os.makedirs("logs", exist_ok=True)

EYE_PAD = 8
SMOOTH_WINDOW = 5
SMOOTH_EAR_WINDOW = 5
DEFAULT_EAR_THRESHOLD = 0.20
FRAME_SKIP = 1  # set 2 or 3 later if you want more FPS

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

SHARED_STATE = {
    "camera_ok": False,
    "fps": 0.0,
    "frame_size": (0, 0),
    "last_update": None,
    "faces": {},
    "calibrated": False,
    "ear_threshold": DEFAULT_EAR_THRESHOLD,
    "alarm": False,
    "pitch0": 0.0,
    "yaw0": 0.0,
    "model_ok": False
}

_detector_thread = None
_stop_event = threading.Event()
_latest_jpeg = None
_latest_jpeg_lock = threading.Lock()
_state_lock = threading.Lock()


def update_state(key, value):
    with _state_lock:
        SHARED_STATE[key] = value


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def load_calibration():
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            with _state_lock:
                SHARED_STATE["ear_threshold"] = float(data.get("ear_threshold", DEFAULT_EAR_THRESHOLD))
                SHARED_STATE["pitch0"] = float(data.get("pitch0", 0.0))
                SHARED_STATE["yaw0"] = float(data.get("yaw0", 0.0))
                SHARED_STATE["calibrated"] = True
            log(f"Loaded calibration: {data}")
            return data
        except Exception as e:
            log(f"Failed to load calibration.json: {e}")
    return None


def save_calibration(data):
    try:
        with _state_lock:
            current = {
                "ear_threshold": SHARED_STATE["ear_threshold"],
                "pitch0": SHARED_STATE["pitch0"],
                "yaw0": SHARED_STATE["yaw0"]
            }
        for k in current:
            if k in data:
                current[k] = float(data[k])
        with open(CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        with _state_lock:
            SHARED_STATE["ear_threshold"] = current["ear_threshold"]
            SHARED_STATE["pitch0"] = current["pitch0"]
            SHARED_STATE["yaw0"] = current["yaw0"]
            SHARED_STATE["calibrated"] = True
        log(f"Saved calibration: {current}")
        return True
    except Exception as e:
        log(f"Failed to save calibration.json: {e}")
        return False


def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def compute_ear(landmarks, idx, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in idx]
    A = euclid(pts[1], pts[5])
    B = euclid(pts[2], pts[4])
    C = max(euclid(pts[0], pts[3]), 1e-6)
    return (A + B) / (2.0 * C), pts


def extract_eye_roi(frame, pts, pad=EYE_PAD, target_size=(80, 80), channels=3):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    h, w = frame.shape[:2]
    x1 = max(min(xs) - pad, 0)
    y1 = max(min(ys) - pad, 0)
    x2 = min(max(xs) + pad, w)
    y2 = min(max(ys) + pad, h)
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 6 or patch.shape[1] < 6:
        return None, (x1, y1, x2, y2)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    patch = cv2.resize(patch, target_size).astype("float32") / 255.0
    if channels == 1:
        patch = cv2.cvtColor((patch * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)[..., None] / 255.0
    return patch, (x1, y1, x2, y2)


# Load CNN
if os.path.exists(MODEL_PATH):
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH)
        log(f"Loaded CNN model from {MODEL_PATH}")
        SHARED_STATE["model_ok"] = True
    except Exception as e:
        log(f"Error loading model: {e}")
        MODEL = None
        SHARED_STATE["model_ok"] = False
else:
    log(f"Model path not found: {MODEL_PATH}")
    MODEL = None
    SHARED_STATE["model_ok"] = False

if MODEL is not None:
    try:
        _, H_in, W_in, C_in = MODEL.input_shape
        MODEL_INPUT_SIZE = (H_in, W_in)
        MODEL_CHANNELS = C_in
        MODEL_IS_SIGMOID = (len(MODEL.output_shape) == 2 and MODEL.output_shape[-1] == 1)
    except Exception:
        MODEL_INPUT_SIZE = (80, 80)
        MODEL_CHANNELS = 3
        MODEL_IS_SIGMOID = True
else:
    MODEL_INPUT_SIZE = (80, 80)
    MODEL_CHANNELS = 3
    MODEL_IS_SIGMOID = True


try:
    from src.alarm import play_alarm
except ImportError:
    def play_alarm(path=ALARM_PATH):
        return


def _detector_loop(cam=0):
    global _latest_jpeg
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        log("Camera cannot be opened.")
        update_state("camera_ok", False)
        return

    update_state("camera_ok", True)
    load_calibration()

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    smooth_cnn = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
    smooth_ear = defaultdict(lambda: deque(maxlen=SMOOTH_EAR_WINDOW))
    closed_count = defaultdict(int)

    head_pitch_sum = 0.0
    head_yaw_sum = 0.0
    max_ear_list = []
    head_calib_frames = 0
    N_CAL = 50

    frames = 0
    last_time = time.time()
    frame_index = 0

    while not _stop_event.is_set():
        ok, frame = cap.read()
        if not ok:
            log("Frame read failed.")
            break

        frames += 1
        frame_index += 1
        h, w = frame.shape[:2]
        update_state("frame_size", (w, h))

        # Basic frame skip (if you later set FRAME_SKIP > 1)
        if FRAME_SKIP > 1 and frame_index % FRAME_SKIP != 0:
            # still encode for MJPEG
            ok_enc, enc = cv2.imencode(".jpg", frame)
            if ok_enc:
                with _latest_jpeg_lock:
                    _latest_jpeg = enc.tobytes()
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        faces = {}

        if results.multi_face_landmarks:
            for fid, lm in enumerate(results.multi_face_landmarks):
                ear_l, pts_l = compute_ear(lm.landmark, LEFT_EYE, w, h)
                ear_r, pts_r = compute_ear(lm.landmark, RIGHT_EYE, w, h)
                ear = (ear_l + ear_r) / 2.0
                smooth_ear[fid].append(ear)
                ear = sum(smooth_ear[fid]) / len(smooth_ear[fid])

                cnn_vals = []
                cnn = 1.0  # assume eyes open if we skip CNN

                # CPU optimization: only run CNN if EAR is low (possible closing)
                if MODEL is not None and ear < SHARED_STATE["ear_threshold"] + 0.08:
                    lp, _ = extract_eye_roi(frame, pts_l, target_size=MODEL_INPUT_SIZE, channels=MODEL_CHANNELS)
                    rp, _ = extract_eye_roi(frame, pts_r, target_size=MODEL_INPUT_SIZE, channels=MODEL_CHANNELS)

                    def cnn_pred(patch):
                        # Convert to tensor and call directly for speed (avoid .predict overhead)
                        input_tensor = tf.convert_to_tensor(np.expand_dims(patch, 0), dtype=tf.float32)
                        p = MODEL(input_tensor, training=False).numpy()
                        if MODEL_IS_SIGMOID:
                            return float(p[0][0])
                        if p.shape[-1] == 2:
                            return float(p[0][1])
                        return float(p[0].flatten()[0])

                    try:
                        if lp is not None: cnn_vals.append(cnn_pred(lp))
                        if rp is not None: cnn_vals.append(cnn_pred(rp))
                    except Exception as e:
                        log(f"CNN predict error: {e}")

                    if cnn_vals:
                        cnn = sum(cnn_vals) / len(cnn_vals)

                smooth_cnn[fid].append(cnn)
                cnn = sum(smooth_cnn[fid]) / len(smooth_cnn[fid])

                # Head pose PnP
                lm2 = lm.landmark
                img_pts = np.array([
                    (lm2[1].x * w,   lm2[1].y * h),
                    (lm2[152].x * w, lm2[152].y * h),
                    (lm2[33].x * w,  lm2[33].y * h),
                    (lm2[263].x * w, lm2[263].y * h),
                    (lm2[61].x * w,  lm2[61].y * h),
                    (lm2[291].x * w, lm2[291].y * h)
                ], dtype=np.float64)
                cam_mat = np.array([[w, 0, w/2],[0, w, h/2],[0, 0, 1]], dtype=np.float64)
                dist = np.zeros((4,1))

                try:
                    ok_pnp, rv, tv = cv2.solvePnP(MODEL_POINTS, img_pts, cam_mat, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                    rot, _ = cv2.Rodrigues(rv)
                    mat = cv2.hconcat((rot, tv))
                    _, _, _, _, _, _, ang = cv2.decomposeProjectionMatrix(mat)
                    pitch, yaw, roll = map(float, ang.flatten())
                except Exception as e:
                    log(f"Head pose error: {e}")
                    pitch = yaw = roll = 0.0

                # Auto baseline
                if head_calib_frames < N_CAL and not SHARED_STATE["alarm"]:
                    head_pitch_sum += pitch
                    head_yaw_sum += yaw
                    max_ear_list.append(ear)
                    head_calib_frames += 1
                    if head_calib_frames == N_CAL:
                        p0 = head_pitch_sum / N_CAL
                        y0 = head_yaw_sum / N_CAL
                        # Set EAR threshold to 75% of the average open EAR seen during calib
                        avg_open_ear = sum(max_ear_list) / len(max_ear_list)
                        new_ear_th = max(0.15, min(0.30, avg_open_ear * 0.75))
                        
                        update_state("pitch0", p0)
                        update_state("yaw0", y0)
                        update_state("ear_threshold", new_ear_th)
                        
                        save_calibration({
                            "pitch0": p0,
                            "yaw0": y0,
                            "ear_threshold": new_ear_th
                        })
                        log(f"Auto-calib complete: EAR_TH={new_ear_th:.3f}, Pitch0={p0:.1f}, Yaw0={y0:.1f}")

                pitch -= SHARED_STATE["pitch0"]
                yaw -= SHARED_STATE["yaw0"]

                # Normalize for compass
                pitch = max(-30, min(30, pitch / 3.0))
                yaw   = max(-30, min(30, yaw / 3.0))

                # Hybrid drowsiness
                with _state_lock:
                    th = SHARED_STATE["ear_threshold"]
                is_geom = ear < th
                
                if MODEL is not None:
                    is_tex = cnn < 0.5
                    # Hybrid logic: both must agree it's eyes-closed to avoid false positives from EAR noise
                    # BUT if EAR is extremely low (<0.12), we should trust it regardless of CNN
                    drowsy = bool((is_geom and is_tex) or (ear < 0.12))
                else:
                    # Fallback to EAR-only if model is missing
                    drowsy = is_geom

                if drowsy:
                    closed_count[fid] += 1
                else:
                    closed_count[fid] = 0

                with _state_lock:
                    fps = SHARED_STATE["fps"]
                SHARED_STATE["alarm"] = closed_count[fid] >= max(int(fps * 2), 5)

                faces[str(fid)] = {
                    "face_id": fid,
                    "ear": float(ear),
                    "cnn_open_prob": float(cnn),
                    "drowsy": drowsy,
                    "closed_frames": int(closed_count[fid]),
                    "head_pose": {"pitch": pitch, "yaw": yaw, "roll": roll}
                }

                # draw eyes
                color = (0, 0, 255) if drowsy else (0, 255, 0)
                for pt in pts_l + pts_r:
                    cv2.circle(frame, tuple(pt), 1, color, -1)

        with _state_lock:
            SHARED_STATE["faces"] = faces
            SHARED_STATE["last_update"] = datetime.utcnow().isoformat()

        if SHARED_STATE["alarm"]:
            play_alarm(ALARM_PATH)

        ok_enc, enc = cv2.imencode(".jpg", frame)
        if ok_enc:
            with _latest_jpeg_lock:
                _latest_jpeg = enc.tobytes()

        now = time.time()
        if now - last_time >= 1.0:
            update_state("fps", frames / max(now - last_time, 1e-6))
            frames = 0
            last_time = now

    update_state("camera_ok", False)
    face_mesh.close()
    cap.release()
    log("Detector thread exiting.")


def start_detector(cam=0):
    global _detector_thread
    if _detector_thread and _detector_thread.is_alive():
        return True
    _stop_event.clear()
    _detector_thread = threading.Thread(target=_detector_loop, args=(cam,), daemon=True)
    _detector_thread.start()
    time.sleep(0.5)
    return True


def stop_detector():
    _stop_event.set()
    if _detector_thread:
        _detector_thread.join(timeout=2.0)


def get_mjpeg_frame():
    with _latest_jpeg_lock:
        return _latest_jpeg


def run_frame():
    frame = get_mjpeg_frame()
    with _state_lock:
        state = SHARED_STATE.copy()
    return frame, state