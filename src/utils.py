import cv2

def resize_frame(frame, width=640):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, None, fx=scale, fy=scale)