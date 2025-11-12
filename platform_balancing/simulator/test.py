import cv2
import os

def get_camera_index_by_name(name_substring):
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            backend = cap.getBackendName()
            # Try to get device path (Linux/Mac only)
            if hasattr(cap, 'get'):
                path = f"/dev/video{i}"
                if os.path.exists(path):
                    if name_substring.lower() in os.popen(f"udevadm info --query=all --name={path}").read().lower():
                        cap.release()
                        return i
            cap.release()
    raise RuntimeError("Camera not found")

idx = get_camera_index_by_name("logitech")
cap = cv2.VideoCapture(idx)