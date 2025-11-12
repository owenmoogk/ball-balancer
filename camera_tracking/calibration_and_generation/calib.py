#!/usr/bin/env python3
"""
Manual AprilTag-only camera calibration using tag_positions.csv
Prompts user to capture images and saves camera matrix + distortion.
"""

import cv2
import numpy as np
import yaml
import csv
import os

# --- SETTINGS ---
CSV_FILE = "generated/tags_positions.csv"  # from your plate generator
TAG_SIZE_MM = 20.0                           # must match your generator
NUM_IMAGES = 15
OUTPUT_YAML = "camera_calibration.yaml"

# --- LOAD REAL TAG POSITIONS ---
tag_positions = {}  # id -> (x_mm, y_mm)
with open(CSV_FILE, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tag_id = int(row["tag_id"])
        x = float(row["x_mm"])
        y = float(row["y_mm"])
        tag_positions[tag_id] = (x, y)

# Prepare object points per tag
# Each tag has 4 corners in order: top-left, top-right, bottom-right, bottom-left
def tag_corners_3d(tag_id):
    x_c, y_c = tag_positions[tag_id]
    s = TAG_SIZE_MM / 2
    # Z=0 plane
    return np.array([
        [x_c - s, y_c - s, 0],
        [x_c + s, y_c - s, 0],
        [x_c + s, y_c + s, 0],
        [x_c - s, y_c + s, 0]
    ], dtype=np.float32)

# --- INIT ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters_create()

obj_points = []  # 3D points
img_points = []  # 2D points

print(f"Press SPACE to capture image ({NUM_IMAGES} total). ESC to quit.")

captured = 0

while captured < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        continue

    display = frame.copy()
    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
        cv2.putText(display, f"Detected IDs: {ids.flatten()}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
    cv2.imshow("AprilTag Detection", display)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        if ids is None:
            print("No tags detected. Move camera to see tags.")
            continue

        valid = True
        frame_obj_pts = []
        frame_img_pts = []
        for i, tag_id in enumerate(ids.flatten()):
            if tag_id not in tag_positions:
                print(f"Warning: Tag {tag_id} not in CSV, skipping")
                valid = False
                continue
            frame_obj_pts.append(tag_corners_3d(tag_id))
            frame_img_pts.append(corners[i].reshape(4,2))
        if not valid or len(frame_obj_pts) == 0:
            print("No valid tags in frame, try again.")
            continue

        obj_points.extend(frame_obj_pts)
        img_points.extend(frame_img_pts)
        captured += 1
        print(f"Captured image {captured}/{NUM_IMAGES}")

cap.release()
cv2.destroyAllWindows()

if len(obj_points) < 1:
    raise RuntimeError("No points captured, cannot calibrate")

# --- CALIBRATE CAMERA ---
# Convert to correct shape
obj_pts_all = [op.reshape(-1,3) for op in obj_points]
img_pts_all = [ip.reshape(-1,2) for ip in img_points]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_pts_all, img_pts_all, frame.shape[1::-1], None, None
)

print("Calibration complete")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())

# --- SAVE ---
with open(OUTPUT_YAML, "w") as f:
    yaml.safe_dump({
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist()
    }, f)

print(f"Saved calibration to {OUTPUT_YAML}")