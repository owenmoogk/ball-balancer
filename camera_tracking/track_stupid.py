#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import csv
import math
import sys

# ---------------- CONFIG ----------------
CALIB_FILE = "/Users/brendanchharawala/Documents/GitHub/ball-balancer/camera_tracking/calibration_and_generation/camera_calibration.yaml"
TAG_CSV = "/Users/brendanchharawala/Documents/GitHub/ball-balancer/camera_tracking/calibration_and_generation/generated/tags_positions.csv"

DOT_RADIUS = 10
AXIS_LENGTH = 150  # pixels for display

tol_ball_low = -20; tol_ball_high = 15
tol_ring_low = -10; tol_ring_high = 10

BALL_LOWER_BGR = np.array([max(50+tol_ball_low,0), max(156+tol_ball_low,0), max(220+tol_ball_low,0)], dtype=np.uint8)
BALL_UPPER_BGR = np.array([min(75+tol_ball_high,255), min(205+tol_ball_high,255), min(252+tol_ball_high,255)], dtype=np.uint8)
RING_LOWER_BGR = np.array([max(33+tol_ring_low,0), max(63+tol_ring_low,0), max(167+tol_ring_low,0)], dtype=np.uint8)
RING_UPPER_BGR = np.array([min(64+tol_ring_high,255), min(79+tol_ring_high,255), min(182+tol_ring_high,255)], dtype=np.uint8)

# ---------------- LOAD CAMERA CALIBRATION ----------------
with open(CALIB_FILE, "r") as f:
    calib = yaml.safe_load(f)
camera_matrix = np.array(calib["camera_matrix"], dtype=np.float64)
dist_coeffs = np.array(calib["dist_coeff"], dtype=np.float64)
cx, cy = camera_matrix[0,2], camera_matrix[1,2]

# ---------------- LOAD TAG POSITIONS ----------------
tag_positions = {}
with open(TAG_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tag_positions[int(row["tag_id"])] = np.array([float(row["x_mm"]), float(row["y_mm"])])

# ---------------- ARUCO INIT ----------------
cap = cv2.VideoCapture(0)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters_create()

# ---------------- HELPERS ----------------
def detect_ball(frame):
    mask = cv2.inRange(frame, BALL_LOWER_BGR, BALL_UPPER_BGR)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8), iterations=1)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None, mask
    cx_px = int(M['m10']/M['m00']); cy_px = int(M['m01']/M['m00'])
    return (cx_px, cy_px), mask

def detect_ring(frame):
    mask = cv2.inRange(frame, RING_LOWER_BGR, RING_UPPER_BGR)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    pts = np.vstack(contours).squeeze()
    if pts.ndim != 2 or pts.shape[0] < 3:
        return None, mask
    A = np.c_[2*pts[:,0], 2*pts[:,1], np.ones(pts.shape[0])]
    b = pts[:,0]**2 + pts[:,1]**2
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = x[0], x[1]
        return (int(cx), int(cy)), mask
    except np.linalg.LinAlgError:
        return None, mask

def draw_axes(frame, visible_tags_px, center_px):
    axis_pairs = [(0,2), (1,3)]
    pc = np.array(center_px)
    for plus_id, minus_id in axis_pairs:
        vec = None
        if plus_id in visible_tags_px and minus_id in visible_tags_px:
            vec = np.array(visible_tags_px[plus_id]) - np.array(visible_tags_px[minus_id])
        elif plus_id in visible_tags_px:
            vec = np.array(visible_tags_px[plus_id]) - pc
        elif minus_id in visible_tags_px:
            vec = pc - np.array(visible_tags_px[minus_id])
        if vec is not None:
            norm_vec = vec / np.linalg.norm(vec) * AXIS_LENGTH
            color = (0,0,255) if plus_id in [0,2] else (0,255,0)
            cv2.arrowedLine(frame, tuple(pc.astype(int)), tuple((pc+norm_vec).astype(int)), color, 2, tipLength=0.1)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_undist = cv2.undistort(frame, camera_matrix, dist_coeffs)
    display = frame_undist.copy()

    # Detect AprilTags
    corners, ids, _ = cv2.aruco.detectMarkers(frame_undist, aruco_dict, parameters=parameters)
    visible_tags_px = {}
    if ids is not None:
        for i, tid in enumerate(ids.flatten()):
            tid = int(tid)
            if tid not in tag_positions:
                continue
            c = corners[i][0]
            center_px = tuple(c.mean(axis=0).astype(int))
            visible_tags_px[tid] = center_px
            for pt in c.astype(int):
                cv2.circle(display, tuple(pt), 3, (0,200,200), -1)
            cv2.putText(display, str(tid), (center_px[0]+4, center_px[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # Draw reference cross
    if 3 in visible_tags_px and 1 in visible_tags_px:
        cv2.line(display, visible_tags_px[3], visible_tags_px[1], (255,0,0),2)
    if 0 in visible_tags_px and 2 in visible_tags_px:
        cv2.line(display, visible_tags_px[0], visible_tags_px[2], (255,0,0),2)

    # Detect ball and ring
    ring_px, ring_mask = detect_ring(frame_undist)
    ball_px, ball_mask = detect_ball(frame_undist)

    # Compute averaged plate center using ring and tag midpoints
    centers = []
    if ring_px is not None:
        centers.append(ring_px)
    if 0 in visible_tags_px and 2 in visible_tags_px:
        centers.append(((visible_tags_px[0][0]+visible_tags_px[2][0])//2,
                        (visible_tags_px[0][1]+visible_tags_px[2][1])//2))
    if 1 in visible_tags_px and 3 in visible_tags_px:
        centers.append(((visible_tags_px[1][0]+visible_tags_px[3][0])//2,
                        (visible_tags_px[1][1]+visible_tags_px[3][1])//2))

    plate_center = None
    if centers:
        avg_x = int(np.mean([c[0] for c in centers]))
        avg_y = int(np.mean([c[1] for c in centers]))
        plate_center = (avg_x, avg_y)
        cv2.circle(display, plate_center, 8, (0,255,255), -1)

    # Draw coordinate axes
    if plate_center is not None:
        draw_axes(display, visible_tags_px, plate_center)

    # --- Draw ball and compute coordinates in plate frame ---
    pixel_xy = (0,0)
    mm_xy = (0,0)
    r_theta = (0,0)
    if ball_px is not None and plate_center is not None:
        cv2.circle(display, ball_px, 6, (255,0,0), -1)
        offset_img = np.array(ball_px) - np.array(plate_center)

        # --- Plate frame axes in pixels ---
        def unit_vector(vec): return vec / np.linalg.norm(vec)

        if 0 in visible_tags_px and 2 in visible_tags_px:
            x_axis = unit_vector(np.array(visible_tags_px[0]) - np.array(visible_tags_px[2]))
        elif 0 in visible_tags_px:
            x_axis = unit_vector(np.array(visible_tags_px[0]) - np.array(plate_center))
        elif 2 in visible_tags_px:
            x_axis = unit_vector(np.array(plate_center) - np.array(visible_tags_px[2]))
        else:
            x_axis = np.array([1.0,0.0])

        if 1 in visible_tags_px and 3 in visible_tags_px:
            y_axis = unit_vector(np.array(visible_tags_px[1]) - np.array(visible_tags_px[3]))
        elif 1 in visible_tags_px:
            y_axis = unit_vector(np.array(visible_tags_px[1]) - np.array(plate_center))
        elif 3 in visible_tags_px:
            y_axis = unit_vector(np.array(plate_center) - np.array(visible_tags_px[3]))
        else:
            y_axis = np.array([0.0,1.0])

        # --- Project ball to plate axes (pixel space) ---
        plate_x_px = np.dot(offset_img, x_axis)
        plate_y_px = np.dot(offset_img, y_axis)
        pixel_xy = (plate_x_px, plate_y_px)

        # --- Convert pixel distances to mm using tag_positions ---
        scale_x = 1.0; scale_y = 1.0
        if 0 in visible_tags_px and 2 in visible_tags_px:
            dist_px = np.linalg.norm(np.array(visible_tags_px[0])-np.array(visible_tags_px[2]))
            dist_mm = np.linalg.norm(tag_positions[0]-tag_positions[2])
            scale_x = dist_mm / dist_px
        if 1 in visible_tags_px and 3 in visible_tags_px:
            dist_px = np.linalg.norm(np.array(visible_tags_px[1])-np.array(visible_tags_px[3]))
            dist_mm = np.linalg.norm(tag_positions[1]-tag_positions[3])
            scale_y = dist_mm / dist_px

        mm_x = plate_x_px * scale_x
        mm_y = plate_y_px * scale_y
        mm_xy = (mm_x, mm_y)

        # --- Polar coordinates in mm ---
        r = math.hypot(mm_x, mm_y)
        theta = math.degrees(math.atan2(mm_y, mm_x))
        r_theta = (r, theta)

    # --- Terminal print ---
    sys.stdout.write(f"\rPlane XY(mm): ({mm_xy[0]:.1f}, {mm_xy[1]:.1f}) | rθ(mm,°): ({r_theta[0]:.1f}, {r_theta[1]:.1f}°) ")
    sys.stdout.flush()

    cv2.imshow("Plate Tracker", display)
    if ball_mask is not None:
        cv2.imshow("Ball Mask", ball_mask)
    if ring_mask is not None:
        cv2.imshow("Ring Mask", ring_mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()