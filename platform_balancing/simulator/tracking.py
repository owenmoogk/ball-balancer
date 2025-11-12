#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import csv
import math
import sys
from typing import Optional

class BallTracker:
    def __init__(
        self,
        calib_file: str,
        tag_csv: str,
        camera_index: int = 1,
        show_debug: bool = False
    ):
        self.show_debug = show_debug
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

        # --- Load calibration ---
        with open(calib_file, "r") as f:
            calib = yaml.safe_load(f)
        self.camera_matrix = np.array(calib["camera_matrix"], dtype=np.float64)
        self.dist_coeffs = np.array(calib["dist_coeff"], dtype=np.float64)

        # --- Load tag positions ---
        self.tag_positions = {}
        with open(tag_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.tag_positions[int(row["tag_id"])] = np.array([float(row["x_mm"]), float(row["y_mm"])])

        # --- Initialize ArUco ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # --- Color tolerances ---
        self.BALL_LOWER_BGR = np.array([30, 136, 200], dtype=np.uint8)
        self.BALL_UPPER_BGR = np.array([90, 220, 255], dtype=np.uint8)
        self.RING_LOWER_BGR = np.array([23, 53, 157], dtype=np.uint8)
        self.RING_UPPER_BGR = np.array([74, 89, 192], dtype=np.uint8)

    # ---------------- DETECTION HELPERS ----------------
    def _detect_ball(self, frame):
        mask = cv2.inRange(frame, self.BALL_LOWER_BGR, self.BALL_UPPER_BGR)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8), iterations=1)
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))

    def _detect_ring(self, frame):
        mask = cv2.inRange(frame, self.RING_LOWER_BGR, self.RING_UPPER_BGR)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        pts = np.vstack(contours).squeeze()
        if pts.ndim != 2 or pts.shape[0] < 3:
            return None
        A = np.c_[2*pts[:,0], 2*pts[:,1], np.ones(pts.shape[0])]
        b = pts[:,0]**2 + pts[:,1]**2
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            return (int(x[0]), int(x[1]))
        except np.linalg.LinAlgError:
            return None

    # ---------------- MAIN METHOD ----------------
    def get_position(self) -> Optional[np.ndarray]:
        """Return ball position [x_m, y_m] in platform frame (meters)."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame_undist = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        corners, ids, _ = cv2.aruco.detectMarkers(frame_undist, self.aruco_dict, parameters=self.parameters)
        visible_tags_px = {}
        if ids is not None:
            for i, tid in enumerate(ids.flatten()):
                tid = int(tid)
                if tid in self.tag_positions:
                    center_px = tuple(corners[i][0].mean(axis=0).astype(int))
                    visible_tags_px[tid] = center_px

        # Ring and ball
        ring_px = self._detect_ring(frame_undist)
        ball_px = self._detect_ball(frame_undist)
        if ball_px is None:
            return None

        # Plate center
        centers = []
        if ring_px is not None:
            centers.append(ring_px)
        if 0 in visible_tags_px and 2 in visible_tags_px:
            centers.append(((visible_tags_px[0][0]+visible_tags_px[2][0])//2,
                            (visible_tags_px[0][1]+visible_tags_px[2][1])//2))
        if 1 in visible_tags_px and 3 in visible_tags_px:
            centers.append(((visible_tags_px[1][0]+visible_tags_px[3][0])//2,
                            (visible_tags_px[1][1]+visible_tags_px[3][1])//2))

        if not centers:
            return None
        plate_center = np.mean(centers, axis=0)

        # --- Compute plate axes ---
        def unit(v):
            return v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-9 else v

        if 0 in visible_tags_px and 2 in visible_tags_px:
            x_axis = unit(np.array(visible_tags_px[0]) - np.array(visible_tags_px[2]))
        else:
            x_axis = np.array([1.0, 0.0])
        if 1 in visible_tags_px and 3 in visible_tags_px:
            y_axis = unit(np.array(visible_tags_px[1]) - np.array(visible_tags_px[3]))
        else:
            y_axis = np.array([0.0, 1.0])

        offset = np.array(ball_px) - plate_center
        x_px = np.dot(offset, x_axis)
        y_px = np.dot(offset, y_axis)

        # --- Pixel to mm scale ---
        scale_x = scale_y = 1.0
        if 0 in visible_tags_px and 2 in visible_tags_px:
            dist_px = np.linalg.norm(np.array(visible_tags_px[0]) - np.array(visible_tags_px[2]))
            dist_mm = np.linalg.norm(self.tag_positions[0] - self.tag_positions[2])
            scale_x = dist_mm / dist_px
        if 1 in visible_tags_px and 3 in visible_tags_px:
            dist_px = np.linalg.norm(np.array(visible_tags_px[1]) - np.array(visible_tags_px[3]))
            dist_mm = np.linalg.norm(self.tag_positions[1] - self.tag_positions[3])
            scale_y = dist_mm / dist_px

        x_mm = x_px * scale_x
        y_mm = y_px * scale_y

        if self.show_debug:
            display = frame_undist.copy()
            cv2.circle(display, tuple(np.int32(ball_px)), 6, (255,0,0), -1)
            cv2.circle(display, tuple(np.int32(plate_center)), 8, (0,255,255), -1)
            cv2.imshow("Ball Tracker", display)
            cv2.waitKey(1)

        # convert to meters
        return np.array([x_mm, y_mm]) / 1000.0

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
