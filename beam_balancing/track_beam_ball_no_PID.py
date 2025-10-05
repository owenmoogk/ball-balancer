import cv2
import numpy as np
import serial
import time
import math


class BeamBallTracker:
    def __init__(self, params):
        self.camera_index = params["camera_index"]
        self.serial_port = params["serial_port"]
        self.baud_rate = params["baud_rate"]

        # --- RGB thresholds ---
        self.lower_ball_rgb = np.array(params["lower_ball_rgb"], dtype=np.uint8)
        self.upper_ball_rgb = np.array(params["upper_ball_rgb"], dtype=np.uint8)
        self.lower_red_rgb = np.array(params["lower_red_rgb"], dtype=np.uint8)
        self.upper_red_rgb = np.array(params["upper_red_rgb"], dtype=np.uint8)

        # Beam temporal stabilization
        self.prev_p1 = None
        self.prev_p2 = None
        self.nominal_length = None
        self.alpha = 0.3

        self.cap = None
        self.arduino = None
        self.connect_camera()
        self.connect_arduino()

    # ----------------------------------------------------
    def connect_camera(self):
        while True:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                print("[INFO] Camera connected.")
                return
            print("[ERROR] No camera detected. Retrying...")
            time.sleep(1)

    def connect_arduino(self):
        if not self.serial_port:
            print("[INFO] No serial port specified. Skipping Arduino connection.")
            return
        while True:
            try:
                self.arduino = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
                time.sleep(2)
                print("[INFO] Arduino connected.")
                return
            except serial.SerialException:
                print("[ERROR] Arduino not detected. Retrying...")
                time.sleep(1)

    # ----------------------------------------------------
    def segment_beam(self, frame):
        """Detect red beam, handle breaks, smooth endpoints."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(rgb, self.lower_red_rgb, self.upper_red_rgb)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            return mask, frame, None, None, None

        pts = np.column_stack((xs, ys)).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)

        diffs = pts - np.array([x0, y0])
        proj = diffs @ np.array([vx, vy])
        t_min, t_max = np.percentile(proj, [1, 99])

        p1 = np.array([x0 + t_min * vx, y0 + t_min * vy])
        p2 = np.array([x0 + t_max * vx, y0 + t_max * vy])
        curr_len = np.linalg.norm(p2 - p1)

        if self.nominal_length is None:
            self.nominal_length = curr_len
        else:
            if abs(curr_len - self.nominal_length) / self.nominal_length > 0.15:
                p1 = self.prev_p1 if self.prev_p1 is not None else p1
                p2 = self.prev_p2 if self.prev_p2 is not None else p2
            else:
                self.nominal_length = 0.9 * self.nominal_length + 0.1 * curr_len

        if self.prev_p1 is not None:
            p1 = self.alpha * p1 + (1 - self.alpha) * self.prev_p1
            p2 = self.alpha * p2 + (1 - self.alpha) * self.prev_p2

        self.prev_p1, self.prev_p2 = p1, p2
        p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))

        annotated = frame.copy()
        cv2.line(annotated, p1, p2, (0, 0, 255), 3)
        cv2.circle(annotated, p1, 6, (0, 255, 255), -1)
        cv2.circle(annotated, p2, 6, (0, 255, 255), -1)

        return mask, annotated, p1, p2, np.array([vx, vy])

    # ----------------------------------------------------
    def segment_ball(self, frame, annotated):
        """Detect orange ball (centroid only)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(rgb, self.lower_ball_rgb, self.upper_ball_rgb)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask, annotated, None

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return mask, annotated, None
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        cv2.circle(annotated, (cx, cy), 12, (0, 140, 255), -1)
        return mask, annotated, np.array([cx, cy])

    # ----------------------------------------------------
    def run(self):
        print("[INFO] Starting tracker. Press 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Frame read failed. Reconnecting...")
                self.connect_camera()
                continue

            frame = cv2.flip(frame, 1)

            beam_mask, annotated, p1, p2, direction = self.segment_beam(frame)
            if p1 is None or p2 is None:
                cv2.imshow("Full Feed", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            ball_mask, annotated, ball_center = self.segment_ball(frame, annotated)
            if ball_center is not None:
                # Project ball onto beam line
                beam_vec = (np.array(p2) - np.array(p1)).astype(np.float32)
                ball_proj = np.dot(ball_center - np.array(p1), beam_vec) / np.dot(beam_vec, beam_vec)
                pos = np.clip(ball_proj, 0, 1)
                print(f"Ball position: {pos:.3f}")  # terminal output

            combined_mask = cv2.bitwise_or(beam_mask, ball_mask)
            mask_view = cv2.bitwise_and(frame, frame, mask=combined_mask)

            cv2.imshow("Full Feed", annotated)
            cv2.imshow("Mask View", mask_view)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cleanup()

    # ----------------------------------------------------
    def cleanup(self):
        if self.cap:
            self.cap.release()
        if self.arduino:
            self.arduino.close()
        cv2.destroyAllWindows()
        print("[INFO] Tracker stopped.")


# ===========================================================
if __name__ == "__main__":
    params = {
        "camera_index": 0,
        "serial_port": None,
        "baud_rate": 115200,

        "lower_ball_rgb": (170, 80, 0),
        "upper_ball_rgb": (250, 175, 80),
        "lower_red_rgb": (190, 40, 50),
        "upper_red_rgb": (255, 90, 90),
    }

    tracker = BeamBallTracker(params)
    tracker.run()
