import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time


class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def update(self, target, current):
        now = time.time()
        dt = now - self.last_time
        if dt <= 0:
            return 0.0
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        self.last_time = now
        return output



class MotorSerial:
    def __init__(self):
        self.ser = None

    def select_serial_port(self):
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("No serial devices found.")
            return None
        print("Available serial ports:")
        for i, p in enumerate(ports):
            print(f"[{i}] {p.device} — {p.description}")
        idx = input("Select port number: ")
        try:
            return ports[int(idx)].device
        except (ValueError, IndexError):
            return None

    def connect(self):
        port = self.select_serial_port()
        if not port:
            return
        try:
            self.ser = serial.Serial(port, 115200, timeout=1)
            time.sleep(2)
            print(f"[INFO] Connected to {port}")
        except Exception as e:
            print(f"[ERROR] Serial connection failed: {e}")

    def send_angle(self, angle):
        if self.ser:
            self.ser.write(f"{angle:.2f}\n".encode())

class BeamBallTracker:
    def __init__(self, params, motor, pid):
        self.camera_index = params["camera_index"]
        self.lower_ball_rgb = np.array(params["lower_ball_rgb"], dtype=np.uint8)
        self.upper_ball_rgb = np.array(params["upper_ball_rgb"], dtype=np.uint8)
        self.lower_red_rgb = np.array(params["lower_red_rgb"], dtype=np.uint8)
        self.upper_red_rgb = np.array(params["upper_red_rgb"], dtype=np.uint8)
        self.cap = None
        self.motor = motor
        self.pid = pid

        self.angle_min, self.angle_max = params["angle_min"], params["angle_max"]
        self.level_angle = params["level_angle"]
        self.target = 0.5  # center position

        self.connect_camera()

    def connect_camera(self):
        while True:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                print("[INFO] Camera connected.")
                return
            print("[ERROR] No camera detected. Retrying...")
            time.sleep(1)

    def segment_beam(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask = cv2.inRange(rgb, self.lower_red_rgb, self.upper_red_rgb)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            return mask, frame, None, None

        vx, vy, x0, y0 = cv2.fitLine(np.column_stack((xs, ys)).astype(np.float32),
                                     cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
        pts = np.column_stack((xs, ys)).astype(np.float32)
        diffs = pts - np.array([x0, y0])
        proj = diffs @ np.array([vx, vy])
        t_min, t_max = np.percentile(proj, [1, 99])
        p1 = np.array([x0 + t_min * vx, y0 + t_min * vy])
        p2 = np.array([x0 + t_max * vx, y0 + t_max * vy])
        p1, p2 = tuple(map(int, p1)), tuple(map(int, p2))

        annotated = frame.copy()
        cv2.line(annotated, p1, p2, (0, 0, 255), 3)
        return mask, annotated, p1, p2

    def segment_ball(self, frame, annotated):
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

    def run(self):
        print("[INFO] Starting PID-controlled tracker. Press 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Frame read failed. Reconnecting...")
                self.connect_camera()
                continue

            frame = cv2.flip(frame, 1)
            beam_mask, annotated, p1, p2 = self.segment_beam(frame)
            if p1 is None or p2 is None:
                stacked = np.vstack([annotated, frame])
                cv2.imshow("Beam-Ball Control", stacked)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            ball_mask, annotated, ball_center = self.segment_ball(frame, annotated)
            if ball_center is not None:
                beam_vec = (np.array(p2) - np.array(p1)).astype(np.float32)
                ball_proj = np.dot(ball_center - np.array(p1), beam_vec) / np.dot(beam_vec, beam_vec)
                pos = np.clip(ball_proj, 0, 1)

                # PID control
                control = self.pid.update(self.target, pos)
                servo_angle = self.level_angle - control
                servo_angle = np.clip(servo_angle, self.angle_min, self.angle_max)
                self.motor.send_angle(servo_angle)

                error = self.pid.prev_error
                print(f"\rBall: {pos:.3f} | Angle: {servo_angle:.2f} | Error: {error:+.4f}", end="", flush=True)

            combined_mask = cv2.bitwise_or(beam_mask, ball_mask)
            mask_view = cv2.bitwise_and(frame, frame, mask=combined_mask)

            stacked = np.vstack([annotated, mask_view])
            cv2.imshow("Beam-Ball Control", stacked)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.motor.ser:
            self.motor.ser.close()
        print("[INFO] Tracker stopped.")


if __name__ == "__main__":
    params = {
        "camera_index": 0,
        "lower_ball_rgb": (170, 80, 0),
        "upper_ball_rgb": (250, 175, 80),
        "lower_red_rgb": (190, 40, 50),
        "upper_red_rgb": (255, 90, 90),
        "angle_min": 135.0,
        "angle_max": 180.0,
        "level_angle": 157.5,
    }

    pid = PID(Kp=60, Ki=0.0, Kd=0.0)
    motor = MotorSerial()
    motor.connect()
    tracker = BeamBallTracker(params, motor, pid)
    tracker.run()
