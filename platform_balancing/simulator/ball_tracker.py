import cv2
import numpy as np


class BallTracker:
    def __init__(self, camera_index=1):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Calibrated HSV range
        self.lower_hsv = np.array([10, 100, 110], dtype=np.uint8)
        self.upper_hsv = np.array([18, 255, 255], dtype=np.uint8)

        self.last_position = np.array([0.0, 0.0])
        self.center = None
        self.frame_shape = (480, 640)

        # Window setup
        cv2.namedWindow("Ball Tracking")
        cv2.setMouseCallback("Ball Tracking", self.set_center)

    def set_center(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.center = (x, y)
            print(f"Center set to: {self.center}")

    def draw_center(self, frame):
        h, w, _ = frame.shape
        if self.center is None:
            cx, cy = w // 2, h // 2
        else:
            cx, cy = self.center

        color = (0, 0, 255)  # red
        size = 10
        thickness = 2

        # Horizontal line
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)

        # Vertical line
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)

        # Small center dot
        cv2.circle(frame, (cx, cy), 3, color, -1)

    def get_x_y(self, display=True):
        """Returns (x, y) position in normalized coordinates (-0.5..0.5 range)."""
        ret, frame = self.cap.read()
        if not ret:
            return self.last_position

        h, w, _ = frame.shape
        self.frame_shape = (h, w)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pos = self.last_position

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 10:
                cx, cy = self.center if self.center is not None else (w / 2, h / 2)
                pos = np.array([(x - cx) / w, (y - cy) / h])
                self.last_position = pos

                if display:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"({pos[0]:.3f}, {pos[1]:.3f})",
                        (int(x) + 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

        if display:
            self.draw_center(frame)
            cv2.imshow("Ball Tracking", frame)
            cv2.waitKey(1)

        if not contours:
            return None

        return pos

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
