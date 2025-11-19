import cv2
import numpy as np

HSV_TOL = 30
class BallTracker:
    def __init__(self, camera_index=4):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.alpha_pos = 0.3
        self.filtered_position = None
        self.last_position = None

        # Calibrated HSV range
        self.lower_hsv = np.array([5, 172, 121], dtype=np.uint8)
        self.upper_hsv = np.array([22, 255, 239], dtype=np.uint8)

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
        ret, frame = self.cap.read()
        if not ret:
            return None #self.filtered_position  # or None

        h, w, _ = frame.shape
        self.frame_shape = (h, w)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius > 10:
                cx, cy = self.center if self.center is not None else (w / 2, h / 2)
                raw_pos = np.array([(x - cx) / w, (y - cy) / h])

                if self.filtered_position is None:
                    self.filtered_position = raw_pos
                else:
                    # Exponential moving average
                    self.filtered_position = (
                        (1 - self.alpha_pos) * self.filtered_position
                        + self.alpha_pos * raw_pos
                    )

                self.last_position = raw_pos

                if display:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"({self.filtered_position[0]:.3f}, {self.filtered_position[1]:.3f})",
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
        print("Filtered Pos: ", self.filtered_position)
        return self.filtered_position

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
