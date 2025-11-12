import cv2
import numpy as np

# Calibrated HSV range for the ball
BALL_LOWER_HSV = np.array([11, 140, 188], dtype=np.uint8)
BALL_UPPER_HSV = np.array([16, 255, 255], dtype=np.uint8)

# Start webcam (camera index 1)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV for color-based masking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold image for ball color
    mask = cv2.inRange(hsv, BALL_LOWER_HSV, BALL_UPPER_HSV)

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Use largest contour as the ball
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"({int(x)}, {int(y)})",
                (int(x) + 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Ball Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
