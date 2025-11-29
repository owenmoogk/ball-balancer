import cv2
import numpy as np

camera_index = 1  # change if needed
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

samples = []


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color = hsv[y, x].astype(int)
        samples.append(color)
        print(f"Sample {len(samples)}: H={color[0]}, S={color[1]}, V={color[2]}")


def compute_range(samples):
    samples = np.array(samples)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    lower = np.maximum(mean - 2 * std, [0, 0, 0]).astype(int)
    upper = np.minimum(mean + 2 * std, [179, 255, 255]).astype(int)

    print("\nRecommended HSV range:")
    print(f"Lower: np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
    print(f"Upper: np.array([{upper[0]}, {upper[1]}, {upper[2]}])")


cv2.namedWindow("Color Sampler")
cv2.setMouseCallback("Color Sampler", on_click)

print("Click on the ball a few times to sample colors. Press 'q' when done.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Color Sampler", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

if samples:
    compute_range(samples)
else:
    print("No samples collected.")
