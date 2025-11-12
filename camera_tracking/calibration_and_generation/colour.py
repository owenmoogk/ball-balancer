#!/usr/bin/env python3
import cv2
import numpy as np

CLICK_COUNT = 5
ball_points = []
ring_points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_list, name = param
        points_list.append((x, y))
        print(f"{name} point {len(points_list)}: {x}, {y}")
        if len(points_list) >= CLICK_COUNT:
            print(f"Finished selecting {name} points.")

# --- LOAD IMAGE ---
img_path = input("Enter image path: ").strip()
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("Image not found")

temp_img = img.copy()

# --- SELECT BALL POINTS ---
print(f"Click {CLICK_COUNT} points on the ball in the image window...")
cv2.namedWindow("Color Picker")
cv2.setMouseCallback("Color Picker", click_event, (ball_points, "ball"))

while len(ball_points) < CLICK_COUNT:
    cv2.imshow("Color Picker", temp_img)
    if cv2.waitKey(1) == 27:
        break

# --- SELECT RING POINTS ---
print(f"Click {CLICK_COUNT} points on the ring in the image window...")
ring_points = []
cv2.setMouseCallback("Color Picker", click_event, (ring_points, "ring"))

while len(ring_points) < CLICK_COUNT:
    cv2.imshow("Color Picker", temp_img)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

# --- COMPUTE BGR RANGE ---
def compute_bgr_range(points, img):
    pixels = np.array([img[y, x] for x, y in points])  # OpenCV is BGR
    b_min, g_min, r_min = pixels.min(axis=0)
    b_max, g_max, r_max = pixels.max(axis=0)
    # Expand range slightly
    b_min = max(b_min-5, 0); g_min = max(g_min-5, 0); r_min = max(r_min-5, 0)
    b_max = min(b_max+5, 255); g_max = min(g_max+5, 255); r_max = min(r_max+5, 255)
    return (b_min, g_min, r_min), (b_max, g_max, r_max)

ball_lower, ball_upper = compute_bgr_range(ball_points, img)
ring_lower, ring_upper = compute_bgr_range(ring_points, img)

print("Ball BGR range:")
print("Lower:", ball_lower)
print("Upper:", ball_upper)
print("\nRing BGR range:")
print("Lower:", ring_lower)
print("Upper:", ring_upper)

# --- DISPLAY BGR GRADIENTS ---
def create_gradient_bgr(lower, upper, width=300, height=50):
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        ratio = i / (width - 1)
        b = int(lower[0] + ratio * (upper[0] - lower[0]))
        g = int(lower[1] + ratio * (upper[1] - lower[1]))
        r = int(lower[2] + ratio * (upper[2] - lower[2]))
        gradient[:, i] = (b, g, r)
    return gradient

ball_grad = create_gradient_bgr(ball_lower, ball_upper)
ring_grad = create_gradient_bgr(ring_lower, ring_upper)

cv2.imshow("Ball BGR Range", ball_grad)
cv2.imshow("Ring BGR Range", ring_grad)
cv2.waitKey(0)
cv2.destroyAllWindows()