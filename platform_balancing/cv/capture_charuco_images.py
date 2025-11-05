import cv2
import os
from datetime import datetime

# === USER SETTINGS ===
SAVE_DIR = "charuco_images"     # Folder to save captured images
CAMERA_ID = 0                   # Usually 0 for default webcam
IMAGE_FORMAT = "png"            # or "jpg"
DISPLAY_SCALE = 1.0             # e.g. 0.75 to shrink preview window

# Create output directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

# Set resolution for Logitech C920S (Full HD)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # ensure autofocus is on

if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("\n=== ChArUco Capture Tool ===")
print("Press 's' to save an image")
print("Press 'q' to quit\n")

counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed — check camera connection.")
        break

    # Resize preview for easier display (optional)
    if DISPLAY_SCALE != 1.0:
        frame_display = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    else:
        frame_display = frame

    # Show live feed
    cv2.imshow("ChArUco Capture", frame_display)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"charuco_{timestamp}.{IMAGE_FORMAT}"
        filepath = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(filepath, frame)
        counter += 1
        print(f"[{counter:02d}] Saved {filepath}")
    elif key == ord('q'):
        print("\nExiting capture.")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
