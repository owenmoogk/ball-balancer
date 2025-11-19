import cv2
import numpy as np

# SETTINGS
CAM_INDEX = 4
NUM_POINTS = 10

clicked_hsv_values = []
frame_for_click = None

def mouse_callback(event, x, y, flags, param):
    global clicked_hsv_values, frame_for_click

    if event == cv2.EVENT_LBUTTONDOWN:
        if frame_for_click is not None:
            hsv_frame = cv2.cvtColor(frame_for_click, cv2.COLOR_BGR2HSV)
            hsv_value = hsv_frame[y, x]  # (H, S, V)
            clicked_hsv_values.append(hsv_value)

            print(f"[{len(clicked_hsv_values)}] HSV at ({x}, {y}): {hsv_value}")

def main():
    global frame_for_click

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera index {CAM_INDEX}")
        return

    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", mouse_callback)

    print("Click 10 points on the ping pong ball.")
    print("Press 'q' to quit early.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_for_click = frame.copy()
        display_frame = frame.copy()

        # Draw count on screen
        cv2.putText(display_frame, f"Clicks: {len(clicked_hsv_values)}/{NUM_POINTS}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Camera", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if len(clicked_hsv_values) >= NUM_POINTS:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(clicked_hsv_values) == 0:
        print("No points clicked.")
        return

    hsv_array = np.array(clicked_hsv_values)
    # Compute min/max per channel
    lower = hsv_array.min(axis=0)
    upper = hsv_array.max(axis=0)

    # Expand range slightly for robustness
    lower = np.maximum(lower - np.array([5, 30, 30]), 0)
    upper = np.minimum(upper + np.array([5, 30, 30]), [179, 255, 255])

    print("\n========== RESULT ==========")
    print("Clicked HSV values:")
    for v in clicked_hsv_values:
        print(v)

    print("\nSuggested HSV Lower Bound:", lower.tolist())
    print("Suggested HSV Upper Bound:", upper.tolist())

    print("\nCopy/paste into your code as:")
    print(f"lower_hsv = np.array({lower.tolist()})")
    print(f"upper_hsv = np.array({upper.tolist()})")

if __name__ == "__main__":
    main()
