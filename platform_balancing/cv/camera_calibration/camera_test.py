import cv2
import numpy as np

# Load calibration data
camera_matrix = np.array([[932.0900126242127,0,638.6549019259521],
                [0,931.4949715786727,342.9911521888232],
                [0,0,1]])
dist_coeffs = np.array([0.10393692155716833,
                        -0.17168576648599584,
                        -0.0005549638588882746,
                        -0.0009705263907376315,
                        0.03137706338566113])

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame")
        break

    h, w = frame.shape[:2]
    new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1)
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_cam_mtx)

    # Combine views safely
    frame_small = cv2.resize(frame, (w//2, h//2))
    undist_small = cv2.resize(undistorted, (w//2, h//2))
    combined = np.hstack((frame_small, undist_small))

    cv2.imshow("Distorted (L) | Undistorted (R)", combined)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
