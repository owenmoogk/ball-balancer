import cv2
import os
import json

ARUCO_DICT = cv2.aruco.DICT_5X5_100  # Dictionary ID
SQUARES_VERTICALLY = 6               # Number of squares vertically
SQUARES_HORIZONTALLY = 9             # Number of squares horizontally
SQUARE_LENGTH = 90                   # Square side length (in pixels)
MARKER_LENGTH = SQUARE_LENGTH * 0.7  # ArUco marker side length (in pixels)
MARGIN_PX = 20                       # Margins size (in pixels)
IMG_SIZE = tuple(i * SQUARE_LENGTH + 2 * MARGIN_PX for i in (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY))
OUTPUT_NAME = 'ChArUco_Marker.png'

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, IMG_SIZE, marginSize=MARGIN_PX)
    cv2.imwrite(OUTPUT_NAME, img)

def get_calibration_parameters(img_dir):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    charucodetector = cv2.aruco.CharucoDetector(board)
    
    # Load images from directory
    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]
    all_charuco_ids = []
    all_charuco_corners = []

    # Loop over images and extraction of corners
    i = 1
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgSize = image.shape
        image_copy = image.copy()
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
        print(i)
        i+=1
        if len(marker_ids) > 0: # If at least one marker is detected
            # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            charucoCorners, charucoIds, marker_corners, marker_ids = charucodetector.detectBoard(image)
            if charucoIds is not None and len(charucoCorners) > 3:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)
    
    # Calibrate camera with extracted information
    result, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, imgSize, None, None)
    return mtx, dist

GENERATE = False
if __name__ == "__main__":
  if GENERATE:
    create_and_save_new_board()
  else:
    CAMERA = 'Logitech C920'
    OUTPUT_JSON = 'calibration.json'

    mtx, dist = get_calibration_parameters(img_dir=os.path.join(os.getcwd(),"charuco_images"))
    data = { "mtx": mtx.tolist(), "dist": dist.tolist()}

    with open(OUTPUT_JSON, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    print(f'Data has been saved to {OUTPUT_JSON}')