#!/usr/bin/env python3
"""
Try multiple aruco / apriltag dictionaries on an input image or camera feed.
Usage:
  python detect_dictionaries.py --image /path/to/photo.jpg
  python detect_dictionaries.py --cam 0
"""
import cv2, argparse, sys, time, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='single image file')
parser.add_argument('--cam', type=int, help='camera index to open (e.g. 0)')
args = parser.parse_args()

# candidate dictionaries to try (OpenCV names). Add more if needed.
CANDIDATES = [
    'DICT_APRILTAG_16h5',
    'DICT_APRILTAG_25h9',
    'DICT_APRILTAG_36h10',
    'DICT_APRILTAG_36h11',
    'DICT_4X4_50',
    'DICT_5X5_100',
    'DICT_6X6_250'
]

# sanity check OpenCV
print("OpenCV version:", cv2.__version__)

# build dictionary objects where available
available = {}
aruco = cv2.aruco
for name in CANDIDATES:
    try:
        id_const = getattr(aruco, name)
        try:
            d = aruco.getPredefinedDictionary(id_const)
        except Exception:
            d = aruco.Dictionary_get(id_const)
        available[name] = d
    except Exception:
        print("Not available in cv2:", name)

if len(available) == 0:
    print("No candidate dictionaries available in this cv2 build. Exiting.")
    sys.exit(1)

# helper to run detection for one dictionary
def detect_with_dict(gray, dictionary):
    # detector obj if available
    try:
        params = aruco.DetectorParameters()  # new API
    except Exception:
        params = aruco.DetectorParameters_create()
    try:
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except Exception:
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids

def run_on_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = {}
    for name, d in available.items():
        corners, ids = detect_with_dict(gray, d)
        ids_list = []
        if ids is not None:
            ids_list = [int(x) for x in ids.flatten()]
        results[name] = (corners, ids_list)
    return results

def show_results_annotated(frame, results):
    vis = frame.copy()
    y = 20
    for name, (corners, ids_list) in results.items():
        txt = f"{name}: {len(ids_list)}"
        cv2.putText(vis, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
        y += 18
    # also draw the first successful detection outlines (for visual)
    for name, (corners, ids_list) in results.items():
        if corners is not None and len(corners) > 0:
            for c, idv in zip(corners, ids_list):
                c2 = np.array(c).reshape(4,2).astype(int)
                cv2.polylines(vis, [c2], True, (0,255,0), 2)
                cx = int(c2[:,0].mean()); cy = int(c2[:,1].mean())
                cv2.putText(vis, str(idv), (cx-8, cy+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            # break after drawing from first dictionary with detections
            break
    return vis

# run once on image or in loop on camera
if args.image:
    frame = cv2.imread(args.image)
    if frame is None:
        print("Failed to read image:", args.image); sys.exit(1)
    results = run_on_frame(frame)
    for name, (c, ids_list) in results.items():
        print(f"{name:20s} -> {len(ids_list):2d} ids: {ids_list}")
    vis = show_results_annotated(frame, results)
    cv2.imshow("results", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif args.cam is not None:
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Cannot open camera", args.cam); sys.exit(1)
    print("Press q to quit, space to print the latest detection summary")
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = run_on_frame(frame)
        vis = show_results_annotated(frame, results)
        cv2.imshow("camera", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '):
            for name, (c, ids_list) in results.items():
                print(f"{name:20s} -> {len(ids_list):2d} ids: {ids_list}")
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Provide --image or --cam")