#!/usr/bin/env python3
"""
Verbose diagnostic version of Plate + Ball tracker.
Adds many prints and checks to trace PnP/homography failures and corner-order issues.
"""
import cv2
import yaml
import numpy as np
import csv
import time
from scipy.spatial.transform import Rotation as R
import math
import sys
from collections import Counter

# ------------- CONFIG -------------
CALIB_FILE = "/Users/brendanchharawala/Documents/GitHub/ball-balancer/camera_tracking/calibration_and_generation/camera_calibration.yaml"
TAG_CSV = "/Users/brendanchharawala/Documents/GitHub/ball-balancer/camera_tracking/calibration_and_generation/generated/tags_positions.csv"

TAG_SIZE_MM = 75.0
AXIS_LENGTH_MM = 200.0
TICK_MM = 50.0

tol_ball_low = -24; tol_ball_high = 24
tol_ring_low = -15; tol_ring_high = 10
BALL_LOWER_BGR = np.array([51+tol_ball_low, 133+tol_ball_low, 191+tol_ball_low], dtype=np.uint8)
BALL_UPPER_BGR = np.array([87+tol_ball_high, 163+tol_ball_high, 216+tol_ball_high], dtype=np.uint8)
RING_LOWER_BGR = np.array([33+tol_ring_low, 63+tol_ring_low, 167+tol_ring_low], dtype=np.uint8)
RING_UPPER_BGR = np.array([64+tol_ring_high, 79+tol_ring_high, 182+tol_ring_high], dtype=np.uint8)

MIN_HOMOGRAPHY_POINTS = 8
MAX_REPROJ_ERR_PX = 10.0
PNP_REPROJ_THRESHOLD = 5.0
PNP_ITERATIONS = 100
PNP_CONFIDENCE = 0.99

DRAW_CORNER_INDICES = True
VERBOSE = True  # master switch for console prints

# ------------- LOAD CALIBRATION -------------
with open(CALIB_FILE, "r") as f:
    calib = yaml.safe_load(f)
camera_matrix = np.array(calib["camera_matrix"], dtype=np.float64)
dist_coeffs = np.array(calib["dist_coeff"], dtype=np.float64)
fx = camera_matrix[0,0]; fy = camera_matrix[1,1]
cx = camera_matrix[0,2]; cy = camera_matrix[1,2]
K_inv = np.linalg.inv(camera_matrix)

def cond_number(mat):
    try:
        return np.linalg.cond(mat)
    except Exception:
        return float("inf")

if VERBOSE:
    print("Camera matrix:\n", camera_matrix)
    print("dist coeffs:", dist_coeffs)
    print("Camera matrix cond:", cond_number(camera_matrix))

# ------------- LOAD TAG POSITIONS (plate XY mm) -------------
tag_positions = {}
with open(TAG_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tag_positions[int(row["tag_id"])] = np.array([float(row["x_mm"]), float(row["y_mm"])], dtype=np.float64)

if VERBOSE:
    print(f"Loaded {len(tag_positions)} tag positions from {TAG_CSV}")
    if len(tag_positions) == 0:
        print("ERROR: no tag positions loaded. Exiting.")
        sys.exit(1)

# ------------- HELPERS -------------
def tag_corners_3d(tag_id):
    x_c, y_c = tag_positions[tag_id]
    s = TAG_SIZE_MM / 2.0
    # Order assumed: 0=tl,1=tr,2=br,3=bl
    return np.array([
        [x_c - s, y_c - s, 0.0],
        [x_c + s, y_c - s, 0.0],
        [x_c + s, y_c + s, 0.0],
        [x_c - s, y_c + s, 0.0],
    ], dtype=np.float64)

def solve_pnp_ransac(obj_pts, img_pts):
    if obj_pts is None or img_pts is None or len(obj_pts) < 4:
        return False, None, None, None
    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, camera_matrix, dist_coeffs,
            reprojectionError=PNP_REPROJ_THRESHOLD,
            iterationsCount=PNP_ITERATIONS,
            confidence=PNP_CONFIDENCE,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        return bool(ok), rvec, tvec, inliers
    except Exception as e:
        if VERBOSE:
            print("solvePnPRansac exception:", e)
        return False, None, None, None

def detect_ball(frame):
    mask = cv2.inRange(frame, BALL_LOWER_BGR, BALL_UPPER_BGR)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8), iterations=1)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None, None, mask
    cx_px = int(M['m10']/M['m00']); cy_px = int(M['m01']/M['m00'])
    return (cx_px, cy_px), c, mask

def detect_ring(frame):
    mask = cv2.inRange(frame, RING_LOWER_BGR, RING_UPPER_BGR)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, mask
    c = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    return (int(cx), int(cy)), c, mask

def project_point_cam_to_image(pt_cam):
    X = pt_cam.flatten()
    if X[2] <= 1e-9:
        return None
    u = int((X[0]*fx)/X[2] + cx)
    v = int((X[1]*fy)/X[2] + cy)
    return (u, v)

def plate_point_to_image(pt_plate_mm, R_mat, tvec):
    cam = R_mat @ pt_plate_mm.reshape(3,1) + tvec.reshape(3,1)
    return project_point_cam_to_image(cam)

def image_px_to_plate_ray(u, v, R_mat, tvec):
    if R_mat is None or tvec is None:
        return None
    d = np.array([(u - cx)/fx, (v - cy)/fy, 1.0], dtype=np.float64).reshape(3,1)
    Rt = R_mat.T
    numerator = (Rt @ tvec.reshape(3,1))[2,0]
    denom = (Rt @ d)[2,0]
    if abs(denom) < 1e-9:
        return None
    s = numerator / denom
    X_cam = (s * d).reshape(3,1)
    X_plate = Rt @ (X_cam - tvec.reshape(3,1))
    return X_plate.flatten()

def homography_to_pose(H_plate_to_img):
    if H_plate_to_img is None:
        return None, None
    Hn = K_inv @ H_plate_to_img
    h1 = Hn[:,0]; h2 = Hn[:,1]; h3 = Hn[:,2]
    norm1 = np.linalg.norm(h1); norm2 = np.linalg.norm(h2)
    s = (norm1 + norm2) / 2.0
    if s < 1e-9:
        if VERBOSE:
            print("homography_to_pose: scale s too small", s)
        return None, None
    r1 = h1 / s; r2 = h2 / s
    r3 = np.cross(r1, r2)
    R_approx = np.column_stack((r1, r2, r3))
    U, S, Vt = np.linalg.svd(R_approx)
    R_mat = U @ Vt
    if np.linalg.det(R_mat) < 0:
        R_mat[:,2] *= -1
    t = h3 / s
    if VERBOSE:
        print("homography_to_pose: norms", norm1, norm2, "s", s, "svd S", S)
    return R_mat, t

def image_px_to_plate_with_fallback(u, v, R_mat, tvec, H_plate_to_img):
    if R_mat is not None and tvec is not None:
        p = image_px_to_plate_ray(u, v, R_mat, tvec)
        if p is not None:
            return p
    if H_plate_to_img is not None:
        try:
            Hinv = np.linalg.inv(H_plate_to_img)
        except np.linalg.LinAlgError:
            if VERBOSE:
                print("H inverse failed: singular")
            return None
        pt_img = np.array([u, v, 1.0], dtype=np.float64)
        pt_plate_h = Hinv @ pt_img
        if abs(pt_plate_h[2]) < 1e-9:
            return None
        x = pt_plate_h[0] / pt_plate_h[2]
        y = pt_plate_h[1] / pt_plate_h[2]
        return np.array([x, y, 0.0], dtype=np.float64)
    return None

def per_tag_ordering_check(obj4, img4):
    """
    Check cyclic shifts of obj4 (assumed tl,tr,br,bl) against img4 and return best shift and error.
    """
    img = img4.astype(np.float64)
    best_err = float("inf"); best_k = 0
    for k in range(4):
        obj = np.roll(obj4[:, :2], -k, axis=0)
        # compute affine 2D least squares (6 dof) mapping obj->img
        A = []
        b = []
        for (xo, yo), (xi, yi) in zip(obj, img):
            A.append([xo, yo, 1, 0, 0, 0])
            A.append([0, 0, 0, xo, yo, 1])
            b.append(xi); b.append(yi)
        A = np.array(A); b = np.array(b)
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
            M = np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [0,0,1]])
            mapped = (M @ np.vstack([obj.T, np.ones((1,4))]))[:2,:].T
            err = np.mean(np.linalg.norm(mapped - img, axis=1))
            if err < best_err:
                best_err = err; best_k = k
        except Exception as e:
            continue
    return best_k, best_err

# ------------- CAMERA & ARUCO INIT -------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera index 0 not available")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters_create()

prev_time = time.time()
fps = 0.0
frame_count = 0
t0 = time.time()

print("Press ESC to quit. Verbose mode:", VERBOSE)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_count += 1
    frame_undist = cv2.undistort(frame, camera_matrix, dist_coeffs)
    display = frame_undist.copy()
    loop_t0 = time.time()

    corners, ids, _ = cv2.aruco.detectMarkers(frame_undist, aruco_dict, parameters=parameters)
    if ids is not None:
        ids_flat = ids.flatten()
    else:
        ids_flat = np.array([], dtype=np.int32)

    visible_ids = []
    obj_pts_list = []
    img_pts_list = []
    missing_ids = []
    per_tag_info = []

    if len(ids_flat) > 0:
        cv2.aruco.drawDetectedMarkers(display, corners, ids)
        # detect duplicates
        dup_counts = Counter(ids_flat.tolist())
        if any(v>1 for v in dup_counts.values()) and VERBOSE:
            print("Duplicate detected IDs in single frame:", {k:v for k,v in dup_counts.items() if v>1})

        for i, raw_id in enumerate(ids_flat):
            tid = int(raw_id)
            corner_pts = np.array(corners[i][0], dtype=np.float64)  # 4x2
            center_px = tuple(corner_pts.mean(axis=0).astype(int))
            for p in corner_pts.astype(int):
                cv2.circle(display, tuple(p), 3, (0,200,200), -1)
            cv2.circle(display, center_px, 4, (0,255,255), -1)
            cv2.putText(display, str(tid), (center_px[0]+6, center_px[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            if DRAW_CORNER_INDICES:
                for j, p in enumerate(corner_pts.astype(int)):
                    cv2.putText(display, str(j), (int(p[0])+4, int(p[1])+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
            if tid not in tag_positions:
                missing_ids.append(tid)
                continue
            visible_ids.append(tid)

            obj4 = tag_corners_3d(tid)   # 4x3
            img4 = corner_pts            # 4x2

            # ordering check
            best_k, best_err = per_tag_ordering_check(obj4, img4)
            per_tag_info.append((tid, best_k, best_err))
            if VERBOSE:
                print(f"[Tag {tid}] center img {center_px}, best corner cyclic shift {best_k}, avg 2D affine err {best_err:.3f}")

            # store original obj4 (no rotation) for later concatenation;
            # we will also store the best shift to help debug mismatches
            # keep both: original + best-shifted for optional tests
            obj_pts_list.append(obj4)    # keep unaligned for now
            img_pts_list.append(img4)

    if VERBOSE and missing_ids:
        print("Detected ids missing from CSV:", missing_ids)

    # Concatenate correspondences and print diagnostics
    if len(obj_pts_list) > 0:
        obj_pts_all = np.concatenate(obj_pts_list, axis=0)  # Nx3
        img_pts_all = np.concatenate(img_pts_list, axis=0)  # Nx2
        n_img_pts = img_pts_all.shape[0]
        if VERBOSE:
            print(f"Frame {frame_count}: {len(visible_ids)} visible ids, {n_img_pts} image points total")
            print("Visible ids:", visible_ids)
            # per-tag summary
            for tid, shift, err in per_tag_info:
                print(f"  Tag {tid}: reported best shift {shift}, shift_err {err:.3f}")

        cv2.putText(display, f"Img pts: {n_img_pts}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # BEFORE calling PnP: sanity checks on matching
        # check distances between expected object corner distances vs detected image distances
        try:
            # compute object edge lengths in mm for each corner pair per tag
            obj_edge_lengths = []
            img_edge_lengths = []
            for i_block in range(0, obj_pts_all.shape[0], 4):
                obj4 = obj_pts_all[i_block:i_block+4, :2]
                img4 = img_pts_all[i_block:i_block+4, :]
                # pairwise distance sets (sorted) for topology check
                obj_d = sorted([np.linalg.norm(obj4[a]-obj4[b]) for a in range(4) for b in range(a+1,4)])
                img_d = sorted([np.linalg.norm(img4[a]-img4[b]) for a in range(4) for b in range(a+1,4)])
                obj_edge_lengths.append(obj_d)
                img_edge_lengths.append(img_d)
            if VERBOSE:
                for k,(od,id_) in enumerate(zip(obj_edge_lengths, img_edge_lengths)):
                    print(f" Tag-block {k}: obj pairwise distances (mm) {['{:.1f}'.format(x) for x in od]}")
                    print(f" Tag-block {k}: img pairwise distances (px) {['{:.1f}'.format(x) for x in id_]}")
        except Exception as e:
            print("Error computing pairwise distances:", e)

        # PnP primary
        pnp_success = False
        if n_img_pts >= 4:
            pnp_call_t0 = time.time()
            success, rvec, tvec, inliers = solve_pnp_ransac(obj_pts_all.astype(np.float64), img_pts_all.astype(np.float64))
            pnp_time = (time.time() - pnp_call_t0)
            if VERBOSE:
                print(f"solvePnPRansac returned success={success}, inliers_shape={None if inliers is None else inliers.shape}, time={pnp_time:.3f}s")
            if success and rvec is not None and tvec is not None:
                img_proj, _ = cv2.projectPoints(obj_pts_all, rvec, tvec, camera_matrix, dist_coeffs)
                img_proj = img_proj.reshape(-1,2)
                errs = np.linalg.norm(img_proj - img_pts_all, axis=1)
                reproj_err = float(errs.mean())
                per_pt_errs = errs.reshape(-1,4) if errs.size % 4 == 0 else None
                if VERBOSE:
                    print(f"Global PnP reproj err mean: {reproj_err:.3f}px, min {errs.min():.3f}, max {errs.max():.3f}")
                    if per_pt_errs is not None:
                        for i_tag, row in enumerate(per_pt_errs):
                            print(f"  Tag idx {i_tag} per-corner reproj errs: {['{:.2f}'.format(x) for x in row]}")
                    # check rvec/tvec magnitude
                    print("rvec norm:", np.linalg.norm(rvec), "tvec norm:", np.linalg.norm(tvec))
                # detailed checks: reproject each tag separately using its 3D corners and compare ordering
                for i_block in range(0, obj_pts_all.shape[0], 4):
                    obj4 = obj_pts_all[i_block:i_block+4]
                    img4 = img_pts_all[i_block:i_block+4]
                    proj4, _ = cv2.projectPoints(obj4, rvec, tvec, camera_matrix, dist_coeffs)
                    proj4 = proj4.reshape(-1,2)
                    dists = np.linalg.norm(proj4 - img4, axis=1)
                    if VERBOSE:
                        print(f" Block {i_block//4} proj->det corner diffs px: {['{:.2f}'.format(x) for x in dists]}")
                # check reproj threshold
                if reproj_err <= MAX_REPROJ_ERR_PX:
                    pnp_success = True
                    R_mat, _ = cv2.Rodrigues(rvec)
                    tvec = tvec.reshape(3,)
                    rot = R.from_matrix(R_mat)
                    roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
                    method_used = "PnP"
                    # draw reprojection debug
                    for det_pt, proj_pt in zip(img_pts_all.reshape(-1,2).astype(int), img_proj.astype(int)):
                        det = tuple(det_pt.tolist()); proj = tuple(proj_pt.tolist())
                        cv2.circle(display, det, 3, (0,0,255), -1)
                        cv2.circle(display, proj, 3, (0,255,0), -1)
                        cv2.line(display, det, proj, (100,100,100), 1)
                else:
                    if VERBOSE:
                        print("PnP rejected: reproj too large", reproj_err, ">", MAX_REPROJ_ERR_PX)
                    method_used = "PnP-rejected"
            else:
                if VERBOSE:
                    print("PnP failed to return valid rvec/tvec")
                reproj_err = None
        else:
            if VERBOSE:
                print(f"Not enough pts for PnP ({n_img_pts} < 4)")
            reproj_err = None

        # Homography fallback and pose decomposition (so RPY available)
        H_plate_to_img = None
        mask_h = None
        if not pnp_success and n_img_pts >= MIN_HOMOGRAPHY_POINTS:
            H_call_t0 = time.time()
            try:
                # use only XY columns of object points
                src = obj_pts_all[:, :2].astype(np.float64)
                dst = img_pts_all.astype(np.float64)
                H_plate_to_img, mask_h = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
            except Exception as e:
                if VERBOSE:
                    print("findHomography exception:", e)
                H_plate_to_img = None
            H_time = time.time() - H_call_t0
            if VERBOSE:
                print("findHomography time:", H_time, "returned H is None?", H_plate_to_img is None)
            if H_plate_to_img is not None:
                H_plate_to_img = H_plate_to_img.astype(np.float64)
                inliers = int(mask_h.sum()) if mask_h is not None else 0
                method_used = f"H ({inliers}/{n_img_pts})"
                if VERBOSE:
                    print(f"Homography inliers: {inliers}/{n_img_pts}; H cond: {cond_number(H_plate_to_img)}")
                # compute homography reprojection error per correspondence
                try:
                    src_h = np.hstack([src, np.ones((src.shape[0],1))]).T  # 3xN
                    mapped = (H_plate_to_img @ src_h).T  # Nx3
                    mapped /= mapped[:,2:3]
                    errs_h = np.linalg.norm(mapped[:,:2] - dst, axis=1)
                    if VERBOSE:
                        print(f"Homography reproj mean: {errs_h.mean():.3f}px, min {errs_h.min():.3f}, max {errs_h.max():.3f}")
                except Exception as e:
                    print("Homography reproj check failed:", e)
                # Decompose to R,t
                R_h, t_h = homography_to_pose(H_plate_to_img)
                if R_h is not None and t_h is not None:
                    R_mat = R_h
                    tvec = t_h.reshape(3,)
                    rot = R.from_matrix(R_mat)
                    roll, pitch, yaw = rot.as_euler('xyz', degrees=True)
                    reproj_err = None
                else:
                    if VERBOSE:
                        print("Homography -> pose failed")
            else:
                if VERBOSE:
                    print("Homography failed (H_plate_to_img is None)")

    else:
        obj_pts_all = None
        img_pts_all = None
        n_img_pts = 0
        if VERBOSE:
            print("No known tag positions matched detected ids this frame")

    # detect ring and ball (use undistorted frame)
    ring_px, ring_contour, ring_mask = detect_ring(frame_undist)
    if ring_px is not None:
        cv2.circle(display, ring_px, 8, (0,255,0), -1)

    ball_px, ball_contour, ball_mask = detect_ball(frame_undist)
    if ball_px is not None:
        cv2.circle(display, (int(ball_px[0]), int(ball_px[1])), 6, (255,0,0), -1)

    # Choose origin: ring centre if visible and mappable, else mean of visible tag plate centers
    origin_plate = None
    origin_img = None

    if ring_px is not None:
        rp = image_px_to_plate_with_fallback(ring_px[0], ring_px[1], locals().get('R_mat', None), locals().get('tvec', None), locals().get('H_plate_to_img', None))
        if rp is not None:
            origin_plate = rp
            origin_img = (int(ring_px[0]), int(ring_px[1]))
            if VERBOSE:
                print("Ring->plate mapping succeeded, origin_plate", origin_plate)

    if origin_plate is None and len(visible_ids) >= 1:
        pts = np.array([tag_positions[tid] for tid in visible_ids], dtype=np.float64)
        center = pts.mean(axis=0)
        origin_plate = np.array([center[0], center[1], 0.0], dtype=np.float64)
        if 'R_mat' in locals() and 'tvec' in locals() and R_mat is not None and tvec is not None:
            p = plate_point_to_image(origin_plate, R_mat, tvec)
            if p is not None:
                origin_img = p
        if VERBOSE:
            print("Fallback origin plate (mean of visible tags):", origin_plate)

    # Draw axes through origin using computed pose (or homography-derived pose)
    if origin_plate is not None and ('R_mat' in locals() and 'tvec' in locals() and R_mat is not None and tvec is not None):
        gray = (200,200,200); thickness = 3; arrow_sz_px = 12
        x_pos_plate = origin_plate + np.array([AXIS_LENGTH_MM, 0.0, 0.0], dtype=np.float64)
        y_pos_plate = origin_plate + np.array([0.0, AXIS_LENGTH_MM, 0.0], dtype=np.float64)

        p_o = plate_point_to_image(origin_plate, R_mat, tvec)
        p_xp = plate_point_to_image(x_pos_plate, R_mat, tvec)
        p_yp = plate_point_to_image(y_pos_plate, R_mat, tvec)

        if p_o is not None and p_xp is not None:
            cv2.line(display, p_o, p_xp, gray, thickness)
        if p_o is not None and p_yp is not None:
            cv2.line(display, p_o, p_yp, gray, thickness)

        def draw_arrow(pt_from, pt_to):
            if pt_from is None or pt_to is None:
                return
            dx = pt_to[0] - pt_from[0]; dy = pt_to[1] - pt_from[1]
            L = max(1.0, np.hypot(dx, dy)); ux, uy = dx/L, dy/L
            wing1 = (int(pt_to[0] - arrow_sz_px*(ux + 0.4*uy)), int(pt_to[1] - arrow_sz_px*(uy - 0.4*ux)))
            wing2 = (int(pt_to[0] - arrow_sz_px*(ux - 0.4*uy)), int(pt_to[1] - arrow_sz_px*(uy + 0.4*ux)))
            cv2.line(display, pt_to, wing1, gray, thickness); cv2.line(display, pt_to, wing2, gray, thickness)

        draw_arrow(p_o, p_xp); draw_arrow(p_o, p_yp)
        if p_o is not None:
            cv2.circle(display, p_o, 6, (180,180,180), -1)
            cv2.putText(display, f"Origin ({origin_plate[0]:.1f}, {origin_plate[1]:.1f}) mm", (p_o[0]+10, p_o[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

    # Compute ball plate XY using ray-plane or H inverse
    ball_plate_xy = None
    if ball_px is not None:
        ball_plate_pt = image_px_to_plate_with_fallback(int(ball_px[0]), int(ball_px[1]), locals().get('R_mat', None), locals().get('tvec', None), locals().get('H_plate_to_img', None))
        if ball_plate_pt is not None:
            if origin_plate is not None:
                dx = ball_plate_pt[0] - origin_plate[0]; dy = ball_plate_pt[1] - origin_plate[1]
                ball_plate_xy = (dx, dy)
            else:
                ball_plate_xy = (ball_plate_pt[0], ball_plate_pt[1])
            # visualize projected plate point back to image
            p_img = None
            if 'R_mat' in locals() and 'tvec' in locals() and R_mat is not None and tvec is not None:
                p_img = plate_point_to_image(ball_plate_pt, R_mat, tvec)
            if p_img is None and 'H_plate_to_img' in locals() and H_plate_to_img is not None:
                pt = np.array([ball_plate_pt[0], ball_plate_pt[1], 1.0], dtype=np.float64)
                mapped = H_plate_to_img @ pt
                if abs(mapped[2]) > 1e-9:
                    p_img = (int(mapped[0]/mapped[2]), int(mapped[1]/mapped[2]))
            if p_img is not None:
                cv2.circle(display, p_img, 6, (255,0,255), 2)
                cv2.line(display, (int(ball_px[0]), int(ball_px[1])), p_img, (255,0,255), 1)
        if VERBOSE:
            print("Ball px:", ball_px, "ball_plate (estimated):", ball_plate_pt if 'ball_plate_pt' in locals() else None, "ball_xy_rel_origin:", ball_plate_xy)

    # FPS and timings
    curr_time = time.time()
    loop_dt = curr_time - loop_t0
    fps = 0.9*fps + 0.1*(1.0/(curr_time - prev_time)) if curr_time != prev_time else fps
    prev_time = curr_time

    # Info block (top-right)
    info_lines = [
        f"Method: {locals().get('method_used','--')}",
        f"Roll={locals().get('roll',0.0):.1f}, Pitch={locals().get('pitch',0.0):.1f}, Yaw={locals().get('yaw',0.0):.1f}",
        f"Visible tags: {visible_ids if visible_ids else '--'}",
        f"Origin px: {origin_img if origin_img is not None else '--'}",
        f"Ball XY (mm) rel origin: {f'{ball_plate_xy[0]:.1f}, {ball_plate_xy[1]:.1f}' if ball_plate_xy is not None else '--'}",
        f"PnP reproj: {locals().get('reproj_err',-1):.2f}px" if locals().get('reproj_err',None) is not None else "PnP reproj: --",
        f"FPS: {fps:.1f}",
        f"Loop dt: {loop_dt*1000:.1f} ms"
    ]
    x0 = display.shape[1] - 560; y0 = 30
    for i, line in enumerate(info_lines):
        cv2.putText(display, line, (x0, y0 + 26*i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # Show windows
    cv2.imshow("Plate + Ball Tracker (verbose)", display)
    if 'ball_mask' in locals() and ball_mask is not None:
        cv2.imshow("Ball Mask", ball_mask)
    if 'ring_mask' in locals() and ring_mask is not None:
        cv2.imshow("Ring Mask", ring_mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

# End loop
cap.release()
cv2.destroyAllWindows()
print("Exited. Frames processed:", frame_count, "elapsed time:", time.time()-t0)