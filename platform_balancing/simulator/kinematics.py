import numpy as np
from typing import Tuple, Optional, Sequence
from numpy.typing import NDArray
from settings import Settings

def rotation_matrix(roll: float, pitch: float) -> NDArray[np.float64]:
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return R_y @ R_x  # type: ignore


def bearing_point_exact(
    base: Sequence[float],
    p_world: Sequence[float],
    l1: float,
    l2: float,
) -> Tuple[bool, Optional[NDArray[np.float64]]]:
    B = np.array(base, dtype=np.float64)
    P = np.array(p_world, dtype=np.float64)
    O = np.array([0.0, 0.0, 0.0])

    d = np.linalg.norm(P - B)
    if d > l1 + l2 + 1e-9 or d < abs(l1 - l2) - 1e-9 or d < 1e-9:
        return False, None

    v = (P - B) / d
    a = (l1**2 - l2**2 + d**2) / (2.0 * d)
    h2 = l1**2 - a**2
    if h2 < -1e-12:
        return False, None
    h2 = max(h2, 0.0)
    h = np.sqrt(h2)

    M = B + a * v

    # Define radial direction in XY plane
    radial_dir = B - O
    radial_dir[2] = 0.0
    norm = np.linalg.norm(radial_dir)
    if norm < 1e-9:
        radial_dir = np.array([1.0, 0.0, 0.0])
    else:
        radial_dir /= norm

    # Plane normal = cross(radial, Z)
    plane_normal = np.cross(radial_dir, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(plane_normal) < 1e-9:
        plane_normal = np.array([0.0, 1.0, 0.0])
    else:
        plane_normal /= np.linalg.norm(plane_normal)

    # Perpendicular in plane
    n_perp = np.cross(v, plane_normal)
    n_perp /= np.linalg.norm(n_perp)

    bearing1 = M + h * n_perp
    bearing2 = M - h * n_perp

    print(bearing1, bearing2)

    bearing = bearing1 if bearing1[2] < bearing2[2] else bearing2

    return True, bearing

def motor_angle_deg(base, bearing):
    radial = base.copy(); radial[2] = 0
    if np.linalg.norm(radial) < 1e-9: radial = np.array([1,0,0])
    e1 = radial / np.linalg.norm(radial)
    n = np.cross(e1, np.array([0,0,1])); n /= np.linalg.norm(n)
    e2 = np.cross(n, e1)
    L = bearing - base
    Lp = L - np.dot(L, n)*n
    x, y = np.dot(Lp, e1), np.dot(Lp, e2)
    return float(np.rad2deg(np.arctan2(y, x)))

def leg_points_rigid(base, contact_local, plane_pose, l1, l2):
    roll, pitch, z = plane_pose
    R = rotation_matrix(roll, pitch)
    P_world = R @ np.array([contact_local[0], contact_local[1], 0.0]) + np.array([0, 0, z])
    ok, bearing = bearing_point_exact(base, P_world, l1, l2)
    if not ok: return None
    return base, bearing

def solve_motor_angles_for_plane(roll_deg, pitch_deg, z=Settings.TABLE_HEIGHT):
    roll = np.deg2rad(roll_deg); pitch = np.deg2rad(pitch_deg)
    out = np.full(3, np.nan)
    for i, (b, c) in enumerate(zip(Settings.BASES, Settings.CONTACTS)):
        segs = leg_points_rigid(np.array(b), np.array([c[0], c[1], 0.0]), (roll, pitch, z), Settings.MOTOR_LINK_LEN, Settings.PUSH_LINK_LEN)
        if segs is None: continue
        b1, br = segs
        out[i] = motor_angle_deg(b1, br)
    return out