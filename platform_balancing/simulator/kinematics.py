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

    bearing = bearing1 if bearing1[2] < bearing2[2] else bearing2

    return True, bearing

def motor_angle_deg(base, bearing):
    base = np.array(base, dtype=float)
    bearing = np.array(bearing, dtype=float)
    L = bearing - base
    x, y, z = L
    angle_rad = np.arctan2(z, np.hypot(x, y))
    return float(np.rad2deg(angle_rad))

def leg_points_rigid(
    base: NDArray[np.float64],
    contact_local: NDArray[np.float64],
    plane_pose: Tuple[float, float, float],
    l1: float,
    l2: float,
) -> Optional[
    Tuple[
        Tuple[NDArray[np.float64], NDArray[np.float64]],
        Tuple[NDArray[np.float64], NDArray[np.float64]],
    ]
]:
    roll, pitch, z = plane_pose
    R = rotation_matrix(roll, pitch)
    P_world = R @ np.array([contact_local[0], contact_local[1], 0.0]) + np.array([0, 0, z])
    ok, bearing = bearing_point_exact(base, P_world, l1, l2)
    if not ok:
        return None
    return (base.copy(), bearing), (bearing.copy(), P_world.copy())

def solve_motor_angles_for_plane(roll_deg, pitch_deg, z=Settings.TABLE_HEIGHT):
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    out = np.full(3, np.nan)
    for i, (b, c) in enumerate(zip(Settings.BASES, Settings.CONTACTS)):
        leg_points = leg_points_rigid(np.array(b), np.array([c[0], c[1], 0.0]), (roll, pitch, z), Settings.MOTOR_LINK_LEN, Settings.PUSH_LINK_LEN)
        if leg_points:
            leg_1_points, _ = leg_points
            base_point, bearing_point = leg_1_points
            out[i] = motor_angle_deg(base_point, bearing_point)
    return out