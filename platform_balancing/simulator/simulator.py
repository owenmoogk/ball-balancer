import numpy as np
from typing import Tuple, List, Optional, Sequence
from numpy.typing import NDArray

# ------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------
MOTOR_LINK_LEN: float = 0.3
PUSH_LINK_LEN: float = 0.25
BALL_RADIUS: float = 0.02
G: float = 9.81
TABLE_HEIGHT: float = 0.4
PLATFORM_RADIUS: float = 0.5  # Radius of the circular platform
BASE_RADIUS: float = 0.6  # Radius of the base circle
PLATFORM_THICKNESS: float = 0.01

DT: float = 0.005
TARGET_FPS: int = 60

# Generate 120-degree spaced points
angles = np.deg2rad([0, 120, 240])
BASES: List[Tuple[float, float, float]] = [
    (BASE_RADIUS * np.cos(a), BASE_RADIUS * np.sin(a), 0.0) for a in angles
]
CONTACTS: List[Tuple[float, float]] = [
    (PLATFORM_RADIUS * np.cos(a), PLATFORM_RADIUS * np.sin(a)) for a in angles
]

I_SPHERE: float = (2.0 / 5.0) * BALL_RADIUS**2


# ------------------------------------------------------------------
#  Helper functions
# ------------------------------------------------------------------
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

    gravity = np.array([0.0, 0.0, -1.0])
    bearing = (
        bearing1 if np.dot(bearing1, gravity) < np.dot(bearing2, gravity) else bearing2
    )

    return True, bearing


# ------------------------------------------------------------------
#  Ball – pure rolling
# ------------------------------------------------------------------
class Ball:
    def __init__(self, pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.pos: NDArray[np.float64] = np.array(pos, dtype=float)
        self.vel: NDArray[np.float64] = np.zeros(3)
        self.radius: float = BALL_RADIUS

    def update(self, dt: float, plane_pose: Tuple[float, float, float]) -> None:
        roll, pitch, z = plane_pose
        R = rotation_matrix(roll, pitch)
        n = R @ np.array([0.0, 0.0, 1.0])
        n = n / np.linalg.norm(n)

        plane_point = np.array([0.0, 0.0, z])
        plane_offset = np.dot(n, plane_point)

        g_vec = np.array([0.0, 0.0, -G])
        a_parallel = g_vec - np.dot(g_vec, n) * n
        a_mag = np.linalg.norm(a_parallel)

        if a_mag < 1e-8:
            self.vel *= 0.0
            return

        direction = a_parallel / a_mag
        a = a_mag / (1.0 + I_SPHERE / (self.radius**2))

        self.vel += a * direction * dt
        self.vel -= np.dot(self.vel, n) * n
        self.pos += self.vel * dt

        signed_dist = np.dot(self.pos, n) - plane_offset
        self.pos = self.pos - signed_dist * n + n * self.radius


# ------------------------------------------------------------------
#  Link & simulator
# ------------------------------------------------------------------
class TwoBarLink:
    def __init__(
        self,
        base_point: Tuple[float, float, float],
        contact_point_local: Tuple[float, float],
    ):
        self.base: NDArray[np.float64] = np.array(base_point, dtype=float)
        self.contact_local: NDArray[np.float64] = np.array(
            [contact_point_local[0], contact_point_local[1], 0.0]
        )
        self.l1: float = MOTOR_LINK_LEN
        self.l2: float = PUSH_LINK_LEN


class StewartPlatformSimulator:
    def __init__(
        self,
        dt: float = DT,
        bases: List[Tuple[float, float, float]] = BASES,
        contacts_local: List[Tuple[float, float]] = CONTACTS,
        ball_pos: Tuple[float, float] = (1,0)
    ):
        self.dt: float = dt
        self.plane_pose: NDArray[np.float64] = (0,0,TABLE_HEIGHT)  # [roll, pitch, height]
        self.links: List[TwoBarLink] = [
            TwoBarLink(bases[i], contacts_local[i]) for i in range(3)
        ]
        self.ball: Ball = Ball(pos=(ball_pos[0], ball_pos[1], self.plane_pose[2] + 0.02))
        self.sim_time: float = 0.0

    def step(self, target_pose: Tuple[float, float, float]) -> None:
        self.plane_pose = np.array(target_pose, dtype=float)
        self.ball.update(self.dt, self.plane_pose)
        self.sim_time += self.dt

