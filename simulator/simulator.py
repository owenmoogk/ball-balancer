import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import time
from typing import Tuple, List, Optional, Sequence, Any
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
        plane_pose: Tuple[float, float, float],
        dt: float,
        bases: List[Tuple[float, float, float]],
        contacts_local: List[Tuple[float, float]],
    ):
        self.dt: float = dt
        self.plane_pose: NDArray[np.float64] = np.array(plane_pose, dtype=float)
        self.links: List[TwoBarLink] = [
            TwoBarLink(bases[i], contacts_local[i]) for i in range(3)
        ]
        self.ball: Ball = Ball(pos=(0.0, 0.0, plane_pose[2] + 0.02))
        self.sim_time: float = 0.0

    def step(self, target_pose: Tuple[float, float, float]) -> None:
        self.plane_pose = np.array(target_pose, dtype=float)
        self.ball.update(self.dt, self.plane_pose)
        self.sim_time += self.dt


# ------------------------------------------------------------------
#  Visual helpers – CIRCULAR PLATFORM
# ------------------------------------------------------------------
def create_circular_platform(
    ax: plt.Axes,
    roll: float,
    pitch: float,
    z: float,
    radius: float,
    thickness: float = PLATFORM_THICKNESS,
    n_points: int = 64,
) -> Poly3DCollection:
    # Create local circle vertices
    theta = np.linspace(0, 2 * np.pi, n_points)
    x_local = radius * np.cos(theta)
    y_local = radius * np.sin(theta)
    z_local_top = np.full_like(theta, thickness)
    z_local_bottom = np.zeros_like(theta)

    # Stack top and bottom circles
    top_ring = np.column_stack((x_local, y_local, z_local_top))
    bottom_ring = np.column_stack((x_local, y_local, z_local_bottom))

    # Rotate and translate
    R = rotation_matrix(roll, pitch)
    top_ring_rot = (R @ top_ring.T).T + np.array([0.0, 0.0, z])
    bottom_ring_rot = (R @ bottom_ring.T).T + np.array([0.0, 0.0, z])

    # Create faces
    faces = []

    # Top face
    top_face = [top_ring_rot[i] for i in range(n_points)]
    faces.append(top_face)

    # Bottom face
    bottom_face = [bottom_ring_rot[i] for i in reversed(range(n_points))]
    faces.append(bottom_face)

    # Side faces (quads between top and bottom)
    for i in range(n_points):
        j = (i + 1) % n_points
        faces.append(
            [top_ring_rot[i], top_ring_rot[j], bottom_ring_rot[j], bottom_ring_rot[i]]
        )

    # Create collection
    poly = Poly3DCollection(
        faces,
        facecolors=(0.4, 0.8, 0.4, 0.7),
        edgecolor="k",
        linewidths=0.5,
        alpha=0.8,
        zorder=1
    )
    ax.add_collection3d(poly)
    return poly


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
    P_world = R @ np.array([contact_local[0], contact_local[1], 0.0]) + np.array(
        [0.0, 0.0, z]
    )
    ok, bearing = bearing_point_exact(base, P_world, l1, l2)
    if not ok:
        return None
    return (base.copy(), bearing), (bearing.copy(), P_world.copy())


# ------------------------------------------------------------------
#  Real-time animation
# ------------------------------------------------------------------
def run_real_time_simulation() -> None:
    sim = StewartPlatformSimulator(
        plane_pose=(0.0, 0.0, TABLE_HEIGHT),
        dt=DT,
        bases=BASES,
        contacts_local=CONTACTS,
    )

    fig = plt.figure(figsize=(10, 8))
    ax: plt.Axes3D = fig.add_subplot(111, projection="3d")
    ax.set_title("Real-Time Rolling Ball on Circular 120° Stewart Platform")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=-60)

    platform: Optional[Poly3DCollection] = None
    motor_lines: List[Any] = [
        ax.plot([], [], [], color="steelblue", lw=5)[0] for _ in BASES
    ]
    push_lines: List[Any] = [
        ax.plot([], [], [], color="darkorange", lw=4)[0] for _ in BASES
    ]
    ball_scatter = ax.scatter([], [], [], c="red", s=250, depthshade=True)
    (trail_line,) = ax.plot([], [], [], "r-", lw=1, alpha=0.6)
    trail: List[List[float]] = [[], [], []]

    last_frame_time = time.time()
    frame_interval = 1.0 / TARGET_FPS

    def init() -> Tuple[Any, ...]:
        nonlocal platform
        if platform is not None:
            platform.remove()
        platform = create_circular_platform(ax, 0, 0, TABLE_HEIGHT, PLATFORM_RADIUS)
        return (platform, *motor_lines, *push_lines, ball_scatter, trail_line)

    def update(frame: int) -> Tuple[Any, ...]:
        nonlocal platform, last_frame_time, trail

        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < frame_interval:
            return (platform, *motor_lines, *push_lines, ball_scatter, trail_line)
        last_frame_time = now

        steps = int(elapsed / DT) + 1
        for _ in range(steps):
            t = sim.sim_time
            roll = 0.05*t * math.sin(2 * math.pi * t)
            pitch = 0.05*t * math.cos(2 * math.pi * t)
            z = TABLE_HEIGHT
            sim.step((roll, pitch, z))

        roll, pitch, z = sim.plane_pose
        ball_pos = sim.ball.pos

        if platform is not None:
            platform.remove()
        platform = create_circular_platform(ax, roll, pitch, z, PLATFORM_RADIUS)

        for i, link in enumerate(sim.links):
            segments = leg_points_rigid(
                link.base, link.contact_local, (roll, pitch, z), link.l1, link.l2
            )
            if segments is None:
                motor_lines[i].set_data_3d([], [], [])
                push_lines[i].set_data_3d([], [], [])
                continue
            (b1, br), (br2, p) = segments
            motor_lines[i].set_data_3d([b1[0], br[0]], [b1[1], br[1]], [b1[2], br[2]])
            push_lines[i].set_data_3d([br2[0], p[0]], [br2[1], p[1]], [br2[2], p[2]])

        ball_scatter._offsets3d = ([ball_pos[0]], [ball_pos[1]], [ball_pos[2]])

        trail[0].append(ball_pos[0])
        trail[1].append(ball_pos[1])
        trail[2].append(ball_pos[2])
        if len(trail[0]) > 300:
            for i in range(3):
                trail[i] = trail[i][-300:]
        trail_line.set_data_3d(trail[0], trail[1], trail[2])

        return (platform, *motor_lines, *push_lines, ball_scatter, trail_line)

    animation.FuncAnimation(
        fig, update, init_func=init, interval=0, blit=False, repeat=True
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_real_time_simulation()
