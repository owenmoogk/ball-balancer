from simulator import *  # noqa: F403
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import time
from typing import Tuple, List, Optional, Any
from numpy.typing import NDArray

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
        zorder=1,
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
            roll = 0.05 * t * math.sin(2 * math.pi * t)
            pitch = 0.05 * t * math.cos(2 * math.pi * t)
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

    ani = animation.FuncAnimation(
        fig, update, init_func=init, interval=0, blit=False, repeat=True
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_real_time_simulation()
