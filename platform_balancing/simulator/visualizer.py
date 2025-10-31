import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Tuple, List, Optional, Any
from numpy.typing import NDArray
from settings import Settings
from kinematics import rotation_matrix, leg_points_rigid

def create_circular_platform(
    ax: plt.Axes,
    roll: float,
    pitch: float,
    z: float,
    radius: float,
    thickness: float = Settings.PLATFORM_THICKNESS,
    n_points: int = 64,
) -> Poly3DCollection:
    """Generate a circular platform mesh rotated by roll/pitch."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x_local = radius * np.cos(theta)
    y_local = radius * np.sin(theta)
    z_local_top = np.full_like(theta, thickness)
    z_local_bottom = np.zeros_like(theta)

    top_ring = np.column_stack((x_local, y_local, z_local_top))
    bottom_ring = np.column_stack((x_local, y_local, z_local_bottom))

    R = rotation_matrix(roll, pitch)
    top_ring_rot = (R @ top_ring.T).T + np.array([0.0, 0.0, z])
    bottom_ring_rot = (R @ bottom_ring.T).T + np.array([0.0, 0.0, z])

    faces = []

    # Top + bottom faces
    faces.append([top_ring_rot[i] for i in range(n_points)])
    faces.append([bottom_ring_rot[i] for i in reversed(range(n_points))])

    # Sides
    for i in range(n_points):
        j = (i + 1) % n_points
        faces.append(
            [top_ring_rot[i], top_ring_rot[j], bottom_ring_rot[j], bottom_ring_rot[i]]
        )

    poly = Poly3DCollection(
        faces,
        facecolors=(0.4, 0.8, 0.4, 0.7),
        edgecolor="k",
        linewidths=0.5,
        alpha=0.3,
    )
    ax.add_collection3d(poly)
    return poly


class Visualizer:
    def __init__(self, sim):
        self.sim = sim
        axis_limit = 0.2
        # Setup figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax: plt.Axes3D = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("Stewart Platform Simulation")
        self.ax.set_xlim(-axis_limit, axis_limit)
        self.ax.set_ylim(-axis_limit, axis_limit)
        self.ax.set_zlim(0.0, axis_limit)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.view_init(elev=30, azim=-60)

        # Elements
        self.platform: Optional[Poly3DCollection] = create_circular_platform(
            self.ax, 0, 0, Settings.TABLE_HEIGHT, Settings.PLATFORM_RADIUS
        )
        self.motor_lines: List[Any] = [
            self.ax.plot([], [], [], color="steelblue", lw=5)[0] for _ in Settings.BASES
        ]
        self.push_lines: List[Any] = [
            self.ax.plot([], [], [], color="darkorange", lw=5)[0] for _ in Settings.BASES
        ]
        self.ball_scatter = self.ax.scatter([], [], [], c="red", s=250, depthshade=True)
        (self.trail_line,) = self.ax.plot([], [], [], "r-", lw=1, alpha=0.6)
        self.trail: List[List[float]] = [[], [], []]

        plt.ion()
        plt.show()

    def update(self):
        """Render the current state of the simulator."""
        sim = self.sim
        roll, pitch, z = sim.plane_pose
        ball_pos = sim.ball.pos

        # Update platform
        if self.platform is not None:
            self.platform.remove()
        self.platform = create_circular_platform(
            self.ax, roll, pitch, z, Settings.PLATFORM_RADIUS
        )

        # Update legs
        for i, link in enumerate(sim.links):
            segments = leg_points_rigid(
                link.base, link.contact_local, (roll, pitch, z), link.l1, link.l2
            )
            if segments is None:
                self.motor_lines[i].set_data_3d([], [], [])
                self.push_lines[i].set_data_3d([], [], [])
                continue
            (b1, br), (br2, p) = segments
            self.motor_lines[i].set_data_3d(
                [b1[0], br[0]], [b1[1], br[1]], [b1[2], br[2]]
            )
            self.push_lines[i].set_data_3d(
                [br2[0], p[0]], [br2[1], p[1]], [br2[2], p[2]]
            )

        # Ball
        self.ball_scatter._offsets3d = ([ball_pos[0]], [ball_pos[1]], [ball_pos[2]])

        # Trail
        self.trail[0].append(ball_pos[0])
        self.trail[1].append(ball_pos[1])
        self.trail[2].append(ball_pos[2])
        if len(self.trail[0]) > 300:
            for i in range(3):
                self.trail[i] = self.trail[i][-300:]
        self.trail_line.set_data_3d(self.trail[0], self.trail[1], self.trail[2])

        plt.draw()
        plt.pause(0.001)
