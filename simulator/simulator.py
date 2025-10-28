import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import time

# ------------------------------------------------------------------
#  Constants
# ------------------------------------------------------------------
MOTOR_LINK_LEN = 0.35
PUSH_LINK_LEN = 0.35
BALL_RADIUS = 0.02
G = 9.81
TABLE_HEIGHT = 0.4
PLATFORM_SIDE = 2.0
I_SPHERE = (2.0 / 5.0) * BALL_RADIUS**2

DT = 0.005
TARGET_FPS = 60


# ------------------------------------------------------------------
#  Helper functions
# ------------------------------------------------------------------
def rotation_matrix(roll, pitch):
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return R_y @ R_x


def bearing_point_exact(base, p_world, l1=MOTOR_LINK_LEN, l2=PUSH_LINK_LEN):
    B = np.array(base, dtype=float)
    P = np.array(p_world, dtype=float)
    d = np.linalg.norm(P - B)
    if d > l1 + l2 + 1e-9 or d < abs(l1 - l2) - 1e-9 or d < 1e-9:
        return False, None
    v = (P - B) / d
    a = (l1**2 - l2**2 + d**2) / (2.0 * d)
    h2 = l1**2 - a**2
    if h2 < 0:
        return False, None
    h = np.sqrt(h2)
    M = B + a * v
    tmp = (
        np.array([0.0, 0.0, 1.0])
        if abs(v[0]) > 0.1 or abs(v[1]) > 0.1
        else np.array([1.0, 0.0, 0.0])
    )
    n_perp = np.cross(v, tmp)
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
    def __init__(self, pos=(0, 0, 0)):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3)
        self.radius = BALL_RADIUS

    def update(self, dt, plane_pose):
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
        self, base_point, contact_point_local
    ):
        self.base = np.array(base_point, dtype=float)
        self.contact_local = np.array(
            [contact_point_local[0], contact_point_local[1], 0.0]
        )
        self.l1 = MOTOR_LINK_LEN
        self.l2 = PUSH_LINK_LEN


class StewartPlatformSimulator:
    def __init__(self, plane_pose, dt, bases, contacts_local):
        self.dt = dt
        self.plane_pose = np.array(plane_pose, dtype=float)
        self.links = [TwoBarLink(bases[i], contacts_local[i]) for i in range(3)]
        self.ball = Ball(pos=(0.0, 0.0, plane_pose[2] + 0.02))
        self.sim_time = 0.0

    def step(self, target_pose):
        self.plane_pose = np.array(target_pose, dtype=float)
        self.ball.update(self.dt, self.plane_pose)
        self.sim_time += self.dt


# ------------------------------------------------------------------
#  Visual helpers
# ------------------------------------------------------------------
def create_rotated_platform(ax, roll, pitch, z, side):
    half = side / 2.0
    verts_local = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
            [-half, -half, 0.01],
            [half, -half, 0.01],
            [half, half, 0.01],
            [-half, half, 0.01],
        ]
    )
    R = rotation_matrix(roll, pitch)
    verts = (R @ verts_local.T).T + np.array([0.0, 0.0, z])
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[3], verts[0], verts[4], verts[7]],
    ]
    poly = Poly3DCollection(
        faces, facecolors=(0.4, 0.8, 0.4, 0.6), edgecolor="k", linewidths=0.8
    )
    ax.add_collection3d(poly)
    return poly


def leg_points_rigid(base, contact_local, plane_pose):
    roll, pitch, z = plane_pose
    R = rotation_matrix(roll, pitch)
    P_world = R @ np.array([contact_local[0], contact_local[1], 0.0]) + np.array(
        [0.0, 0.0, z]
    )
    ok, bearing = bearing_point_exact(base, P_world)
    if not ok:
        bearing = np.array(base) + np.array([0.0, 0.0, MOTOR_LINK_LEN])
    return (np.array(base), bearing), (bearing, P_world)


# ------------------------------------------------------------------
#  Real-time animation
# ------------------------------------------------------------------
def run_real_time_simulation():
    bases = [(-0.6, -0.4, 0.0), (0.6, -0.4, 0.0), (0.0, 0.8, 0.0)]
    contacts = [(-0.6, -0.4), (0.6, -0.4), (0.0, 0.8)]

    sim = StewartPlatformSimulator(
        plane_pose=(0.0, 0.0, TABLE_HEIGHT),
        dt=DT,
        bases=bases,
        contacts_local=contacts,
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Real-Time Rolling Ball on Two-Bar Stewart Platform")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=-60)

    platform = None
    motor_lines = [ax.plot([], [], [], color="steelblue", lw=5)[0] for _ in bases]
    push_lines = [ax.plot([], [], [], color="darkorange", lw=4)[0] for _ in bases]
    ball_scatter = ax.scatter([], [], [], c="red", s=250, depthshade=True)
    (trail_line,) = ax.plot([], [], [], "r-", lw=1, alpha=0.6)
    trail = [[], [], []]

    last_frame_time = time.time()
    frame_interval = 1.0 / TARGET_FPS

    def init():
        nonlocal platform
        if platform is not None:
            platform.remove()
        platform = create_rotated_platform(ax, 0, 0, TABLE_HEIGHT, PLATFORM_SIDE)
        return (platform, *motor_lines, *push_lines, ball_scatter, trail_line)

    def update(frame):
        nonlocal platform, last_frame_time, trail

        now = time.time()
        elapsed = now - last_frame_time
        if elapsed < frame_interval:
            return (platform, *motor_lines, *push_lines, ball_scatter, trail_line)
        last_frame_time = now

        steps = int(elapsed / DT) + 1
        for _ in range(steps):
            t = sim.sim_time
            roll = 0.25 * math.sin(2 * math.pi * t * 0.5)
            pitch = 0.20 * math.cos(2 * math.pi * t * 0.5)
            z = TABLE_HEIGHT
            sim.step((roll, pitch, z))

        roll, pitch, z = sim.plane_pose
        ball_pos = sim.ball.pos

        if platform is not None:
            platform.remove()
        platform = create_rotated_platform(ax, roll, pitch, z, PLATFORM_SIDE)

        for i, (base, contact) in enumerate(zip(bases, contacts)):
            (b1, br), (br2, p) = leg_points_rigid(base, contact, (roll, pitch, z))
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
