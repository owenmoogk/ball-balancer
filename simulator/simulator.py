import numpy as np
import math

MOTOR_LINK_LEN = 10.0
BAR_LINK_LEN = 10.0
BALL_RADIUS = 0.02
G = 9.81
TABLE_HEIGHT = 0.4


def clamp(x, a=-1.0, b=1.0):
    return max(a, min(b, x))


def rotation_matrix(roll, pitch):
    """R = R_y(pitch) @ R_x(roll)"""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return R_y @ R_x


def leg_motor_pitch_from_base_and_platform(
    base, p_local, plane_pose, l1=MOTOR_LINK_LEN, l2=BAR_LINK_LEN
):
    roll, pitch, z = plane_pose
    R = rotation_matrix(roll, pitch)
    P_world = R @ np.array(p_local) + np.array([0.0, 0.0, z])
    v = P_world - np.array(base)
    r = np.hypot(v[0], v[1])
    h = v[2]
    d = np.hypot(r, h)
    if d < 1e-8:
        return False, []
    cos_gamma = (l1**2 + d**2 - l2**2) / (2.0 * l1 * d)
    cos_gamma = clamp(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    delta = np.arctan2(h, r)
    theta_down = delta + gamma
    theta_up = delta - gamma
    reachable = (abs(l1 - l2) - 1e-9) <= d <= (l1 + l2 + 1e-9)
    return reachable, [theta_down, theta_up]


def plane_normal_from_pose(roll, pitch):
    R = rotation_matrix(roll, pitch)
    n = R @ np.array([0.0, 0.0, 1.0])
    return n / np.linalg.norm(n)


def ball_acceleration_on_plane(roll, pitch, ball_radius, rolling=True, g=G):
    n = plane_normal_from_pose(roll, pitch)
    g_vec = np.array([0, 0, -g])
    a_parallel = g_vec - np.dot(g_vec, n) * n
    if np.linalg.norm(a_parallel) > 0:
        a_parallel /= np.linalg.norm(a_parallel)
        a_mag = g * math.sin(math.acos(np.dot(n, [0, 0, 1])))
        if rolling:
            a_mag /= 1 + (2 / 5)
        return a_parallel * a_mag
    return np.zeros(3)


class Link:
    def __init__(
        self, base_point, contact_point_local, l1=MOTOR_LINK_LEN, l2=BAR_LINK_LEN
    ):
        self.base = np.array(base_point, dtype=float)
        self.contact_local = np.array(
            [contact_point_local[0], contact_point_local[1], 0.0]
        )
        self.l1 = l1
        self.l2 = l2


class Ball:
    def __init__(self, pos=(0, 0, 0)):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3)
        self.accel = np.zeros(3)
        self.radius = BALL_RADIUS

    def update(self, dt, plane_pose, rolling=True):
        roll, pitch, z = plane_pose
        R = rotation_matrix(roll, pitch)
        n = R @ np.array([0.0, 0.0, 1.0])
        n = n / np.linalg.norm(n)

        plane_point = np.array([0.0, 0.0, z])
        plane_offset = np.dot(n, plane_point)

        g_vec = np.array([0.0, 0.0, -G])
        a_parallel = g_vec - np.dot(g_vec, n) * n
        if rolling:
            a_parallel /= 1.0 + 2.0 / 5.0

        self.accel = a_parallel
        self.vel += self.accel * dt
        self.vel -= np.dot(self.vel, n) * n
        self.pos += self.vel * dt

        signed_dist = np.dot(self.pos, n) - plane_offset
        self.pos = self.pos - signed_dist * n + n * self.radius

        return self.pos.copy(), self.vel.copy(), self.accel.copy()


class StewartPlatformSimulator:
    def __init__(self, plane_pose, dt, bases, contacts_local):
        self.dt = dt
        self.plane_pose = plane_pose
        self.links = [Link(bases[i], contacts_local[i]) for i in range(3)]
        self.ball = Ball(pos=(0.0, 0.0, plane_pose[2] + 0.01))

    def get_motor_angles(self):
        angles = []
        reachable_flags = []
        for link in self.links:
            reachable, sols = leg_motor_pitch_from_base_and_platform(
                link.base, link.contact_local, self.plane_pose, link.l1, link.l2
            )
            reachable_flags.append(reachable)
            angles.append(sols)
        return reachable_flags, angles

    def step(self, target_plane_pose, rolling=True):
        self.plane_pose = target_plane_pose

        reachable, angles = self.get_motor_angles()
        pos, vel, acc = self.ball.update(self.dt, self.plane_pose, rolling=rolling)

        return {
            "plane_pose": self.plane_pose,
            "ball_pos": pos.copy(),
        }


# --------------------------------------------------------------
#  MATPLOTLIB 3-D VISUALISATION
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def run_visual_simulation():
    bases = [(-0.6, -0.4, 0), (0.6, -0.4, 0), (0, 0.8, 0)]
    contacts = [(-0.6, -0.4), (0.6, -0.4), (0, 0.8)]
    sim = StewartPlatformSimulator(
        plane_pose=(0.0, 0.0, TABLE_HEIGHT),
        dt=0.01,
        bases=bases,
        contacts_local=contacts,
    )

    # === Matplotlib figure ===
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Stewart Platform – Matplotlib 3-D")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(0, 1.0)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=-60)

    # --- platform (thin box) ---
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection


    # --- Add this at the top with other imports ---
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # --- Replace the platform initialization ---
    platform = None  # will be a Poly3DCollection


    def create_rotated_platform(ax, roll, pitch, z, size):
        """Create a rotated platform as Poly3DCollection"""
        half = size / 2
        # 8 corners of the platform (local frame, z=0)
        verts_local = np.array(
            [
                [-half, -half, 0],
                [half, -half, 0],
                [half, half, 0],
                [-half, half, 0],
                [-half, -half, 0.01],
                [half, -half, 0.01],
                [half, half, 0.01],
                [-half, half, 0.01],
            ]
        )

        R = rotation_matrix(roll, pitch)
        verts_rot = (R @ verts_local.T).T + np.array([0, 0, z])

        # 6 faces (quads)
        faces = [
            [verts_rot[0], verts_rot[1], verts_rot[2], verts_rot[3]],  # bottom
            [verts_rot[4], verts_rot[5], verts_rot[6], verts_rot[7]],  # top
            [verts_rot[0], verts_rot[1], verts_rot[5], verts_rot[4]],  # front
            [verts_rot[1], verts_rot[2], verts_rot[6], verts_rot[5]],  # right
            [verts_rot[2], verts_rot[3], verts_rot[7], verts_rot[6]],  # back
            [verts_rot[3], verts_rot[0], verts_rot[4], verts_rot[7]],  # left
        ]

        poly = Poly3DCollection(
            faces, facecolors=(0.4, 0.8, 0.4, 0.6), edgecolor="k", linewidths=1
        )
        ax.add_collection3d(poly)
        return poly

    # --- legs (cylinders) ---
    leg_artists = []
    for b in bases:
        leg = ax.plot([], [], [], color="gray", linewidth=4)[0]
        leg_artists.append(leg)

    # --- ball (scatter) ---
    ball_scatter = ax.scatter([], [], [], c="red", s=200, depthshade=True)

    # --- ball trail (line) ---
    (trail_line,) = ax.plot([], [], [], "r-", linewidth=1, alpha=0.6)

    trail_x, trail_y, trail_z = [], [], []

    # ------------------------------------------------------------------
    def init():
        nonlocal platform
        if platform is not None:
            platform.remove()
        platform = create_rotated_platform(ax, 0, 0, TABLE_HEIGHT, TABLE_HEIGHT*2)
        return platform, *leg_artists, ball_scatter, trail_line

    def update(frame):
        # ---- platform motion (same as original) ----
        t = frame * sim.dt
        roll = 1 * math.sin(2 * math.pi * t * 0.5)
        pitch = 0.2 * math.cos(2 * math.pi * t * 0.5)
        z = TABLE_HEIGHT
        target_pose = (roll, pitch, z)

        state = sim.step(target_pose)
        roll, pitch, z = state["plane_pose"]
        ball_pos = state["ball_pos"]

        R = rotation_matrix(roll, pitch)

        # ---- update platform ----
        # remove old bar3d and draw new one (Matplotlib has no mutable bar3d)
        nonlocal platform
        if platform is not None:
            platform.remove()
        platform = create_rotated_platform(ax, roll, pitch, z, TABLE_HEIGHT*2)

        # rotate the platform surface by setting its normal as "up"
        # (visual trick: we tilt a thin box by rotating its vertices)
        # For simplicity we keep the box axis-aligned and just move it.

        # ---- update legs ----
        for i, leg in enumerate(leg_artists):
            p_local = np.array([contacts[i][0], contacts[i][1], 0.0])
            P_world = R @ p_local + np.array([0.0, 0.0, z])
            base_pt = np.array(bases[i])
            leg.set_data_3d(
                [base_pt[0], P_world[0]],
                [base_pt[1], P_world[1]],
                [base_pt[2], P_world[2]],
            )

        # ---- update ball ----
        ball_scatter._offsets3d = ([ball_pos[0]], [ball_pos[1]], [ball_pos[2]])

        # ---- update trail ----
        nonlocal trail_x, trail_y, trail_z
        trail_x.append(ball_pos[0])
        trail_y.append(ball_pos[1])
        trail_z.append(ball_pos[2])
        # keep only last N points
        N = 300
        if len(trail_x) > N:
            trail_x = trail_x[-N:]
            trail_y = trail_y[-N:]
            trail_z = trail_z[-N:]
        trail_line.set_data_3d(trail_x, trail_y, trail_z)

        return platform, *leg_artists, ball_scatter, trail_line

    # ------------------------------------------------------------------
    T = 25.0
    steps = int(T / sim.dt)
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        init_func=init,
        interval=sim.dt * 1000 * 0.5,
        blit=False,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()
    print("Simulation complete.")


if __name__ == "__main__":
    run_visual_simulation()
