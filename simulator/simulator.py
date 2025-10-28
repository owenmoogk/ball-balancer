# --------------------------------------------------------------
#  TWO-BAR LINKAGE STEWART PLATFORM (3 legs) – RIGID LINKS
# --------------------------------------------------------------
import numpy as np
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------------------------------------------------------
#  Geometry & physics constants
# ------------------------------------------------------------------
MOTOR_LINK_LEN = 0.35   # length of the motor-driven bar (L1)
PUSH_LINK_LEN  = 0.35   # length of the push-bar (L2)
BALL_RADIUS    = 0.02
G              = 9.81
TABLE_HEIGHT   = 0.4   # nominal height of the platform centre

# ------------------------------------------------------------------
#  Helper functions
# ------------------------------------------------------------------
def clamp(x, a=-1.0, b=1.0):
    return max(a, min(b, x))

def rotation_matrix(roll, pitch):
    """R = R_y(pitch) @ R_x(roll)"""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return R_y @ R_x

# ------------------------------------------------------------------
#  Exact two-circle intersection for ONE leg
# ------------------------------------------------------------------
def bearing_point_exact(base, p_world,
                        l1=MOTOR_LINK_LEN, l2=PUSH_LINK_LEN):
    """
    Returns (reachable, bearing_point)
    bearing_point is the intersection of
        circle centre = base, radius = l1
        circle centre = p_world, radius = l2
    We keep the solution that lies *below* the line connecting base↔p_world
    (the usual "down" configuration for a Stewart leg).
    """
    B = np.array(base, dtype=float)
    P = np.array(p_world, dtype=float)

    d = np.linalg.norm(P - B)                # distance base ↔ platform joint
    if d > l1 + l2 + 1e-9 or d < abs(l1 - l2) - 1e-9 or d < 1e-9:
        return False, None

    # ---- vector from B to P ------------------------------------------------
    v = (P - B) / d

    # ---- distance from B to the intersection line -------------------------
    a = (l1**2 - l2**2 + d**2) / (2.0 * d)

    # ---- height of the intersection triangle -------------------------------
    h = np.sqrt(l1**2 - a**2)                # may be NaN if out of reach

    # ---- centre of the radical line ----------------------------------------
    M = B + a * v

    # ---- perpendicular vector (in the plane of the leg) --------------------
    # we build a local orthonormal basis (v, n_perp)
    # pick a vector not collinear with v
    if abs(v[0]) > 0.1 or abs(v[1]) > 0.1:
        tmp = np.array([0.0, 0.0, 1.0])
    else:
        tmp = np.array([1.0, 0.0, 0.0])
    n_perp = np.cross(v, tmp)
    n_perp /= np.linalg.norm(n_perp)

    # ---- two possible bearing points ---------------------------------------
    bearing1 = M + h * n_perp
    bearing2 = M - h * n_perp

    # ---- choose the "lower" solution (dot with gravity) --------------------
    gravity = np.array([0.0, 0.0, -1.0])
    if np.dot(bearing1, gravity) < np.dot(bearing2, gravity):
        bearing = bearing1
    else:
        bearing = bearing2

    return True, bearing

# ------------------------------------------------------------------
#  Motor angle (for debugging / optional display)
# ------------------------------------------------------------------
def motor_angle_from_bearing(base, bearing):
    """Angle of the motor bar measured from the upward vertical."""
    vec = np.array(bearing) - np.array(base)
    vec /= np.linalg.norm(vec)
    # angle with +Z axis
    return np.arccos(vec[2])

# ------------------------------------------------------------------
#  Ball physics (unchanged)
# ------------------------------------------------------------------
def plane_normal_from_pose(roll, pitch):
    R = rotation_matrix(roll, pitch)
    n = R @ np.array([0.0, 0.0, 1.0])
    return n / np.linalg.norm(n)

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
            a_parallel /= 1.0 + 2.0/5.0

        self.accel = a_parallel
        self.vel += self.accel * dt
        self.vel -= np.dot(self.vel, n) * n
        self.pos += self.vel * dt

        signed_dist = np.dot(self.pos, n) - plane_offset
        self.pos = self.pos - signed_dist * n + n * self.radius

        return self.pos.copy(), self.vel.copy(), self.accel.copy()

# ------------------------------------------------------------------
#  Link class
# ------------------------------------------------------------------
class TwoBarLink:
    def __init__(self, base_point, contact_point_local,
                 l1=MOTOR_LINK_LEN, l2=PUSH_LINK_LEN):
        self.base = np.array(base_point, dtype=float)
        self.contact_local = np.array([contact_point_local[0],
                                       contact_point_local[1], 0.0])
        self.l1 = l1
        self.l2 = l2

# ------------------------------------------------------------------
#  Simulator
# ------------------------------------------------------------------
class StewartPlatformSimulator:
    def __init__(self, plane_pose, dt, bases, contacts_local):
        self.dt = dt
        self.plane_pose = np.array(plane_pose, dtype=float)
        self.links = [TwoBarLink(bases[i], contacts_local[i])
                      for i in range(3)]
        self.ball = Ball(pos=(0.0, 0.0, plane_pose[2] + 0.02))

    def get_motor_angles(self):
        """Return reachable flags and motor angles (for info only)."""
        reachable, angles = [], []
        for lnk in self.links:
            R = rotation_matrix(*self.plane_pose[:2])
            P = R @ lnk.contact_local + np.array([0, 0, self.plane_pose[2]])
            ok, bearing = bearing_point_exact(lnk.base, P,
                                             lnk.l1, lnk.l2)
            reachable.append(ok)
            if ok:
                ang = motor_angle_from_bearing(lnk.base, bearing)
            else:
                ang = 0.0
            angles.append(ang)
        return reachable, angles

    def step(self, target_plane_pose, rolling=True):
        self.plane_pose = np.array(target_plane_pose, dtype=float)
        reach, angles = self.get_motor_angles()

        pos, vel, acc = self.ball.update(self.dt, self.plane_pose, rolling=rolling)
        return {
            "plane_pose": self.plane_pose.copy(),
            "ball_pos"  : pos.copy(),
            "motor_angles": angles,
            "reachable" : reach,
        }

# ------------------------------------------------------------------
#  Visualisation helpers
# ------------------------------------------------------------------
def create_rotated_platform(ax, roll, pitch, z, side):
    half = side / 2.0
    verts_local = np.array([
        [-half, -half, 0.0], [ half, -half, 0.0],
        [ half,  half, 0.0], [-half,  half, 0.0],
        [-half, -half, 0.01],[ half, -half, 0.01],
        [ half,  half, 0.01],[-half,  half, 0.01],
    ])
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
    poly = Poly3DCollection(faces,
                            facecolors=(0.4, 0.8, 0.4, 0.6),
                            edgecolor='k', linewidths=0.8)
    ax.add_collection3d(poly)
    return poly

def leg_points_rigid(base, contact_local, plane_pose):
    """
    Returns ((base, bearing), (bearing, platform_joint))
    Both segments have *exactly* the prescribed lengths.
    """
    roll, pitch, z = plane_pose
    R = rotation_matrix(roll, pitch)
    P_world = R @ np.array([contact_local[0], contact_local[1], 0.0]) + np.array([0.0, 0.0, z])

    ok, bearing = bearing_point_exact(base, P_world,
                                     MOTOR_LINK_LEN, PUSH_LINK_LEN)
    if not ok:
        # fallback – just draw a short line so the animation never crashes
        bearing = np.array(base) + np.array([0.0, 0.0, MOTOR_LINK_LEN])

    return (np.array(base), bearing), (bearing, P_world)

# ------------------------------------------------------------------
#  Main visualisation routine
# ------------------------------------------------------------------
def run_visual_simulation():
    bases    = [(-0.6, -0.4, 0.0), ( 0.6, -0.4, 0.0), (0.0, 0.8, 0.0)]
    contacts = [(-0.6, -0.4),     ( 0.6, -0.4),     (0.0, 0.8)]

    sim = StewartPlatformSimulator(
        plane_pose=(0.0, 0.0, TABLE_HEIGHT),
        dt=0.01,
        bases=bases,
        contacts_local=contacts,
    )

    # ---------- Matplotlib figure ----------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Rigid Two-Bar Stewart Platform + Rolling Ball')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(0.0, 1.0)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=-60)

    platform = None

    # Two line objects per leg
    motor_lines = [ax.plot([], [], [], color='steelblue', lw=5)[0] for _ in bases]
    push_lines  = [ax.plot([], [], [], color='darkorange', lw=4)[0] for _ in bases]

    ball_scatter = ax.scatter([], [], [], c='red', s=250, depthshade=True)
    trail_line, = ax.plot([], [], [], 'r-', lw=1, alpha=0.6)
    trail_x, trail_y, trail_z = [], [], []

    # ------------------------------------------------------------------
    def init():
        nonlocal platform
        if platform is not None:
            platform.remove()
        platform = create_rotated_platform(ax, 0, 0, TABLE_HEIGHT, side=2.0)
        return (platform, *motor_lines, *push_lines, ball_scatter, trail_line)

    def update(frame):
        nonlocal platform, trail_x, trail_y, trail_z

        t = frame * sim.dt
        roll  = 0.1 * math.sin(2 * math.pi * t * 0.5)
        pitch = 0.1 * math.cos(2 * math.pi * t * 0.5)
        z = TABLE_HEIGHT
        target_pose = (roll, pitch, z)

        state = sim.step(target_pose, rolling=True)
        roll, pitch, z = state['plane_pose']
        ball_pos = state['ball_pos']

        # ---- platform ------------------------------------------------
        if platform is not None:
            platform.remove()
        platform = create_rotated_platform(ax, roll, pitch, z, side=2.0)

        # ---- legs (rigid) --------------------------------------------
        for i, (base, contact) in enumerate(zip(bases, contacts)):
            (base_pt, bearing), (bearing_pt, plat_pt) = leg_points_rigid(base, contact, (roll, pitch, z))

            motor_lines[i].set_data_3d(
                [base_pt[0], bearing[0]],
                [base_pt[1], bearing[1]],
                [base_pt[2], bearing[2]]
            )
            push_lines[i].set_data_3d(
                [bearing_pt[0], plat_pt[0]],
                [bearing_pt[1], plat_pt[1]],
                [bearing_pt[2], plat_pt[2]]
            )

        # ---- ball ----------------------------------------------------
        ball_scatter._offsets3d = ([ball_pos[0]], [ball_pos[1]], [ball_pos[2]])

        # ---- trail ---------------------------------------------------
        trail_x.append(ball_pos[0]); trail_y.append(ball_pos[1]); trail_z.append(ball_pos[2])
        N = 300
        if len(trail_x) > N:
            trail_x = trail_x[-N:]; trail_y = trail_y[-N:]; trail_z = trail_z[-N:]
        trail_line.set_data_3d(trail_x, trail_y, trail_z)

        return (platform, *motor_lines, *push_lines, ball_scatter, trail_line)

    # ------------------------------------------------------------------
    T = 25.0
    steps = int(T / sim.dt)
    ani = animation.FuncAnimation(
        fig, update, frames=steps,
        init_func=init,
        interval=sim.dt*1000,
        blit=False, repeat=False)

    plt.tight_layout()
    plt.show()
    print("Simulation finished.")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    run_visual_simulation()