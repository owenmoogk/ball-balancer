from vpython import sphere, box, cylinder, vector, rate, scene
import numpy as np
import math
from time import sleep

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
    # component of g parallel to plane
    a_parallel = g_vec - np.dot(g_vec, n) * n
    if np.linalg.norm(a_parallel) > 0:
        a_parallel /= np.linalg.norm(a_parallel)  # direction along plane
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
        n = R @ np.array([0.0, 0.0, 1.0])  # normal in world frame
        n = n / np.linalg.norm(n)

        # Plane passes through (0,0,z), so plane equation: n · (x - [0,0,z]) = 0
        plane_point = np.array([0.0, 0.0, z])
        plane_offset = np.dot(n, plane_point)  # n · r0

        # Gravity acceleration parallel to plane
        g_vec = np.array([0.0, 0.0, -G])
        a_parallel = g_vec - np.dot(g_vec, n) * n

        if rolling:
            a_parallel /= (1.0 + 2.0/5.0)  # I = 2/5 m r^2 for solid sphere

        self.accel = a_parallel

        # Integrate velocity
        self.vel += self.accel * dt

        # Remove any velocity component perpendicular to plane (frictionless in normal direction)
        self.vel -= np.dot(self.vel, n) * n

        # Integrate position
        self.pos += self.vel * dt

        # === PROJECT BALL CENTER ONTO PLANE + OFFSET BY RADIUS ===
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


def run_visual_simulation():
    # === setup simulation ===
    bases = [(-0.6, -0.4, 0), (0.6, -0.4, 0), (0, 0.8, 0)]
    contacts = [(-0.6, -0.4), (0.6, -0.4), (0, 0.8)]    
    sim = StewartPlatformSimulator(
        plane_pose=(0.0, 0.0, TABLE_HEIGHT), dt=0.01, bases=bases, contacts_local=contacts
    )

    # === setup VPython scene ===
    scene.title = "Stewart Platform Simulation"
    scene.center = vector(0, 0, 0)
    scene.forward = vector(0, 1, -0.3)
    scene.width = 1200
    scene.height = 800
    scene.range = 2  # zoom out so we see the entire structure

    platform = box(
        pos=vector(0, 0,TABLE_HEIGHT),
        size=vector(TABLE_HEIGHT, TABLE_HEIGHT, 0.01),
        color=vector(0.4, 0.8, 0.4),
        opacity=0.6,
        up=vector(0,0,1)
    )


    # legs = [cylinder(pos=vector(*b), axis=vector(0,0,TABLE_HEIGHT), radius=0.03) for b in bases]
    # scale ball to visible size (1.0 instead of 0.05)
    ball = sphere(
        pos=vector(0, 0, TABLE_HEIGHT + BALL_RADIUS),
        color=vector(1, 0.2, 0.2),
        radius=BALL_RADIUS,
        make_trail=True,
    )

    # === simulation loop ===
    T = 25.0
    steps = int(T / sim.dt)
    for step in range(steps):
        rate(100)

        # oscillate the platform (you can replace this motion)
        roll = 0.1 * math.sin(2 * math.pi * step * sim.dt * 0.5/1000)
        pitch = 0.1 * math.cos(2 * math.pi * step * sim.dt * 0.5)
        z = TABLE_HEIGHT
        target_pose = (roll, pitch, z)

        state = sim.step(target_pose)
        roll, pitch, z = state["plane_pose"]
        ball_pos = state["ball_pos"]
        R = rotation_matrix(roll, pitch)

        # --- update platform transform ---
        platform.pos = vector(0, 0, z)
        # platform.axis = vector(R[0, 2], R[1, 2], R[2, 2])
        platform.up = vector(R[0, 1], R[1, 1], R[2, 1])

        # --- update leg geometry ---
        # for i, leg in enumerate(legs):
        #     p_local = np.array([contacts[i][0], contacts[i][1], 0])
        #     P_world = R @ p_local + np.array([0, 0, z])
        #     leg.pos = vector(*bases[i])
        #     leg.axis = vector(*(P_world - np.array(bases[i])))

        # --- update ball position ---
        # scale its z position so it stays above the platform visually
        ball.pos = vector(ball_pos[0], ball_pos[1], ball_pos[2])

    sleep(0.5)
    print("Simulation complete.")


if __name__ == "__main__":
    run_visual_simulation()
    sleep(10000)