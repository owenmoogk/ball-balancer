import time
import numpy as np
from typing import Tuple, List
from numpy.typing import NDArray

from settings import Settings
from kinematics import rotation_matrix, leg_points_rigid
from visualizer import Visualizer
from pid import PIDController
from tracking import BallTracker


# ---------------- BALL CLASS ----------------
class Ball:
    def __init__(self, pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.pos: NDArray[np.float64] = np.array(pos, dtype=float)
        self.vel: NDArray[np.float64] = np.zeros(3)
        self.radius: float = Settings.BALL_RADIUS

    def update(self, dt: float, plane_pose: Tuple[float, float, float]) -> None:
        """Physics update (not used if using live tracker)."""
        roll, pitch, z = plane_pose
        R = rotation_matrix(roll, pitch)
        n = R @ np.array([0.0, 0.0, 1.0])
        n = n / np.linalg.norm(n)
        plane_point = np.array([0.0, 0.0, z])
        plane_offset = np.dot(n, plane_point)
        g_vec = np.array([0.0, 0.0, -Settings.G])
        a_parallel = g_vec - np.dot(g_vec, n) * n
        a_mag = np.linalg.norm(a_parallel)
        if a_mag < 1e-8:
            self.vel *= 0.0
            return
        direction = a_parallel / a_mag
        a = a_mag / (1.0 + Settings.I_SPHERE / (self.radius**2))
        self.vel += a * direction * dt
        self.vel -= np.dot(self.vel, n) * n
        self.pos += self.vel * dt
        signed_dist = np.dot(self.pos, n) - plane_offset
        self.pos = self.pos - signed_dist * n + n * self.radius


# ---------------- TWO-BAR LINK CLASS ----------------
class TwoBarLink:
    def __init__(self, base_point: Tuple[float, float, float], contact_point_local: Tuple[float, float]):
        self.base: NDArray[np.float64] = np.array(base_point, dtype=float)
        self.contact_local: NDArray[np.float64] = np.array([contact_point_local[0], contact_point_local[1], 0.0])
        self.l1: float = Settings.MOTOR_LINK_LEN
        self.l2: float = Settings.PUSH_LINK_LEN


# ---------------- STEWART PLATFORM SIMULATOR ----------------
class StewartPlatformSimulator:
    def __init__(
        self,
        dt: float = Settings.DT,
        bases: List[Tuple[float, float, float]] = Settings.BASES,
        contacts_local: List[Tuple[float, float]] = Settings.CONTACTS,
        ball_pos: Tuple[float, float] = (0.05, 0.05),
        tracker=None
    ):
        self.dt: float = dt
        self.plane_pose: NDArray[np.float64] = (0, 0, Settings.TABLE_HEIGHT)
        self.links: List[TwoBarLink] = [TwoBarLink(bases[i], contacts_local[i]) for i in range(3)]
        self.ball: Ball = Ball(pos=(ball_pos[0], ball_pos[1], self.plane_pose[2] + 0.02))
        self.sim_time: float = 0.0
        self.tracker = tracker

    def step(self, target_pose: Tuple[float, float, float]) -> None:
        self.plane_pose = np.array(target_pose, dtype=float)

        if self.tracker is not None:
            cam_pos = self.tracker.get_position()  # np.array([x_m, y_m])
            if cam_pos is not None:
                self.ball.pos[0] = cam_pos[0]           # x
                self.ball.pos[1] = cam_pos[1]           # y
                self.ball.pos[2] = self.plane_pose[2] + 0.02  # small z offset
                self.ball.vel[:] = 0.0
        else:
            self.ball.update(self.dt, self.plane_pose)

        self.sim_time += self.dt


# ---------------- MAIN SIMULATION ----------------
def simulation_main():
    # --- Initialize camera tracker ---
    tracker = BallTracker(
        calib_file=r'c:\Users\asali\OneDrive\Desktop\School\3B\MTE 380\MTE-380-proj\platform_balancing\simulator\camera_calibration.yaml',
        tag_csv=r'c:\Users\asali\OneDrive\Desktop\School\3B\MTE 380\MTE-380-proj\platform_balancing\simulator\tags_positions.csv',
        camera_index=1,
        show_debug=True  # set False to hide OpenCV windows
    )

    sim = StewartPlatformSimulator(tracker=tracker)
    pid = PIDController(kp=5, ki=0, kd=5)
    vis = Visualizer(sim=sim)
    angles = (0.0, 0.0)

    try:
        while True:
            t_start = time.time()
            sim.step(target_pose=(angles[0], angles[1], Settings.TABLE_HEIGHT))
            vis.update()
            angles = pid.compute_angles(error=sim.ball.pos, velocity=sim.ball.vel, dt=sim.dt)
            elapsed = time.time() - t_start
            sleep_time = sim.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        tracker.release()


# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    simulation_main()
