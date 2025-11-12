# import time
# import numpy as np
# from typing import Tuple, List
# from numpy.typing import NDArray

# from settings import Settings
# from kinematics import rotation_matrix, leg_points_rigid
# from visualizer import Visualizer
# from pid import PIDController
# from tracking import BallTracker


# # ---------------- BALL CLASS ----------------
# class Ball:
#     def __init__(self, pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
#         self.pos: NDArray[np.float64] = np.array(pos, dtype=float)
#         self.vel: NDArray[np.float64] = np.zeros(3)
#         self.radius: float = Settings.BALL_RADIUS

#     def update(self, dt: float, plane_pose: Tuple[float, float, float]) -> None:
#         """Physics update (not used if using live tracker)."""
#         roll, pitch, z = plane_pose
#         R = rotation_matrix(roll, pitch)
#         n = R @ np.array([0.0, 0.0, 1.0])
#         n = n / np.linalg.norm(n)
#         plane_point = np.array([0.0, 0.0, z])
#         plane_offset = np.dot(n, plane_point)
#         g_vec = np.array([0.0, 0.0, -Settings.G])
#         a_parallel = g_vec - np.dot(g_vec, n) * n
#         a_mag = np.linalg.norm(a_parallel)
#         if a_mag < 1e-8:
#             self.vel *= 0.0
#             return
#         direction = a_parallel / a_mag
#         a = a_mag / (1.0 + Settings.I_SPHERE / (self.radius**2))
#         self.vel += a * direction * dt
#         self.vel -= np.dot(self.vel, n) * n
#         self.pos += self.vel * dt
#         signed_dist = np.dot(self.pos, n) - plane_offset
#         self.pos = self.pos - signed_dist * n + n * self.radius


# # ---------------- TWO-BAR LINK CLASS ----------------
# class TwoBarLink:
#     def __init__(self, base_point: Tuple[float, float, float], contact_point_local: Tuple[float, float]):
#         self.base: NDArray[np.float64] = np.array(base_point, dtype=float)
#         self.contact_local: NDArray[np.float64] = np.array([contact_point_local[0], contact_point_local[1], 0.0])
#         self.l1: float = Settings.MOTOR_LINK_LEN
#         self.l2: float = Settings.PUSH_LINK_LEN


# # ---------------- STEWART PLATFORM SIMULATOR ----------------
# class StewartPlatformSimulator:
#     def __init__(
#         self,
#         dt: float = Settings.DT,
#         bases: List[Tuple[float, float, float]] = Settings.BASES,
#         contacts_local: List[Tuple[float, float]] = Settings.CONTACTS,
#         ball_pos: Tuple[float, float] = (0.05, 0.05),
#         tracker=None
#     ):
#         self.dt: float = dt
#         self.plane_pose: NDArray[np.float64] = (0, 0, Settings.TABLE_HEIGHT)
#         self.links: List[TwoBarLink] = [TwoBarLink(bases[i], contacts_local[i]) for i in range(3)]
#         self.ball: Ball = Ball(pos=(ball_pos[0], ball_pos[1], self.plane_pose[2] + 0.02))
#         self.sim_time: float = 0.0
#         self.tracker = tracker

#     def step(self, target_pose: Tuple[float, float, float]) -> None:
#         self.plane_pose = np.array(target_pose, dtype=float)

#         if self.tracker is not None:
#             cam_pos = self.tracker.get_position()  # np.array([x_m, y_m])
#             if cam_pos is not None:
#                 self.ball.pos[0] = cam_pos[0]           # x
#                 self.ball.pos[1] = cam_pos[1]           # y
#                 self.ball.pos[2] = self.plane_pose[2] + 0.02  # small z offset
#                 self.ball.vel[:] = 0.0
#         else:
#             self.ball.update(self.dt, self.plane_pose)

#         self.sim_time += self.dt


# # ---------------- MAIN SIMULATION ----------------
# def simulation_main():
#     # --- Initialize camera tracker ---
#     tracker = BallTracker(
#         calib_file="/Users/brendanchharawala/Documents/GitHub/ball-balancer/platform_balancing/simulator/camera_calibration.yaml",
#         tag_csv="/Users/brendanchharawala/Documents/GitHub/ball-balancer/platform_balancing/simulator/tags_positions.csv",
#         camera_index=1,
#         show_debug=True  # set False to hide OpenCV windows
#     )

#     sim = StewartPlatformSimulator(tracker=tracker)
#     pid = PIDController(kp=5, ki=0, kd=5)
#     vis = Visualizer(sim=sim)
#     angles = (0.0, 0.0)

#     try:
#         while True:
#             t_start = time.time()
#             sim.step(target_pose=(angles[0], angles[1], Settings.TABLE_HEIGHT))
#             vis.update()
#             angles = pid.compute_angles(error=sim.ball.pos, velocity=sim.ball.vel, dt=sim.dt)
#             elapsed = time.time() - t_start
#             sleep_time = sim.dt - elapsed
#             if sleep_time > 0:
#                 time.sleep(sleep_time)
#     finally:
#         tracker.release()


# # ---------------- ENTRY POINT ----------------
# if __name__ == "__main__":
#     simulation_main()

import time
import numpy as np
from typing import Tuple, List
from numpy.typing import NDArray

from settings import Settings
from kinematics import rotation_matrix, leg_points_rigid
from visualizer import Visualizer
from pid import PIDController
from tracking import BallTracker


# ---------------- BALL CLASS (unchanged) ----------------
class Ball:
    def __init__(self, pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.pos: NDArray[np.float64] = np.array(pos, dtype=float)
        self.vel: NDArray[np.float64] = np.zeros(3)
        self.radius: float = Settings.BALL_RADIUS

    def update(self, dt: float, plane_pose: Tuple[float, float, float]) -> None:
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


# ---------------- TWO-BAR LINK CLASS (unchanged) ----------------
class TwoBarLink:
    def __init__(self, base_point: Tuple[float, float, float], contact_point_local: Tuple[float, float]):
        self.base: NDArray[np.float64] = np.array(base_point, dtype=float)
        self.contact_local: NDArray[np.float64] = np.array([contact_point_local[0], contact_point_local[1], 0.0])
        self.l1: float = Settings.MOTOR_LINK_LEN
        self.l2: float = Settings.PUSH_LINK_LEN


# ---------------- STEWART PLATFORM SIMULATOR (patched) ----------------
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

        # new fields for robust camera handling
        self._last_cam_pos: NDArray[np.float64] | None = None   # 2-element array [x, y]
        self._last_cam_time: float | None = None
        self._hold_position_on_drop = True  # sample-hold behaviour when camera drops

    def step(self, target_pose: Tuple[float, float, float]) -> None:
        self.plane_pose = np.array(target_pose, dtype=float)

        if self.tracker is not None:
            cam_pos = self.tracker.get_position()  # should return np.array([x_m, y_m]) or None
            now = self.sim_time

            if cam_pos is not None:
                cam_pos = np.array(cam_pos, dtype=float)

                # compute camera velocity from last camera sample if available
                if self._last_cam_pos is not None and self._last_cam_time is not None:
                    dt_cam = now - self._last_cam_time
                    if dt_cam > 1e-9:
                        cam_vel = (cam_pos - self._last_cam_pos) / dt_cam
                    else:
                        cam_vel = np.zeros(2, dtype=float)
                else:
                    cam_vel = np.zeros(2, dtype=float)

                # update ball pos/vel from camera (z set to sit on plane)
                self.ball.pos[0] = cam_pos[0]
                self.ball.pos[1] = cam_pos[1]
                self.ball.pos[2] = self.plane_pose[2] + max(self.ball.radius, 0.0)

                self.ball.vel[0] = cam_vel[0]
                self.ball.vel[1] = cam_vel[1]
                self.ball.vel[2] = 0.0

                # store last camera sample
                self._last_cam_pos = cam_pos.copy()
                self._last_cam_time = now

            else:
                # camera returned None (dropout)
                if self._last_cam_pos is not None and self._hold_position_on_drop:
                    # hold last known position and do not overwrite velocity if you want continuity
                    self.ball.pos[0] = self._last_cam_pos[0]
                    self.ball.pos[1] = self._last_cam_pos[1]
                    self.ball.pos[2] = self.plane_pose[2] + max(self.ball.radius, 0.0)
                    # optionally keep previous velocity (no overwrite)
                    # self.ball.vel *= 0.98  # small damping if desired
                else:
                    # fallback to physics simulation when no camera history exists
                    self.ball.update(self.dt, self.plane_pose)
        else:
            # no tracker: pure physics simulation
            self.ball.update(self.dt, self.plane_pose)

        self.sim_time += self.dt


# ---------------- Robust hardware_main pattern ----------------
def hardware_main(controller, serial_reader, dt=Settings.DT):
    """
    controller: your PlatformController or similar (for sending commands).
    serial_reader: function that returns np.array([x,y]) or None when no valid sample available.
    """
    prev_pos = None
    prev_time = None

    while True:
        t0 = time.time()
        # read position from serial/camera. must return numpy array or None
        current_pos = serial_reader(timeout=0.01)  # implement serial_reader to do parsing + timeout

        if current_pos is None:
            # no valid sample this loop. do not perform invalid arithmetic.
            # Option A: skip velocity update and continue control using last known sample
            # Option B: continue reading until valid. Here we choose sample-hold.
            print("warning: no valid position sample; holding last known position")
            # you might still want to run controller with last known ball.pos
        else:
            # got a valid sample; compute velocity safely
            now = time.time()
            if prev_pos is None or prev_time is None:
                velocity = np.zeros_like(current_pos)
            else:
                dt_meas = now - prev_time
                if dt_meas <= 1e-9:
                    velocity = np.zeros_like(current_pos)
                else:
                    velocity = (current_pos - prev_pos) / dt_meas

            # convert 2D current_pos -> 3D ball pos expected by code
            ball_pos_3d = np.array([current_pos[0], current_pos[1], Settings.TABLE_HEIGHT + Settings.BALL_RADIUS], dtype=float)
            ball_vel_3d = np.array([velocity[0], velocity[1], 0.0], dtype=float)

            # use these with your simulator/controller
            # e.g. sim.ball.pos = ball_pos_3d; sim.ball.vel = ball_vel_3d
            prev_pos = current_pos.copy()
            prev_time = now

        # control step & actuation (example)
        # angles = pid.compute_angles(error=sim.ball.pos, velocity=sim.ball.vel, dt=dt)
        # controller.send_angles(angles[0], angles[1])

        # timing
        elapsed = time.time() - t0
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# ---------------- MAIN SIMULATION ENTRY (example) ----------------
def simulation_main():
    tracker = BallTracker(
        calib_file="/Users/brendanchharawala/.../camera_calibration.yaml",
        tag_csv="/Users/brendanchharawala/.../tags_positions.csv",
        camera_index=1,
        show_debug=True
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


if __name__ == "__main__":
    simulation_main()