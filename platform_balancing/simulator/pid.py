import numpy as np
from numpy.typing import NDArray
from settings import Settings
import time
import math

MAX_TILT_DEG = 15


class BallController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(2)
        self.max_tilt_rad = math.radians(MAX_TILT_DEG)
        self._last_time = time.time()

        self.a_max= 0.5
        self.v_max = 0.25

        self.last_compute_time = 0

    def trapezoid_compute(self, start: NDArray[np.float64], end: NDArray[np.float64]):
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        delta = end - start
        dist = float(np.linalg.norm(delta))
        if dist == 0:
            # Return zero-motion function
            return lambda t: start.copy()

        direction = delta / dist

        a = float(self.a_max)
        v = float(self.v_max)

        t_acc = v / a
        d_acc = 0.5 * a * t_acc**2

        # Check if we can reach max velocity
        if 2 * d_acc >= dist:
            # Not enough distance: triangular profile
            # Solve v_peak from dist = 2 * (1/2 * a * t^2)
            # => dist = a * t^2  => t = sqrt(dist / a)
            t_peak = np.sqrt(dist / a)

            def pos_fn(t):
                if t <= 0:
                    return start.copy()

                if t < t_peak:
                    d = 0.5 * a * t**2
                    return start + d * direction

                # Deceleration
                td = t - t_peak
                if t < 2 * t_peak:
                    d = dist - 0.5 * a * td**2
                    return start + d * direction
                return end.copy()

            return pos_fn

        else:
            # Full trapezoid: accelerate → cruise → decelerate
            d_cruise = dist - 2 * d_acc
            t_cruise = d_cruise / v
            t_total = 2 * t_acc + t_cruise

            def pos_fn(t):
                if t <= 0:
                    return start.copy()

                # Accelerating
                if t < t_acc:
                    d = 0.5 * a * t**2
                    return start + d * direction

                # Cruising
                if t < t_acc + t_cruise:
                    tc = t - t_acc
                    d = d_acc + v * tc
                    return start + d * direction

                # Decelerating
                if t < t_total:
                    td = t - (t_acc + t_cruise)
                    d = d_acc + d_cruise + (v * td - 0.5 * a * td**2)
                    return start + d * direction

                return end.copy()

            return pos_fn


    def compute_angles(
        self, position: NDArray[np.float64], velocity: NDArray[np.float64], dt: float
    ) -> np.ndarray:
        
        if (time.time() > self.last_compute_time + 0.2):
            self.pos_fn = self.trapezoid_compute(position, np.zeros(2))
            self.t_ref = time.time()
            self.last_compute_time = time.time()

        target_pos = self.pos_fn(time.time() - self.t_ref)
        print(target_pos)
        error = target_pos - position[0:2]
        # velocity = velocity[0:2]

        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        # PID calculations
        p = self.kp * error
        self.integral += error * dt
        i = self.ki * self.integral
        # d = self.kd * -velocity

        # a_des = p + i + d
        a_des = p + i

        # Convert to tilt
        tilt = a_des / Settings.G
        pitch = tilt[0]
        roll = tilt[1] 

        pitch = np.clip(pitch, -self.max_tilt_rad, self.max_tilt_rad)
        roll = np.clip(roll, -self.max_tilt_rad, self.max_tilt_rad)
        if abs(pitch) == self.max_tilt_rad or abs(roll) == self.max_tilt_rad:
            print("CLIPPED")


        return np.array([roll, pitch])
