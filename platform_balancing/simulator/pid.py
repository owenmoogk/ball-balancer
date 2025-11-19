import numpy as np
from numpy.typing import NDArray
from settings import Settings
import time
import math

MAX_TILT_DEG = 15


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(2)
        self.max_tilt_rad = math.radians(MAX_TILT_DEG)
        self._last_time = time.time()
        self.p = None
        self.i = None
        self.d = None

    def reset_integral(self):
        self.integral = np.zeros(2)

    def compute_angles(
        self, error: NDArray[np.float64], velocity: NDArray[np.float64], dt: float
    ) -> np.ndarray:
        error = error[0:2]
        velocity = velocity[0:2]

        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        # PID calculations
        self.p = self.kp * error
        self.integral += error * dt
        self.i = self.ki * self.integral
        self.d = self.kd * -velocity

        a_des = self.p + self.i + self.d

        # Convert to tilt
        tilt = a_des / Settings.G
        pitch = tilt[0]
        roll = tilt[1] 

        pitch = np.clip(pitch, -self.max_tilt_rad, self.max_tilt_rad)
        roll = np.clip(roll, -self.max_tilt_rad, self.max_tilt_rad)
        print(self.max_tilt_rad)
        if abs(pitch) == self.max_tilt_rad or abs(roll) == self.max_tilt_rad:
            print("CLIPPED")


        return np.array([roll, pitch])
    
    def get_pid_values(self):
        return self.p, self.i, self.d
