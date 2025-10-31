import numpy as np
from numpy.typing import NDArray
from simulator import StewartPlatformSimulator
from visualizer import Visualizer
from simulator import TABLE_HEIGHT, G
import time


MAX_TILT = 0.5

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, dt: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = np.zeros(2)
        self.prev_error = np.zeros(2)
        self.max_tilt = MAX_TILT
    
    def compute_angles(
        self, error: NDArray[np.float64], velocity: NDArray[np.float64]
    ) -> np.ndarray:
        
        error = error[0:2]
        velocity = velocity[0:2]


        p = self.kp * error

        self.integral += error * self.dt
        i = self.ki * self.integral

        d = self.kd * (velocity)

        a_des = p + i + d  # desired acceleration in x,y (m/s^2) (index 0 -> x, 1 -> y)

        tilt = a_des / (G)

        pitch = tilt[0]  # positive pitch accelerates +x
        roll = tilt[1]  # positive roll accelerates +y

        # Apply saturation/clamping
        pitch = np.clip(pitch, -self.max_tilt, self.max_tilt)
        roll = np.clip(roll, -self.max_tilt, self.max_tilt)

        pitch = -pitch


        return np.array([roll, pitch])


if __name__ == "__main__":
    sim = StewartPlatformSimulator()
    pid = PIDController(kp=5, ki=0, kd=5, dt=sim.dt)
    vis = Visualizer(sim=sim)
    angles = (0,0)
    for step in range(10000):
        t_start = time.time()
        sim.step(target_pose=(angles[0], angles[1], TABLE_HEIGHT))
        if (step % 10 == 0 ):
            vis.update()
        angles = pid.compute_angles(error=sim.ball.pos, velocity=sim.ball.vel)
        elapsed = time.time() - t_start
        sleep_time = sim.dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        print(step*sim.dt)