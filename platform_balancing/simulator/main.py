import time
import serial
import serial.tools.list_ports
import numpy as np
import math
from pid import PIDController
from platform_controller import PlatformController
from simulator import simulation_main
from ball_tracker import BallTracker
from logger import Logger

K_P = 2.8*0.5
K_I = 0.5*1
K_D = 1.7

def select_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial devices found.")
        return None
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} — {p.description}")
    try:
        return ports[int(input("Select port number: "))].device
    except:
        return None


def hardware_main():
    port = select_serial_port()
    if not port:
        return

    tracker = BallTracker(
        calib_file="/Users/brendanchharawala/Documents/GitHub/ball-balancer/platform_balancing/simulator/camera_calibration.yaml",
        tag_csv="/Users/brendanchharawala/Documents/GitHub/ball-balancer/platform_balancing/simulator/tags_positions.csv",
        camera_index=0,
        show_debug=True
    )

    plane = PlatformController(port)
    pid = PIDController(kp=K_P, ki=K_I, kd=K_D)
    tracker = BallTracker(camera_index=1)
    config = {
        "kp": K_P,
        "ki": K_I,
        "kd": K_D,
        "max_tilt_angle_deg": 15,
        "platform_height_m": 0.05
    }
    logger = Logger(config)
    setpoint = np.array([0.0, 0.0])
    prev_pos = tracker.get_x_y(display=False)
    prev_time = time.time()

    try:
        while True:
            current_pos = tracker.get_x_y(display=True)
            print(current_pos)
            if current_pos is None:
                pid.reset_integral()
                plane.send_angles(0, 0)
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now

            velocity = np.zeros(3)
            if (prev_pos is not None):
                velocity = (current_pos - prev_pos) / dt
            
            prev_pos = current_pos

            error = setpoint - current_pos

            roll, pitch = pid.compute_angles(error, velocity, dt)
            print(f"Error: {error}, Pos: {current_pos}, Roll: {roll}, Pitch: {pitch}")
            plane.send_angles(math.degrees(roll), math.degrees(pitch))
            logger.log(now, dt, error)

    except KeyboardInterrupt:
        pass
    finally:
        tracker.release()
        plane.close()
        logger.close()


HARDWARE = True
if __name__ == "__main__":
    if HARDWARE:
        hardware_main()
    else:
        simulation_main()
