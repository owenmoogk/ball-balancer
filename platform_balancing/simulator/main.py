import time
import serial
import serial.tools.list_ports
import numpy as np
from pid import PIDController
import math
from platform_controller import PlatformController
from simulator import simulation_main

# Dummy placeholder for ball position function
def get_x_y():
    t = time.time()  # current time in seconds
    x = 0.05 * np.sin(2 * np.pi * 0.5 * t)  
    y = 0.05 * np.cos(2 * np.pi * 0.5 * t)  
    return np.array([x, y])


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

    plane = PlatformController(port)
    pid = PIDController(kp=5, ki=0, kd=5)
    setpoint = np.array([0.0, 0.0])
    prev_pos = get_x_y()  # initial position
    prev_time = time.time()

    try:
        while True:
            # Get current ball position and timestamp
            current_pos = get_x_y()
            now = time.time()
            dt = now - prev_time
            prev_time = now

            # Compute velocity (simple finite difference)
            velocity = (current_pos - prev_pos) / dt
            prev_pos = current_pos

            # Compute error
            error = setpoint - current_pos

            print(error, current_pos)

            # Compute desired roll/pitch angles from PID
            roll, pitch = pid.compute_angles(error, velocity, dt)
            print(roll, pitch)
            # Send commands to plane
            plane.send_angles(math.degrees(roll), math.degrees(pitch))

            # Control loop rate (~30–50 Hz)
            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        plane.close()


HARDWARE = True
if __name__ == "__main__":
    if HARDWARE:
        hardware_main()
    else:
        simulation_main()