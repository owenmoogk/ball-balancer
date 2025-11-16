import time
import serial
import serial.tools.list_ports
import numpy as np
import math
from pid import BallController
from platform_controller import PlatformController
from simulator import simulation_main
from ball_tracker import BallTracker


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
    controller = BallController(kp=14.8, ki=0, kd=0)
    tracker = BallTracker(camera_index=1)

    prev_pos = tracker.get_x_y(display=False)
    prev_time = time.time()

    try:
        while True:
            current_pos = tracker.get_x_y(display=True)
            now = time.time()
            dt = now - prev_time
            prev_time = now


            velocity = (current_pos - prev_pos) / dt
            prev_pos = current_pos

            roll, pitch = controller.compute_angles(current_pos, velocity, dt)
            # print(f"Error: {error}, Pos: {current_pos}, Roll: {roll}, Pitch: {pitch}")
            plane.send_angles(math.degrees(roll), math.degrees(pitch))

            time.sleep(0.03)

    except KeyboardInterrupt:
        pass
    finally:
        tracker.release()
        plane.close()


HARDWARE = True
if __name__ == "__main__":
    if HARDWARE:
        hardware_main()
    else:
        simulation_main()
