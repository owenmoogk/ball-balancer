import time
import serial
import numpy as np
from kinematics import solve_motor_angles_for_plane

class PlatformController:
    def __init__(self, port: str, baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        time.sleep(2)  # allow connection to settle
        self.is_first = True
        self.servo_cmds = [90, 90, 90]


    def send_angles(self, roll_deg, pitch_deg):
        th = solve_motor_angles_for_plane(roll_deg, pitch_deg)
        if np.any(np.isnan(th)):
            print("Unreachable configuration.")
            return False
        if not self.is_first:
            self.servo_cmds = 86 - th

        self.is_first = False

        # print(f"sending {self.servo_cmds}")

        msg = f"{self.servo_cmds[0]:.1f} {self.servo_cmds[1]:.1f} {self.servo_cmds[2]:.1f}\n"
        n = self.ser.write(msg.encode())
        # print(f"-> {msg.strip()}  ({n} bytes)")
        self.read_all()
        return True

    def read_all(self):
        time.sleep(0.02)
        while self.ser.in_waiting:
            ln = self.ser.readline().decode(errors="ignore").rstrip()
            # if ln:
            #     print(f"<- {ln}")

    def close(self):
        self.ser.close()
