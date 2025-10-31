import re
import time
import serial
import serial.tools.list_ports
import numpy as np
from kinematics import solve_motor_angles_for_plane

def select_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial devices found."); return None
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} — {p.description}")
    try: return ports[int(input("Select port number: "))].device
    except: return None

def read_all(ser):
    time.sleep(0.02)
    while ser.in_waiting:
        ln = ser.readline().decode(errors="ignore").rstrip()
        if ln: print(f"<- {ln}")

def main():
    port = select_serial_port()
    if not port: return
    with serial.Serial(port, 115200, timeout=0.1) as ser:
        time.sleep(2)
        print(f"Connected to {port}. Enter roll and pitch (deg). Type q to quit.")
        while True:
            try:
                line = input("> ").strip()
                if line.lower() in {"q", "quit", "exit"}: break
                roll_deg, pitch_deg = map(float, re.findall(r'[-+]?\d*\.?\d+', line))
                th = solve_motor_angles_for_plane(roll_deg, pitch_deg)
                if np.any(np.isnan(th)):
                    print("Unreachable configuration.")
                    continue
                servo_cmds = 86*2 - th
                msg = f"{servo_cmds[0]:.1f} {servo_cmds[1]:.1f} {servo_cmds[2]:.1f}\n"
                n = ser.write(msg.encode())
                print(f"-> {msg.strip()}  ({n} bytes)")
                read_all(ser)
            except ValueError:
                print("Enter roll and pitch in degrees (e.g. '5 -3').")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()