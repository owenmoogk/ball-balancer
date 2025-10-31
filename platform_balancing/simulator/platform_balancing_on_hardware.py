import re
import time
import serial
import serial.tools.list_ports
import numpy as np
from settings import Settings

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

# ------------------------------------------------------------------
#  GEOMETRY + INVERSE KINEMATICS
# ------------------------------------------------------------------
def rotation_matrix(roll, pitch):
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return R_y @ R_x

def bearing_point_exact(base, p_world, l1, l2):
    B = np.array(base, float)
    P = np.array(p_world, float)
    d = np.linalg.norm(P - B)
    if d > l1 + l2 or d < abs(l1 - l2): return False, None
    v = (P - B) / d
    a = (l1**2 - l2**2 + d**2) / (2*d)
    h2 = l1**2 - a**2
    if h2 < 0: return False, None
    h = np.sqrt(h2)
    M = B + a * v
    radial_dir = B.copy(); radial_dir[2] = 0
    if np.linalg.norm(radial_dir) < 1e-9:
        radial_dir = np.array([1, 0, 0])
    else:
        radial_dir /= np.linalg.norm(radial_dir)
    plane_normal = np.cross(radial_dir, np.array([0, 0, 1]))
    plane_normal /= np.linalg.norm(plane_normal)
    n_perp = np.cross(v, plane_normal); n_perp /= np.linalg.norm(n_perp)
    bearing1 = M + h * n_perp
    bearing2 = M - h * n_perp
    gravity = np.array([0, 0, -1])
    bearing = bearing1 if np.dot(bearing1, gravity) < np.dot(bearing2, gravity) else bearing2
    return True, bearing

def motor_angle_deg(base, bearing):
    radial = base.copy(); radial[2] = 0
    if np.linalg.norm(radial) < 1e-9: radial = np.array([1,0,0])
    e1 = radial / np.linalg.norm(radial)
    n = np.cross(e1, np.array([0,0,1])); n /= np.linalg.norm(n)
    e2 = np.cross(n, e1)
    L = bearing - base
    Lp = L - np.dot(L, n)*n
    x, y = np.dot(Lp, e1), np.dot(Lp, e2)
    return float(np.rad2deg(np.arctan2(y, x)))

def leg_points_rigid(base, contact_local, plane_pose, l1, l2):
    roll, pitch, z = plane_pose
    R = rotation_matrix(roll, pitch)
    P_world = R @ np.array([contact_local[0], contact_local[1], 0.0]) + np.array([0, 0, z])
    ok, bearing = bearing_point_exact(base, P_world, l1, l2)
    if not ok: return None
    return base, bearing

def solve_motor_angles_for_plane(roll_deg, pitch_deg, z=Settings.TABLE_HEIGHT):
    roll = np.deg2rad(roll_deg); pitch = np.deg2rad(pitch_deg)
    out = np.full(3, np.nan)
    for i, (b, c) in enumerate(zip(Settings.BASES, Settings.CONTACTS)):
        segs = leg_points_rigid(np.array(b), np.array([c[0], c[1], 0.0]), (roll, pitch, z), Settings.MOTOR_LINK_LEN, Settings.PUSH_LINK_LEN)
        if segs is None: continue
        b1, br = segs
        out[i] = motor_angle_deg(b1, br)
    return out

# ------------------------------------------------------------------
#  MAIN CONTROL LOOP
# ------------------------------------------------------------------
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