import re, time
import serial, serial.tools.list_ports

def select_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial devices found."); return None
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} — {p.description}")
    try: return ports[int(input("Select port number: "))].device
    except: return None

def parse_three(line):
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line.replace(',', ' '))
    if len(nums) < 3: raise ValueError
    a, b, c = map(float, nums[:3])
    if not all(0 <= x <= 180 for x in (a, b, c)): raise ValueError
    return a, b, c

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
        print(f"Connected to {port}. Enter three angles 0–180 as: a b c")
        while True:
            try:
                line = input("> ").strip()
                if line.lower() in {"q","quit","exit"}: break
                a,b,c = parse_three(line)
                msg = f"{a:.1f} {b:.1f} {c:.1f}\n"
                n = ser.write(msg.encode())
                print(f"-> {msg.strip()}  ({n} bytes)")
                read_all(ser)
            except ValueError:
                print("Enter three numbers in [0,180].")
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()
