import serial
import serial.tools.list_ports
import time

def select_serial_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial devices found.")
        return None
    print("Available serial ports:")
    for i, p in enumerate(ports):
        print(f"[{i}] {p.device} — {p.description}")
    idx = input("Select port number: ")
    try:
        return ports[int(idx)].device
    except (ValueError, IndexError):
        return None

def main():
    port = select_serial_port()
    if not port:
        return
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)
    print(f"Connected to {port}")

    while True:
        try:
            angle = float(input("Enter servo angle (0–180): "))
            if 0 <= angle <= 180:
                ser.write(f"{angle}\n".encode())
            else:
                print("Out of range.")
        except ValueError:
            print("Invalid input.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()


