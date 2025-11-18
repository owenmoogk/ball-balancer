import csv
import os
import json
from datetime import datetime

class Logger:
    def __init__(self, config, base_dir="logs"):
        """
        config is a dict with at least:
        {
            "kp": ...,
            "ki": ...,
            "kd": ...,
            "max_tilt_angle_deg": ...,
            "platform_height_m": ...
        }
        """
        self.config = config
        pid_folder = f"kp{config['kp']}_ki{config['ki']}_kd{config['kd']}"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.config["timestamp"] = timestamp
        self.run_dir = os.path.join(base_dir, pid_folder, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

        self.csv_path = os.path.join(self.run_dir, "log.csv")

        self.metadata_path = os.path.join(self.run_dir, "metadata.json")

        with open(self.metadata_path, "w") as meta_file:
            json.dump(self.config, meta_file, indent=4)

        self.file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["time", "dt", "error", "p", "i", "d"])

    def log(self, now, dt, error, p, i ,d):
        self.writer.writerow([
            now,
            dt,
            error,
            p,
            i,
            d
        ])

    def close(self):
        self.file.close()
