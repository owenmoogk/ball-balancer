import csv
import os
from datetime import datetime

class Logger:
    def __init__(self, kp, ki, kd, filepath="logs"):
        os.makedirs(filepath, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(
            filepath,
            f"log_{timestamp}_kp{kp}_ki{ki}_kd{kd}.csv"
        )

        self.file = open(self.filename, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "time",
            "dt",
            "error_x",
            "error_y"
        ])

    def log(self, now, dt, error):
        self.writer.writerow([
            now,
            dt,
            float(error[0]),
            float(error[1])
        ])

    def close(self):
        self.file.close()
