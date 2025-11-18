import os
import pandas as pd
import matplotlib.pyplot as plt

LOGS_DIR = "logs"

def select_option(options, title):
    """Display a list and let user select an index."""
    print(f"\n=== {title} ===")
    for i, opt in enumerate(options):
        print(f"[{i}] {opt}")
    idx = int(input("Select number: "))
    return options[idx]


def load_log_csv(run_path):
    """Load log.csv from a run directory."""
    csv_path = os.path.join(run_path, "log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No log.csv found in {run_path}")
    df = pd.read_csv(csv_path)
    return df


def plot_error(df, run_path):
    """Plot error_x and error_y vs time and save in the same folder."""
    plt.figure(figsize=(10, 6))

    plt.plot(df["time"], df["error_x"], label="error_x", linewidth=1.2)
    plt.plot(df["time"], df["error_y"], label="error_y", linewidth=1.2)

    plt.title("Error vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(run_path, "error_plot.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"\nSaved plot to: {out_path}")


def main():
    pid_folders = [
        f for f in os.listdir(LOGS_DIR)
        if os.path.isdir(os.path.join(LOGS_DIR, f))
    ]
    pid_choice = select_option(pid_folders, "Select PID setting")

    pid_path = os.path.join(LOGS_DIR, pid_choice)
    run_folders = [
        f for f in os.listdir(pid_path)
        if os.path.isdir(os.path.join(pid_path, f))
    ]
    run_choice = select_option(run_folders, "Select run timestamp")

    run_path = os.path.join(pid_path, run_choice)

    df = load_log_csv(run_path)
    plot_error(df, run_path)


if __name__ == "__main__":
    main()
