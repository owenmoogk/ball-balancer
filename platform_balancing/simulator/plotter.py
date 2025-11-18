import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOGS_DIR = "logs"


def select_option(options, title):
    print(f"\n=== {title} ===")
    for i, opt in enumerate(options):
        print(f"[{i}] {opt}")
    idx = int(input("Select number: "))
    return options[idx]


def load_log_csv(run_path):
    csv_path = os.path.join(run_path, "log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No log.csv found in {run_path}")

    # Load raw CSV
    df = pd.read_csv(csv_path)

    # Expand vector-like columns "[x y]" into numeric columns
    vector_cols = ["error", "p", "i", "d"]

    for col in vector_cols:
        df[[f"{col}_x", f"{col}_y"]] = (
            df[col]
            .str.strip("[]")
            .str.split(expand=True)
            .astype(float)
        )

    return df


# ----------- NEW GENERALIZED VECTOR-PLOTTER -------------
def plot_vector_param(df, param, run_path):
    """
    param: 'error', 'p', 'i', or 'd'
    Generates a 3-subplot figure:
      1. x-component
      2. y-component
      3. magnitude
    """
    time = df["time"]
    x = df[f"{param}_x"]
    y = df[f"{param}_y"]
    mag = np.sqrt(x**2 + y**2)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"{param.upper()} vs Time", fontsize=14)

    axes[0].plot(time, x, label=f"{param}_x", color="tab:blue")
    axes[0].set_ylabel(f"{param}_x")
    axes[0].grid(True)

    axes[1].plot(time, y, label=f"{param}_y", color="tab:orange")
    axes[1].set_ylabel(f"{param}_y")
    axes[1].grid(True)

    axes[2].plot(time, mag, label=f"{param}_mag", color="tab:green")
    axes[2].set_ylabel(f"{param}_mag")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(run_path, f"{param}_plot.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved: {out_path}")


# ----------------- MAIN PLOT ERROR + PID ----------------
def plot_all(df, run_path):
    # Plot error, P, I, D
    for param in ["error", "p", "i", "d"]:
        plot_vector_param(df, param, run_path)


# ----------------------------- MAIN ------------------------------
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
    plot_all(df, run_path)


if __name__ == "__main__":
    main()
