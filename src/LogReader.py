import pandas as pd
from pathlib import Path

LOG_FILE = "cpu_profile_gpu_2.log"


def load_log(logfile):
    records = []

    with open(logfile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")

            timestamp = parts[0]
            label = parts[1]

            metrics = {}
            for item in parts[2:]:
                key, value = item.split("=")
                metrics[key] = float(value)

            records.append({
                "timestamp": timestamp,
                "label": label,
                **metrics
            })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def summarize_by_label(df):
    summary = (
        df.groupby("label")
        .agg(
            calls=("wall", "count"),
            wall_mean=("wall", "mean"),
            wall_min=("wall", "min"),
            wall_max=("wall", "max"),
            cpu_mean=("cpu", "mean"),
            cpu_max=("cpu", "max"),
            user_mean=("user", "mean"),
            sys_mean=("sys", "mean"),
            cv_threads=("cv_threads", "max"),
        )
        .sort_values("wall_mean", ascending=False)
    )

    return summary


def print_summary(summary):
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== CPU PROFILING SUMMARY (by label) ===\n")
    print(summary)
    print("\nLegend:")
    print("  wall_*  : wall-clock seconds")
    print("  cpu_*   : effective CPU utilization (%)")
    print("  calls   : number of profiled invocations")
    print("  cv_threads : OpenCV internal thread count\n")


def detect_thread_changes(df):
    changed = df[df["threads_start"] != df["threads_end"]]

    if not changed.empty:
        print("\n⚠ Thread count changed during profiling blocks:\n")
        print(changed[[
            "timestamp", "label",
            "threads_start", "threads_end"
        ]])
    else:
        print("\n✓ No thread count changes detected\n")


def main():
    logfile = Path(LOG_FILE)

    if not logfile.exists():
        raise FileNotFoundError(f"Log file not found: {logfile}")

    df = load_log(logfile)

    summary = summarize_by_label(df)
    print_summary(summary)

    detect_thread_changes(df)

    # Optional CSV export for plotting
    summary.to_csv("cpu_profile_summary.csv")
    print("Summary written to cpu_profile_summary.csv")


if __name__ == "__main__":
    main()