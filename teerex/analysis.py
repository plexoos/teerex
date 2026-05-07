#!/usr/bin/env python3

"""
Analyze and compare multiple Simload CSV output files.

Example usage:

    python analyze_simload_csv.py results/*.csv --outdir plots

    python analyze_simload_csv.py run_cpu_heavy.csv run_gpu_heavy.csv run_balanced.csv \
        --outdir plots \
        --time-column time_s

Outputs:

    plots/combined.csv
    plots/summary_by_run.csv
    plots/timeseries_*.png
    plots/boxplot_*.png
    plots/throughput_over_time.png   if a task/event completion column is detected
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


TIME_CANDIDATES = [
    "time",
    "time_s",
    "timestamp",
    "elapsed",
    "elapsed_s",
    "t",
    "start_time",
    "submit_offset_s",
    "start_offset_s",
    "end_offset_s",
]

RUN_ID_CANDIDATES = [
    "run",
    "run_id",
    "scenario",
    "scenario_id",
]

EVENT_ID_CANDIDATES = [
    "event_id",
    "task_id",
    "job_id",
    "id",
]

METRIC_PATTERNS = [
    r"cpu.*util",
    r"gpu.*util",
    r"cpu.*usage",
    r"gpu.*usage",
    r"latency",
    r"runtime",
    r"duration",
    r"throughput",
    r"memory",
    r"mem",
    r"gpu.*mem",
    r"queue",
    r"wait",
    r"iterations",
    r"h2d",
    r"d2h",
]


def normalize_column_name(col: str) -> str:
    col = col.strip()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[^\w_]+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.strip("_").lower()


def infer_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    columns = list(columns)
    lowered = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def infer_metric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    metrics: list[str] = []

    for col in numeric_cols:
        col_l = col.lower()
        if any(re.search(pattern, col_l) for pattern in METRIC_PATTERNS):
            metrics.append(col)

    exclude = {"event_id", "task_id", "job_id", "id", "run_index"}
    metrics = [c for c in metrics if c.lower() not in exclude]

    return sorted(set(metrics))


def read_one_csv(path: Path, run_label: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    original_cols = list(df.columns)
    df.columns = [normalize_column_name(c) for c in df.columns]

    if len(set(df.columns)) != len(df.columns):
        raise ValueError(
            f"Column-name collision after normalization in {path}. "
            f"Original columns were: {original_cols}"
        )

    label = run_label if run_label is not None else path.stem

    existing_run_col = infer_column(df.columns, RUN_ID_CANDIDATES)
    if existing_run_col is None:
        df["run"] = label
    else:
        df["run"] = df[existing_run_col].astype(str)

    df["source_file"] = str(path)

    return df


def read_many_csv(paths: list[Path]) -> pd.DataFrame:
    frames = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

        df = read_one_csv(path)
        frames.append(df)

    if not frames:
        raise ValueError("No CSV files were loaded.")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    return combined


def save_summary(df: pd.DataFrame, metrics: list[str], outdir: Path) -> pd.DataFrame:
    if not metrics:
        summary = df.groupby("run", dropna=False).size().rename("rows").reset_index()
        summary.to_csv(outdir / "summary_by_run.csv", index=False)
        return summary

    summary = (
        df.groupby("run", dropna=False)[metrics]
        .agg(["count", "mean", "std", "min", "median", "max"])
    )

    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]

    summary = summary.reset_index()
    summary.to_csv(outdir / "summary_by_run.csv", index=False)

    return summary


def plot_timeseries(
    df: pd.DataFrame,
    time_col: str,
    metric: str,
    outdir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for run, g in df.groupby("run", dropna=False):
        g = g[[time_col, metric]].dropna().sort_values(time_col)
        if g.empty:
            continue

        ax.plot(g[time_col], g[metric], label=str(run), linewidth=1.5)

    ax.set_title(f"{metric} over time")
    ax.set_xlabel(time_col)
    ax.set_ylabel(metric)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / f"timeseries_{metric}.png", dpi=160)
    plt.close(fig)


def plot_boxplot(
    df: pd.DataFrame,
    metric: str,
    outdir: Path,
) -> None:
    data = []
    labels = []

    for run, g in df.groupby("run", dropna=False):
        values = g[metric].dropna()
        if values.empty:
            continue
        data.append(values)
        labels.append(str(run))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(f"{metric} distribution by run")
    ax.set_xlabel("run")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(outdir / f"boxplot_{metric}.png", dpi=160)
    plt.close(fig)


def plot_completion_throughput(
    df: pd.DataFrame,
    time_col: str,
    id_col: str,
    outdir: Path,
    bin_width_s: float,
) -> None:
    tmp = df[[time_col, id_col, "run"]].dropna().copy()

    if tmp.empty:
        return

    tmp["time_bin"] = (tmp[time_col] // bin_width_s) * bin_width_s

    throughput = (
        tmp.groupby(["run", "time_bin"], dropna=False)[id_col]
        .nunique()
        .reset_index(name="completed")
    )

    throughput["throughput_per_s"] = throughput["completed"] / bin_width_s

    fig, ax = plt.subplots(figsize=(10, 5))

    for run, g in throughput.groupby("run", dropna=False):
        g = g.sort_values("time_bin")
        ax.plot(g["time_bin"], g["throughput_per_s"], label=str(run), linewidth=1.5)

    ax.set_title("Completion throughput over time")
    ax.set_xlabel(time_col)
    ax.set_ylabel("completed tasks/events per second")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "throughput_over_time.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="CSV files or glob-expanded list of CSV files.",
    )
    parser.add_argument(
        "--outdir",
        default="simload_analysis",
        help="Output directory for combined CSV, summaries, and plots.",
    )
    parser.add_argument(
        "--time-column",
        default=None,
        help="Optional explicit time column name.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional explicit metric columns to plot.",
    )
    parser.add_argument(
        "--bin-width-s",
        type=float,
        default=1.0,
        help="Bin width for throughput plot.",
    )

    args = parser.parse_args()

    paths = [Path(p) for p in args.csv_files]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = read_many_csv(paths)

    df.to_csv(outdir / "combined.csv", index=False)

    if args.time_column is not None:
        time_col = normalize_column_name(args.time_column)
        if time_col not in df.columns:
            raise ValueError(
                f"Requested time column '{time_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
    else:
        time_col = infer_column(df.columns, TIME_CANDIDATES)

    if args.metrics is not None and len(args.metrics) > 0:
        metrics = [normalize_column_name(c) for c in args.metrics]
        missing = [c for c in metrics if c not in df.columns]
        if missing:
            raise ValueError(
                f"Requested metric columns not found: {missing}. "
                f"Available columns: {list(df.columns)}"
            )
    else:
        metrics = infer_metric_columns(df)

    summary = save_summary(df, metrics, outdir)

    print("\nLoaded files:")
    for p in paths:
        print(f"  {p}")

    print(f"\nRows: {len(df)}")
    print(f"Runs: {df['run'].nunique()}")
    print(f"Output directory: {outdir}")

    print("\nDetected columns:")
    print(f"  time column: {time_col}")
    print(f"  metrics: {metrics}")

    print("\nSummary:")
    print(summary.to_string(index=False))

    if time_col is not None:
        for metric in metrics:
            if metric == time_col:
                continue
            plot_timeseries(df, time_col, metric, outdir)

    for metric in metrics:
        plot_boxplot(df, metric, outdir)

    id_col = infer_column(df.columns, EVENT_ID_CANDIDATES)
    if time_col is not None and id_col is not None:
        plot_completion_throughput(
            df=df,
            time_col=time_col,
            id_col=id_col,
            outdir=outdir,
            bin_width_s=args.bin_width_s,
        )

    print("\nWrote:")
    print(f"  {outdir / 'combined.csv'}")
    print(f"  {outdir / 'summary_by_run.csv'}")
    print(f"  {outdir}/*.png")


if __name__ == "__main__":
    main()
