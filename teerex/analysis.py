#!/usr/bin/env python3

"""
Analyze and compare multiple Simload CSV output files.

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


TIMELINE_TIME_COLUMNS = (
    "start_time",
    "end_time",
    "cpu_start_time",
    "cpu_end_time",
    "cpu_pre_start_time",
    "cpu_pre_end_time",
    "cpu_post_start_time",
    "cpu_post_end_time",
    "gpu_submit_time",
    "gpu_start_time",
    "gpu_end_time",
    "gpu_wait_start_time",
    "gpu_wait_end_time",
    "merge_start_time",
    "merge_end_time",
)

TIMELINE_REQUIRED_COLUMNS = (
    "scenario",
    "event_id",
    "cpu_start_time_offset_s",
    "cpu_end_time_offset_s",
    "gpu_start_time_offset_s",
    "gpu_end_time_offset_s",
)


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


def normalize_timeline(
    df: pd.DataFrame,
    default_scenario: str = "run",
) -> pd.DataFrame:
    """Normalize simulated and measured event data to the shared timeline schema.

    New Simload and simg4ox outputs already contain run-local offset columns.
    Older Simload files contain only absolute monotonic timestamps, so offsets
    are derived independently for each scenario. Producer-specific workload
    columns are preserved.
    """

    normalized = df.copy()

    if "scenario" not in normalized.columns:
        normalized["scenario"] = default_scenario
    else:
        normalized["scenario"] = normalized["scenario"].fillna(default_scenario)

    if "dispatch_mode" not in normalized.columns:
        if "gpu_mode" in normalized.columns:
            normalized["dispatch_mode"] = normalized["gpu_mode"]
        elif "gpu_async" in normalized.columns:
            is_async = normalized["gpu_async"].astype(str).str.lower().isin(("true", "1", "yes"))
            normalized["dispatch_mode"] = is_async.map({True: "async", False: "blocking"})
        else:
            normalized["dispatch_mode"] = "unknown"
    if "gpu_mode" not in normalized.columns:
        normalized["gpu_mode"] = normalized["dispatch_mode"]
    if "gpu_async" not in normalized.columns:
        normalized["gpu_async"] = normalized["dispatch_mode"].eq("async")

    for _, indices in normalized.groupby("scenario", dropna=False).groups.items():
        start_columns = [
            column
            for column in (
                "start_time",
                "cpu_start_time",
                "cpu_pre_start_time",
                "gpu_start_time",
            )
            if column in normalized.columns
        ]
        if not start_columns:
            continue

        origin = normalized.loc[indices, start_columns].min().min()
        if pd.isna(origin):
            continue

        for column in TIMELINE_TIME_COLUMNS:
            if column not in normalized.columns:
                continue
            offset_column = f"{column}_offset_s"
            derived = normalized.loc[indices, column] - origin
            if offset_column not in normalized.columns:
                normalized.loc[indices, offset_column] = derived
            else:
                normalized.loc[indices, offset_column] = normalized.loc[
                    indices, offset_column
                ].fillna(derived)

    missing = [column for column in TIMELINE_REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing timeline columns: {', '.join(missing)}")

    return normalized


def read_one_csv(path: Path, scenario_label: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    original_cols = list(df.columns)
    df.columns = [normalize_column_name(c) for c in df.columns]

    if len(set(df.columns)) != len(df.columns):
        raise ValueError(
            f"Column-name collision after normalization in {path}. "
            f"Original columns were: {original_cols}"
        )

    default_scenario = scenario_label or path.stem
    looks_like_timeline = (
        "event_id" in df.columns
        and any(column in df.columns for column in ("cpu_start_time", "cpu_start_time_offset_s"))
        and any(column in df.columns for column in ("gpu_start_time", "gpu_start_time_offset_s"))
    )
    if looks_like_timeline:
        df = normalize_timeline(df, default_scenario=default_scenario)
    elif "scenario" not in df.columns:
        df["scenario"] = default_scenario

    if scenario_label is not None:
        df["scenario"] = scenario_label
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


def timeline_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row of numerical scheduling metrics per scenario."""

    rows = []
    for scenario, run in df.groupby("scenario", dropna=False):
        start = run["cpu_start_time_offset_s"].min()
        end = run[["cpu_end_time_offset_s", "gpu_end_time_offset_s"]].max().max()
        makespan = max(0.0, end - start)
        cpu_busy = run["cpu_runtime_s"].sum() if "cpu_runtime_s" in run else float("nan")
        gpu_busy = run["gpu_runtime_s"].sum() if "gpu_runtime_s" in run else float("nan")
        row = {
            "scenario": scenario,
            "events": run["event_id"].nunique(),
            "makespan_s": makespan,
            "cpu_busy_s": cpu_busy,
            "gpu_busy_s": gpu_busy,
            "cpu_busy_fraction": cpu_busy / makespan if makespan > 0 else float("nan"),
            "gpu_busy_fraction": gpu_busy / makespan if makespan > 0 else float("nan"),
        }
        optional_totals = {
            "gpu_queue_delay_s": "gpu_queue_total_s",
            "gpu_wait_runtime_s": "gpu_wait_total_s",
            "num_photons": "num_photons",
        }
        for column, output_column in optional_totals.items():
            if column in run:
                row[output_column] = run[column].sum(min_count=1)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_run_timeline(
    df: pd.DataFrame,
    run: str | list[str],
    outdir: Path | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
):
    """Plot CPU/GPU busy lanes for one or more normalized scenarios."""

    single_run = isinstance(run, str)
    runs = [run] if single_run else list(run)
    if not runs:
        raise ValueError("run must contain at least one scenario label")

    available_runs = sorted(df["scenario"].dropna().unique())
    missing_runs = [run_label for run_label in runs if run_label not in available_runs]
    if missing_runs:
        raise ValueError(f"Scenario labels not found: {missing_runs}. Available: {available_runs}")

    run_timelines = [
        df.loc[df["scenario"].eq(run_label)].sort_values("event_id").reset_index(drop=True)
        for run_label in runs
    ]
    timeline = pd.concat(run_timelines, ignore_index=True, sort=False)
    x_max = timeline[["cpu_end_time_offset_s", "gpu_end_time_offset_s"]].max().max()
    time_scale = 1_000.0 if x_max < 1.0 else 1.0
    time_unit = "ms" if time_scale == 1_000.0 else "s"
    shared_x_limit = x_max * time_scale * 1.02 if x_max > 0 else 1.0

    if figsize is None:
        figsize = (11, 3 * len(runs))

    resource_y = {"GPU": 0, "CPU": 1}
    fig, axes = plt.subplots(len(runs), 1, figsize=figsize, sharex=sharex, squeeze=False)
    axes = axes.ravel()

    for axis, run_label, run_timeline in zip(axes, runs, run_timelines):
        label_stride = max(1, (len(run_timeline) + 11) // 12)
        for row_number, (_, row) in enumerate(run_timeline.iterrows()):
            event_start = row["cpu_start_time_offset_s"]
            if pd.isna(event_start):
                continue
            event_start *= time_scale
            axis.axvline(event_start, color="0.25", linewidth=0.8, alpha=0.35, zorder=0)
            if row_number % label_stride == 0:
                axis.annotate(
                    f"E{int(row['event_id'])}",
                    (event_start, 1.43),
                    xytext=(3, 0),
                    textcoords="offset points",
                    color="0.3",
                    fontsize=8,
                    ha="left",
                    va="center",
                )

        labels_drawn = {"CPU": False, "GPU": False}
        for _, row in run_timeline.iterrows():
            phase_columns = (
                "cpu_pre_start_time_offset_s",
                "cpu_pre_end_time_offset_s",
                "cpu_post_start_time_offset_s",
                "cpu_post_end_time_offset_s",
            )
            if all(column in row.index and pd.notna(row[column]) for column in phase_columns):
                cpu_segments = [
                    (row[phase_columns[0]], row[phase_columns[1]]),
                    (row[phase_columns[2]], row[phase_columns[3]]),
                ]
            else:
                cpu_segments = [
                    (row["cpu_start_time_offset_s"], row["cpu_end_time_offset_s"])
                ]

            for cpu_start, cpu_end in cpu_segments:
                if pd.isna(cpu_start) or pd.isna(cpu_end) or cpu_end <= cpu_start:
                    continue
                axis.hlines(
                    resource_y["CPU"],
                    cpu_start * time_scale,
                    cpu_end * time_scale,
                    color="tab:blue",
                    linewidth=11,
                    label="CPU busy" if not labels_drawn["CPU"] else "_nolegend_",
                )
                labels_drawn["CPU"] = True

            gpu_start = row["gpu_start_time_offset_s"]
            gpu_end = row["gpu_end_time_offset_s"]
            if pd.notna(gpu_start) and pd.notna(gpu_end) and gpu_end > gpu_start:
                axis.hlines(
                    resource_y["GPU"],
                    gpu_start * time_scale,
                    gpu_end * time_scale,
                    color="tab:orange",
                    linewidth=11,
                    label="GPU busy" if not labels_drawn["GPU"] else "_nolegend_",
                )
                labels_drawn["GPU"] = True

        run_x_max = run_timeline[["cpu_end_time_offset_s", "gpu_end_time_offset_s"]].max().max()
        run_x_limit = run_x_max * time_scale * 1.02 if run_x_max > 0 else 1.0
        cpu_busy = run_timeline["cpu_runtime_s"].sum() * time_scale
        gpu_busy = run_timeline["gpu_runtime_s"].sum() * time_scale
        axis.set_title(
            str(run_label).replace("_", " "),
            loc="left",
            fontsize=13,
            fontweight="bold",
            pad=25,
        )
        axis.text(
            0.0,
            1.02,
            (
                f"{run_timeline['event_id'].nunique()} events  ·  "
                f"makespan {run_x_max * time_scale:.3f} {time_unit}  ·  "
                f"CPU busy {cpu_busy:.3f} {time_unit}  ·  "
                f"GPU busy {gpu_busy:.3f} {time_unit}"
            ),
            transform=axis.transAxes,
            color="0.35",
            fontsize=9,
            ha="left",
            va="bottom",
        )
        axis.set_ylabel("resource", color="0.3")
        axis.set_yticks([resource_y["GPU"], resource_y["CPU"]])
        axis.set_yticklabels(["GPU", "CPU"])
        axis.set_ylim(-0.6, 1.6)
        axis.set_xlim(0, shared_x_limit if sharex else run_x_limit)
        axis.grid(True, axis="x", color="0.85", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0,
            frameon=False,
        )

    axes[-1].set_xlabel(f"time since first stage start in each run ({time_unit})")
    fig.tight_layout()

    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{'_'.join(runs)}_cpu_gpu_busy_intervals.png"
        fig.savefig(outdir / filename, dpi=160)

    return fig, axes[0] if single_run else axes, timeline


def save_summary(df: pd.DataFrame, metrics: list[str], outdir: Path) -> pd.DataFrame:
    if not metrics:
        summary = (
            df.groupby("scenario", dropna=False).size().rename("rows").reset_index()
        )
        summary.to_csv(outdir / "summary_by_scenario.csv", index=False)
        return summary

    summary = (
        df.groupby("scenario", dropna=False)[metrics]
        .agg(["count", "mean", "std", "min", "median", "max"])
    )

    summary.columns = [
        f"{metric}_{stat}" for metric, stat in summary.columns.to_flat_index()
    ]

    summary = summary.reset_index()
    summary.to_csv(outdir / "summary_by_scenario.csv", index=False)

    return summary


def plot_boxplot(
    df: pd.DataFrame,
    metric: str,
    outdir: Path,
) -> None:
    data = []
    labels = []

    for scenario, g in df.groupby("scenario", dropna=False):
        values = g[metric].dropna()
        if values.empty:
            continue
        data.append(values)
        labels.append(str(scenario))

    if not data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_title(f"{metric} distribution by scenario")
    ax.set_xlabel("scenario")
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
    tmp = df[[time_col, id_col, "scenario"]].dropna().copy()

    if tmp.empty:
        return

    tmp["time_bin"] = (tmp[time_col] // bin_width_s) * bin_width_s

    throughput = (
        tmp.groupby(["scenario", "time_bin"], dropna=False)[id_col]
        .nunique()
        .reset_index(name="completed")
    )

    throughput["throughput_per_s"] = throughput["completed"] / bin_width_s

    fig, ax = plt.subplots(figsize=(10, 5))

    for scenario, g in throughput.groupby("scenario", dropna=False):
        g = g.sort_values("time_bin")
        ax.plot(
            g["time_bin"],
            g["throughput_per_s"],
            label=str(scenario),
            linewidth=1.5,
        )

    ax.set_title("Completion throughput over time")
    ax.set_xlabel(time_col)
    ax.set_ylabel("completed tasks/events per second")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "throughput_over_time.png", dpi=160)
    plt.close(fig)
