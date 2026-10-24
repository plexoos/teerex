import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    from teerex.analysis import (
        plot_run_timeline,
        read_many_csv,
        timeline_summary,
    )

    notebook_dir = Path(__file__).resolve().parent
    return Path, mo, notebook_dir, plot_run_timeline, read_many_csv, timeline_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CPU/GPU event timeline comparison

    Compare synthetic teerex runs with measured event timelines such as
    `simg4ox` output. The shared columns describe event IDs, dispatch modes,
    CPU pre/post intervals, GPU submission/execution, queueing, waiting, and
    run-local time offsets. Producer-specific payload columns remain available
    in the event table.
    """)
    return


@app.cell
def _(notebook_dir):
    simphony_timing_dir = notebook_dir.parents[2] / "pfrich_timing"
    measured_paths = (
        sorted(
            simphony_timing_dir.rglob("events.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if simphony_timing_dir.exists()
        else []
    )
    candidate_paths = [
        notebook_dir.parent / "simload_runs" / "mode_blocking.csv",
        notebook_dir.parent / "simload_runs" / "mode_async.csv",
        *measured_paths[:1],
    ]
    default_paths = [path for path in candidate_paths if path.exists()]
    return (default_paths,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inputs

    Enter one CSV path per line. Existing `scenario` values are preserved;
    older files without that column use their filename stem. Absolute
    monotonic timestamps are normalized to offsets independently for each
    scenario.
    """)
    return


@app.cell
def _(default_paths, mo):
    csv_paths_input = mo.ui.text_area(
        value="\n".join(str(path) for path in default_paths),
        label="Timeline CSV files",
        rows=max(4, len(default_paths) + 1),
        full_width=True,
    )
    csv_paths_input
    return (csv_paths_input,)


@app.cell
def _(Path, csv_paths_input, mo):
    csv_paths = [
        Path(line.strip()).expanduser()
        for line in csv_paths_input.value.splitlines()
        if line.strip()
    ]
    missing_paths = [path for path in csv_paths if not path.exists()]
    mo.stop(not csv_paths, mo.callout("Add at least one timeline CSV.", kind="warn"))
    mo.stop(
        bool(missing_paths),
        mo.callout(
            "Missing timeline CSV files:\n\n"
            + "\n".join(f"- `{path}`" for path in missing_paths),
            kind="danger",
        ),
    )
    return (csv_paths,)


@app.cell
def _(csv_paths, read_many_csv):
    events = read_many_csv(csv_paths)
    scenario_labels = sorted(events["scenario"].dropna().unique().tolist())
    return events, scenario_labels


@app.cell
def _(mo, scenario_labels):
    scenario_selector = mo.ui.multiselect(
        options=scenario_labels,
        value=scenario_labels,
        label="Scenarios",
        full_width=True,
    )
    shared_time_scale = mo.ui.checkbox(
        value=False,
        label="Use one shared time scale",
    )
    mo.vstack([scenario_selector, shared_time_scale])
    return scenario_selector, shared_time_scale


@app.cell
def _(events, mo, scenario_selector):
    selected_events = events.loc[
        events["scenario"].isin(scenario_selector.value)
    ].copy()
    mo.stop(
        selected_events.empty,
        mo.callout("Select at least one scenario.", kind="warn"),
    )
    return (selected_events,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical summary

    Busy fractions divide the sum of measured busy intervals by scenario
    makespan. Queue, wait, and optical-photon totals appear when the producer
    records them.
    """)
    return


@app.cell
def _(mo, selected_events, timeline_summary):
    summary = timeline_summary(selected_events)
    summary_table = mo.ui.table(
        summary,
        selection=None,
        show_column_summaries=False,
        show_download=True,
    )
    summary_table
    return (summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## CPU/GPU busy lanes

    Vertical rules mark event starts. New-schema runs display CPU work before
    and after GPU dispatch as separate segments; legacy runs fall back to the
    whole recorded CPU interval. Independent x-axes make millisecond measured
    runs legible beside longer synthetic runs; enable the shared scale for a
    direct makespan comparison.
    """)
    return


@app.cell
def _(plot_run_timeline, scenario_selector, selected_events, shared_time_scale):
    timeline_figure, _, timeline = plot_run_timeline(
        selected_events,
        scenario_selector.value,
        sharex=shared_time_scale.value,
    )
    timeline_figure
    return (timeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Event-level numbers

    Search, filter, sort, or download the normalized rows below. This table is
    the numerical companion to the busy-lane figure.
    """)
    return


@app.cell
def _(mo, timeline):
    preferred_columns = [
        "scenario",
        "dispatch_mode",
        "event_id",
        "num_photons",
        "start_offset_s",
        "end_offset_s",
        "cpu_pre_runtime_s",
        "cpu_post_runtime_s",
        "cpu_runtime_s",
        "gpu_queue_delay_s",
        "gpu_runtime_s",
        "gpu_wait_runtime_s",
    ]
    event_columns = [column for column in preferred_columns if column in timeline.columns]
    event_table = mo.ui.table(
        timeline[event_columns],
        selection=None,
        page_size=20,
        show_column_summaries=False,
        show_download=True,
    )
    event_table
    return


if __name__ == "__main__":
    app.run()
