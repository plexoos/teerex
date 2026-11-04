# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.25.2", "matplotlib", "pandas"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Measured CPU/GPU timelines")


@app.cell(hide_code=True)
async def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    from teerex.analysis import plot_run_timeline, read_one_csv

    if sys.platform == "emscripten":
        from pyodide.http import pyfetch

        _path = Path("/tmp/events.csv.gz")
        _response = await pyfetch(str(mo.notebook_location() / "public" / _path.name))
        if not _response.ok:
            raise RuntimeError(f"Could not load production events: HTTP {_response.status}")
        _path.write_bytes(await _response.bytes())
        default_csv = str(_path)
    else:
        default_csv = os.environ.get(
            "TEEREX_TIMELINE_CSV",
            str(Path(__file__).resolve().parents[1] / "docs" / "study" / "events.csv.gz"),
        )
    group_specs = [
        dict(label="pfRICH · 10 primaries", geometry="pfrich", primaries=10),
        dict(label="pfRICH · 100 primaries", geometry="pfrich", primaries=100),
        dict(label="dRICH · 10 primaries", geometry="drich", primaries=10),
        dict(label="dRICH · 100 primaries", geometry="drich", primaries=100),
        dict(label="Balanced argon · 65,536 × 1 keV", geometry="argon",
             primaries=65536, kinetic_mev=0.001),
        dict(label="GPU-heavy argon · 10 × 10 MeV", geometry="argon",
             primaries=10, kinetic_mev=10.0),
        dict(label="GPU-heavy argon · 100 × 1 MeV", geometry="argon",
             primaries=100, kinetic_mev=1.0),
    ]
    return (
        FuncFormatter, Patch, Path, default_csv, group_specs, mo, pd,
        plot_run_timeline, plt, read_one_csv,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Measured CPU/GPU timelines

    Seven workload groups cover CPU-dominated RICH events, balanced liquid
    argon, and GPU-dominated liquid argon. Each RICH group has **electron and
    muon configurations at 20 GeV/c**; argon energies are **kinetic energies**.
    There are eleven configurations, each with 2,000 analyzed events across
    four seed pairs. The first five events of each run are already excluded.

    **CPU** includes generation and cleanup. **GPU** is host-observed service,
    including transfers and synchronization.

    The configuration table uses all events. Timelines default to **16
    consecutive events from one seed run**. Each panel starts at its first
    displayed event and has its own time scale. Switch the RICH particle to
    view muons or both particle types; independent run clocks stay separate.
    """)
    return


@app.cell(hide_code=True)
def _(default_csv, mo):
    csv_path = mo.ui.text(
        value=default_csv, label="Production event CSV", full_width=True,
    )
    csv_path
    return (csv_path,)


@app.cell(hide_code=True)
def _(Path, csv_path, mo, read_one_csv):
    _source = Path(csv_path.value).expanduser()
    mo.stop(
        not _source.is_file(),
        mo.callout(
            "Enter a path to the bundled events.csv.gz or another ratio-study-v3 CSV.",
            kind="warn",
        ),
    )
    try:
        production_events = read_one_csv(_source)
        _required = {
            "geometry_name", "particle", "momentum_gev_c", "multiplicity",
            "kinetic_energy_mev", "run_id", "replication", "event_id",
            "num_photons", "cpu_pre_runtime_s", "cpu_post_runtime_s",
            "gpu_runtime_s", "cpu_pre_start_time_offset_s",
            "cpu_pre_end_time_offset_s", "cpu_post_start_time_offset_s",
            "cpu_post_end_time_offset_s", "gpu_start_time_offset_s",
            "gpu_end_time_offset_s", "start_offset_s", "end_offset_s",
        }
        _missing = sorted(_required.difference(production_events.columns))
        if _missing:
            raise ValueError("Missing measurement columns: " + ", ".join(_missing))
        if production_events.duplicated(["run_id", "event_id"]).any():
            raise ValueError("The CSV contains duplicate run/event identifiers.")
    except (OSError, ValueError) as _error:
        mo.stop(True, mo.callout(str(_error), kind="warn"))
    return (production_events,)


@app.cell(hide_code=True)
def _(group_specs, mo, pd, production_events):
    configuration_frames = {}
    _summary_rows = []
    for _spec in group_specs:
        _particles = ["e-"] if _spec["geometry"] == "argon" else ["e-", "mu-"]
        for _particle in _particles:
            _mask = (
                production_events.geometry_name.eq(_spec["geometry"])
                & production_events.multiplicity.eq(_spec["primaries"])
                & production_events.particle.eq(_particle)
            )
            if _spec["geometry"] == "argon":
                _mask &= production_events.kinetic_energy_mev.eq(_spec["kinetic_mev"])
            else:
                _mask &= production_events.momentum_gev_c.eq(20)
            _frame = production_events.loc[_mask].copy()
            mo.stop(
                _frame.empty,
                mo.callout(f"Missing configuration: {_spec['label']} ({_particle}).", kind="warn"),
            )
            configuration_frames[(_spec["label"], _particle)] = _frame
            _cpu = _frame.cpu_pre_runtime_s + _frame.cpu_post_runtime_s
            _gpu = _frame.gpu_runtime_s
            _summary_rows.append({
                "Workload": _spec["label"],
                "Particle": "μ−" if _particle == "mu-" else "e−",
                "Events": len(_frame),
                "Seed pairs": _frame.replication.nunique(),
                "Mean photons/event": round(_frame.num_photons.mean()),
                "Median CPU (ms)": _cpu.median() * 1000,
                "Median GPU (ms)": _gpu.median() * 1000,
                "Median CPU:GPU": (_cpu / _gpu).median(),
                "Median GPU share (%)": (_gpu / (_cpu + _gpu)).median() * 100,
            })
    configuration_summary = pd.DataFrame(_summary_rows)
    available_seeds = sorted(set.intersection(*[
        set(int(rep) for rep in frame.replication.unique())
        for frame in configuration_frames.values()
    ]))
    mo.stop(not available_seeds, mo.callout("No seed pair is shared by all configurations.", kind="warn"))
    min_run_events = min(
        int(frame.groupby("run_id").size().min())
        for frame in configuration_frames.values()
    )
    return available_seeds, configuration_frames, configuration_summary, min_run_events


@app.cell(hide_code=True)
def _(configuration_summary, mo):
    mo.vstack([
        mo.md("## Configuration summary\n\nAll seed runs contribute to these statistics; changing the displayed window does not change this table. CPU:GPU is the median of event-wise ratios."),
        mo.ui.table(
            configuration_summary.round(3), selection=None, page_size=11,
            show_column_summaries=False,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(available_seeds, group_specs, min_run_events, mo):
    _labels = [spec["label"] for spec in group_specs]
    dataset_selection = mo.ui.multiselect(
        options=_labels, value=_labels, label="Workload groups", full_width=True,
    )
    rich_particle = mo.ui.dropdown(
        options={"Electrons (e−)": "e-", "Muons (μ−)": "mu-", "Both": "both"},
        value="Electrons (e−)", label="RICH particle",
    )
    _seed_options = {f"Seed pair {rep}": rep for rep in available_seeds}
    seed_pair = mo.ui.dropdown(
        options=_seed_options, value=next(iter(_seed_options)), label="Seed run",
    )
    window_size = mo.ui.slider(
        start=1, stop=min(128, min_run_events), value=min(16, min_run_events),
        step=1, label="Events per timeline", show_value=True,
        include_input=True, debounce=True,
    )
    mo.vstack([
        mo.md("## Timeline windows"), dataset_selection,
        mo.hstack([rich_particle, seed_pair, window_size]),
    ])
    return dataset_selection, rich_particle, seed_pair, window_size


@app.cell(hide_code=True)
def _(min_run_events, mo, window_size):
    window_offset = mo.ui.slider(
        start=0, stop=max(0, min_run_events - window_size.value), step=1, value=0,
        label="Skip analyzed events in each run", show_value=True,
        include_input=True, debounce=True,
    )
    window_offset
    return (window_offset,)


@app.cell(hide_code=True)
def _(
    configuration_frames, dataset_selection, group_specs, mo, pd,
    rich_particle, seed_pair, window_offset, window_size,
):
    mo.stop(not dataset_selection.value, mo.callout("Select a workload group to display."))
    _windows = []
    panel_info = []
    for _spec in group_specs:
        if _spec["label"] not in dataset_selection.value:
            continue
        _particles = (
            ["e-"] if _spec["geometry"] == "argon"
            else (["e-", "mu-"] if rich_particle.value == "both" else [rich_particle.value])
        )
        for _particle in _particles:
            _configuration = configuration_frames[(_spec["label"], _particle)]
            _run = _configuration.loc[
                _configuration.replication.eq(seed_pair.value)
            ].sort_values("event_id")
            mo.stop(
                _run.run_id.nunique() != 1,
                mo.callout("Each displayed timeline must come from one process.", kind="warn"),
            )
            _window = _run.iloc[
                window_offset.value:window_offset.value + window_size.value
            ].copy()
            _particle_name = "μ−" if _particle == "mu-" else "e−"
            _title = f"{_spec['label']} · {_particle_name} · seed pair {seed_pair.value}"
            _origin = float(_window.cpu_pre_start_time_offset_s.iloc[0])
            _offset_columns = [
                column for column in _window
                if column.endswith("_time_offset_s") or column in ("start_offset_s", "end_offset_s")
            ]
            _window[_offset_columns] = _window[_offset_columns] - _origin
            _window["scenario"] = _title
            _windows.append(_window)
            _cpu = _configuration.cpu_pre_runtime_s + _configuration.cpu_post_runtime_s
            _gpu = _configuration.gpu_runtime_s
            panel_info.append({
                "title": _title,
                "events": len(_window),
                "first_event": int(_window.event_id.iloc[0]),
                "last_event": int(_window.event_id.iloc[-1]),
                "span_s": float(_window.end_offset_s.max()),
                "full_events": len(_configuration),
                "full_ratio": float((_cpu / _gpu).median()),
                "full_gpu_share": float((_gpu / (_cpu + _gpu)).median()) * 100,
            })
    windowed_events = pd.concat(_windows, ignore_index=True)
    return panel_info, windowed_events


@app.cell(hide_code=True)
def _(FuncFormatter, Patch, mo, panel_info, plot_run_timeline, plt, windowed_events):
    _scenarios = [panel["title"] for panel in panel_info]
    timeline_figure, _axes, _ = plot_run_timeline(
        windowed_events, _scenarios, sharex=False,
        figsize=(12, 2.35 * len(_scenarios) + 0.55),
    )
    _plot_scale = (
        1000.0 if windowed_events[["cpu_end_time_offset_s", "gpu_end_time_offset_s"]].max().max() < 1
        else 1.0
    )
    for _axis, _panel in zip(_axes, panel_info):
        _unit = "ms" if _panel["span_s"] < 1 else "s"
        _display_scale = 1000.0 if _unit == "ms" else 1.0
        _factor = _display_scale / _plot_scale
        _axis.xaxis.set_major_formatter(FuncFormatter(
            lambda value, position, factor=_factor: f"{value * factor:.3g}"
        ))
        _axis.set_xlabel(f"Elapsed time from first displayed event ({_unit})", fontsize=9)
        _axis.set_ylabel("")
        _axis.set_title(_panel["title"], loc="left", fontsize=11, fontweight="bold", pad=27)
        for _text in list(_axis.texts):
            if "makespan" in _text.get_text():
                _text.remove()
        _axis.get_legend().remove()
        _axis.text(
            0, 1.02,
            f"Events {_panel['first_event']}–{_panel['last_event']} · {_panel['events']} events · "
            f"window {_panel['span_s'] * _display_scale:.3g} {_unit} · "
            f"all {_panel['full_events']:,} events: CPU:GPU {_panel['full_ratio']:.2f}:1, "
            f"GPU {_panel['full_gpu_share']:.1f}%",
            transform=_axis.transAxes, fontsize=8, color="0.35", va="bottom",
        )
    timeline_figure.legend(
        handles=[Patch(color="tab:blue", label="CPU generation + cleanup"),
                 Patch(color="tab:orange", label="GPU host service")],
        loc="upper right", ncol=2, frameon=False, fontsize=9,
    )
    timeline_figure.tight_layout(rect=(0, 0, 1, 0.975))
    plt.close(timeline_figure)
    mo.vstack([
        mo.md("Each axis has its own scale. Event labels use the original recorded IDs; only the displayed window is rebased to zero."),
        timeline_figure,
    ])
    return (timeline_figure,)


if __name__ == "__main__":
    app.run()
