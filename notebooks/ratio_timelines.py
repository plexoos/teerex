# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.25.2", "matplotlib", "numpy", "pandas", "plotly"]
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
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from teerex.analysis import read_one_csv

    if sys.platform == "emscripten":
        from pyodide.http import pyfetch

        _path = Path("/tmp/events.csv.gz")
        _response = await pyfetch(str(mo.notebook_location() / "public" / _path.name))
        if not _response.ok:
            raise RuntimeError(f"Could not load production events: HTTP {_response.status}")
        _path.write_bytes(await _response.bytes())
        default_csv = str(_path)
    else:
        _path = Path(__file__).resolve().parents[1] / "docs" / "study" / "events.csv"
        if not _path.is_file():
            _path = _path.with_suffix(".csv.gz")
        default_csv = os.environ.get("TEEREX_TIMELINE_CSV", str(_path))
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
        Path, default_csv, go, group_specs, make_subplots, mo, pd, read_one_csv,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Measured CPU/GPU timelines

    The bundled dataset covers seven electron workloads from blocking runs.
    Four CPU-dominated RICH workloads use pfRICH or dRICH, with 10 or 100
    primary electrons per event at **20 GeV/c** per primary. The three
    liquid-argon workloads span one balanced and two GPU-dominated
    configurations. The labels follow the run settings: RICH uses primary
    momentum, while argon uses kinetic energy per primary. For example,
    **10 × 10 MeV** denotes 10 primaries per event at 10 MeV each.

    Each workload has **2,000 analyzed events**, with 500 events from each
    of four runs. Each run uses one pair of random-number seeds, one for
    CLHEP and one for the GPU. The dataset excludes the first five events
    of each run.

    The first blue segment shows CPU generation. The second shows CPU
    post-processing, including cleanup. The gap after it is elapsed time
    before the next event starts. The CPU time metric sums the two blue
    intervals, and the recorded timestamps give only the duration of the gap.
    The orange segment shows host-observed GPU service, including transfers
    and synchronization.

    The configuration summary and histograms draw on all four seed runs.
    Each timeline opens on the first 10 analyzed events of the first seed
    run. Drag to pan, use the toolbar to zoom, or adjust the range slider
    below the time axis to explore the rest of that run.
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
            "Enter a path to the bundled events.csv or another ratio-study-v3 CSV.",
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
    _has_muons = production_events.particle.eq("mu-").any()
    for _spec in group_specs:
        _particles = ["e-", "mu-"] if _has_muons and _spec["geometry"] != "argon" else ["e-"]
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
    first_seed = available_seeds[0]
    return configuration_frames, configuration_summary, first_seed


@app.cell(hide_code=True)
def _(configuration_summary, mo):
    mo.vstack([
        mo.md("## Configuration summary\n\nAll seed runs contribute to these statistics; changing the displayed window does not change this table. CPU:GPU is the median of event-wise ratios. Median GPU share (%) is the median across events of 100 × GPU time / (CPU time + GPU time)."),
        mo.ui.table(
            configuration_summary.round(3), selection=None, page_size=11,
            show_column_summaries=False,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(configuration_frames, first_seed, go, mo):
    timeline_figures = []
    for _configuration in configuration_frames.values():
        _run = _configuration.loc[
            _configuration.replication.eq(first_seed)
        ].sort_values("event_id")
        mo.stop(
            _run.run_id.nunique() != 1,
            mo.callout("Each displayed timeline must come from one process.", kind="warn"),
        )

        _origin = _run.start_offset_s.iloc[0]
        _initial_end_ms = (_run.head(10).end_offset_s.max() - _origin) * 1000
        _run_end_ms = (_run.end_offset_s.max() - _origin) * 1000
        _figure = go.Figure()
        for _phase, _lane, _name, _color in [
            ("cpu_pre", "CPU", "CPU generation", "#1f77b4"),
            ("cpu_post", "CPU", "CPU cleanup", "#1f77b4"),
            ("gpu", "GPU", "GPU host service", "#ff7f0e"),
        ]:
            _starts = _run[f"{_phase}_start_time_offset_s"]
            _ends = _run[f"{_phase}_end_time_offset_s"]
            _valid = _ends.gt(_starts)
            _figure.add_bar(
                orientation="h",
                x=((_ends - _starts)[_valid] * 1000).tolist(),
                base=((_starts[_valid] - _origin) * 1000).tolist(),
                y=[_lane] * int(_valid.sum()),
                customdata=_run.loc[_valid, "event_id"].tolist(),
                name=_name, marker_color=_color, width=0.55,
                hovertemplate="Event %{customdata}<br>%{x:.3g} ms<extra>%{fullData.name}</extra>",
            )
        _figure.update_layout(
            template="plotly_white", height=310, barmode="overlay", dragmode="pan", showlegend=False,
            margin=dict(l=55, r=25, t=15, b=45),
            xaxis=dict(
                title="Elapsed time (ms)", range=[0, float(_initial_end_ms)],
                minallowed=0, gridcolor="#e5e7eb",
                rangeslider=dict(
                    visible=True, thickness=0.18, range=[0, float(_run_end_ms)],
                ),
            ),
            yaxis=dict(categoryorder="array", categoryarray=["GPU", "CPU"], fixedrange=True),
        )
        timeline_figures.append(_figure)
    return (timeline_figures,)


@app.cell(hide_code=True)
def _(configuration_frames, go, make_subplots):
    import numpy as np

    distribution_figures = []
    _colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    _dashes = ["solid", "dash", "dot", "dashdot"]
    for (_label, _), _frame in configuration_frames.items():
        _cpu_upper = 0.97 if _label in (
            "Balanced argon · 65,536 × 1 keV",
            "GPU-heavy argon · 100 × 1 MeV",
        ) else 0.99
        _figure = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "table"}, {"type": "table"}]],
            row_heights=[0.67, 0.33], vertical_spacing=0.17,
            subplot_titles=["CPU generation + cleanup", "GPU host service"],
        )
        _cpu_ms = (_frame.cpu_pre_runtime_s + _frame.cpu_post_runtime_s) * 1000
        _gpu_ms = _frame.gpu_runtime_s * 1000
        _seeds = sorted(_frame.replication.unique())

        for _column, _times in enumerate((_cpu_ms, _gpu_ms), start=1):
            _valid = _times[np.isfinite(_times) & (_times > 0)]
            _upper = _cpu_upper if _column == 1 else 0.99
            if _valid.empty:
                _low, _high = 0, 1
            else:
                _low, _high = _valid.quantile([0.01, _upper])
                _padding = 0.25 * (_high - _low) if _high > _low else 0.01 * _low
                _low, _high = max(0, _low - _padding), _high + _padding
            _bins = np.linspace(_low, _high, 26)
            _stats = {
                "Seed pair": [], "Shown / total": [], "Mean (ms)": [],
                "Std dev (ms)": [], "CV (%)": [],
            }
            for _index, _seed in enumerate(_seeds):
                _sample = _valid.loc[_frame.replication.eq(_seed)]
                _shown = _sample[_sample.between(_low, _high)]
                _counts, _ = np.histogram(_shown, bins=_bins)
                _figure.add_scatter(
                    x=_bins.repeat(2)[1:-1].tolist(), y=_counts.repeat(2).tolist(),
                    mode="lines", name=f"Seed pair {_seed}", legendgroup=str(_seed),
                    showlegend=_column == 1,
                    line=dict(
                        color=_colors[_index % len(_colors)],
                        dash=_dashes[_index % len(_dashes)],
                    ),
                    hovertemplate="%{y} events<extra>%{fullData.name}</extra>",
                    row=1, col=_column,
                )
                _stats["Seed pair"].append(str(_seed))
                _stats["Shown / total"].append(f"{len(_shown)} / {len(_sample)}")
                _mean = _sample.mean()
                _std = _sample.std(ddof=0)
                _stats["Mean (ms)"].append(f"{_mean:.4g}" if len(_sample) else "—")
                _stats["Std dev (ms)"].append(f"{_std:.4g}" if len(_sample) else "—")
                _stats["CV (%)"].append(f"{100 * (_std / _mean):.4g}" if _mean > 0 else "—")

            _figure.add_table(
                header=dict(values=list(_stats), fill_color="#f3f4f6", align="center"),
                cells=dict(values=list(_stats.values()), fill_color="white", align="center"),
                row=2, col=_column,
            )
        _figure.update_xaxes(type="linear", title_text="Time per event (ms)", gridcolor="#e5e7eb")
        _figure.update_yaxes(title_text="Events", rangemode="tozero", gridcolor="#e5e7eb")
        _figure.update_layout(
            template="plotly_white", height=465,
            legend=dict(orientation="h", y=1.14),
            margin=dict(l=55, r=25, t=72, b=10),
        )
        distribution_figures.append(_figure)
    return (distribution_figures,)


@app.cell(hide_code=True)
def _(configuration_frames, distribution_figures, first_seed, mo, timeline_figures):
    _blocks = []
    for (_label, _particle), _timeline, _histogram in zip(
        configuration_frames, timeline_figures, distribution_figures, strict=True,
    ):
        _heading = _label.replace(" · ", ", ")
        _blocks.append(mo.vstack([
            mo.md(f"### {_heading}, {_particle}, seed pair {first_seed}"),
            mo.ui.plotly(_timeline),
            mo.ui.plotly(_histogram),
        ]))
    mo.vstack([
        mo.md("## Workload plots"),
        mo.md(
            "Each histogram starts with the 1st to 99th percentile of positive, finite "
            "times pooled across seeds. The two CPU panels for balanced argon and "
            "GPU-heavy argon (100 × 1 MeV) start with the 97th percentile at the upper end. "
            "We add 25% of that span to each side, stop at zero on the left, and use "
            "25 shared bins across the resulting range. The tables show how many events "
            "are plotted and report the mean, standard deviation, and CV before trimming. "
            "CV is 100 × standard deviation / mean. Click a seed in the "
            "legend to toggle it in both plots."
        ),
        *_blocks,
    ])
    return


if __name__ == "__main__":
    app.run()
