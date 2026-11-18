# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.25.2", "numpy", "pandas", "plotly"]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Geant4 CPU/GPU timelines: multiple GPUs")


@app.cell(hide_code=True)
def _():
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from teerex.notebook_data import resolve_data_file

    if sys.platform == "emscripten":
        browser_base = str(mo.notebook_location())
        data_root = Path("/tmp/teerex-timeline-inputs")
        _directory = Path("public")
    else:
        browser_base = None
        data_root = Path(__file__).resolve().parents[1]
        _directory = Path("data/multi_gpu")
    default_paths = {name: str(_directory / name) for name in (
        "manifest.json", "events.csv.gz", "run-summary.csv",
        "condition-summary.csv", "paired-summary.csv",
    )}
    return browser_base, data_root, default_paths, go, json, make_subplots, mo, pd, resolve_data_file


@app.cell(hide_code=True)
async def load_data(browser_base, condition_summary_path, csv_path, data_root, json, manifest_path,
                    mo, paired_summary_path, pd, resolve_data_file, run_summary_path):
    try:
        _manifest_file = await resolve_data_file(manifest_path.value, root=data_root, browser_base=browser_base)
        manifest = json.loads(_manifest_file.read_text())
        events = pd.read_csv(await resolve_data_file(csv_path.value, root=data_root, browser_base=browser_base))
        runs = pd.read_csv(await resolve_data_file(run_summary_path.value, root=data_root, browser_base=browser_base))
        conditions = pd.read_csv(await resolve_data_file(condition_summary_path.value, root=data_root, browser_base=browser_base))
        pairs = pd.read_csv(await resolve_data_file(paired_summary_path.value, root=data_root, browser_base=browser_base))
    except (OSError, ValueError, RuntimeError) as _error:
        mo.stop(True, mo.callout(
            f"Cannot load the multi-GPU evidence bundle: {_error}. "
            "Choose files from one compatible measurement bundle.", kind="warn",
        ))
    workload_labels = {
        "pfrich_n10": "pfRICH · 10 × 20 GeV/c",
        "pfrich_n100": "pfRICH · 100 × 20 GeV/c",
        "drich_n10": "dRICH · 10 × 20 GeV/c",
        "drich_n100": "dRICH · 100 × 20 GeV/c",
        "argon_balanced": "DUNE mock · 65,536 × 1 keV",
        "argon_gpu_n10": "DUNE mock · 10 × 10 MeV",
        "argon_gpu_n100": "DUNE mock · 100 × 1 MeV",
    }
    return conditions, events, manifest, pairs, runs, workload_labels


@app.cell(hide_code=True)
def _(events, runs, workload_labels):
    dual_runs = runs.loc[runs.gpu_count.eq(2)].copy()
    dual_events = events.merge(
        dual_runs[["run_id", "condition", "replication", "warmup_events"]],
        on="run_id", how="inner", validate="many_to_one",
    )
    _completed = dual_events.sort_values(["run_id", "end_offset_s", "event_id"])
    _retained = _completed.loc[
        _completed.groupby("run_id").cumcount().ge(_completed.warmup_events)
    ]
    configuration_frames = {
        condition: _retained.loc[_retained.condition.eq(condition)].copy()
        for condition in workload_labels
    }
    return configuration_frames, dual_events, dual_runs


@app.cell(hide_code=True)
def _(dual_events, dual_runs, mo):
    mo.md(f"""
    # Geant4 CPU/GPU timelines: multiple GPUs

    **2 GPUs · {len(dual_runs)} process runs · {len(dual_events):,} recorded events ·
    {int(dual_runs.retained_events.sum()):,} retained completions · {dual_runs.replication.nunique()} seed pairs.**

    Four Geant4 threads track the charged primaries in each run on npps0. Each
    producer submits one optical payload per event and can start its next event
    once the queue admits the payload. A dedicated thread in the same
    `simg4ox` process runs it on an available RTX 4090 and collects its hits.

    The pfRICH and dRICH workloads use 20 GeV/c electrons. The DUNE liquid-argon
    mock varies both the number of electrons and their kinetic energy.

    Blue segments show CPU preparation. Grey segments show
    producer handoff, including blocking while waiting for space in the GPU queue.
    Each GPU has one lane: orange shows host-observed service, including staging,
    transport and synchronization; purple shows CPU cleanup while that GPU remains
    reserved before taking another event. These timings come from the host clock,
    so the orange intervals include both host work and device waiting.
    Once handoff ends, the CPU producer can prepare its next event
    while the admitted payload waits for or runs on a GPU. That later payload time
    is not shown as CPU waiting. Handoff timing includes submission overhead, so
    the grey interval is not a measurement of pure CPU idle time.

    The configuration summary and histograms draw on all four two-GPU seed runs.
    Histograms exclude the first eight completed events of each run, as in the summary.
    The throughput comparison uses the matching one-GPU controls as its baseline.
    Each workload shows a two-GPU timeline for the first seed pair, opening
    on the first 10 event IDs. Drag to pan, use the toolbar to zoom, or adjust the
    range slider below the time axis to explore the rest of that run.
    """)
    return


@app.cell(hide_code=True)
def _(browser_base, default_paths, mo):
    csv_path = mo.ui.text(value=default_paths["events.csv.gz"], label="Event CSV", full_width=True)
    run_summary_path = mo.ui.text(value=default_paths["run-summary.csv"], label="Run summary CSV", full_width=True)
    condition_summary_path = mo.ui.text(value=default_paths["condition-summary.csv"], label="Condition summary CSV", full_width=True)
    paired_summary_path = mo.ui.text(value=default_paths["paired-summary.csv"], label="Paired summary CSV", full_width=True)
    manifest_path = mo.ui.text(value=default_paths["manifest.json"], label="Manifest JSON", full_width=True)
    mo.vstack([
        csv_path, run_summary_path, condition_summary_path, paired_summary_path, manifest_path,
        mo.md(("Paths are relative to this notebook's web address. " if browser_base else
               "Paths are relative to the project folder. ") + "Use files from the same measurement bundle."),
    ])
    return condition_summary_path, csv_path, manifest_path, paired_summary_path, run_summary_path


@app.cell(hide_code=True)
def _(go, manifest):
    physical_devices = {
        gpu["uuid"]: f"GPU {gpu['index']} · {gpu['uuid'].removeprefix('GPU-')[:8]}"
        for gpu in manifest["gpus"]
    }

    def make_timeline(frame, warmup_events):
        if frame.run_id.nunique() != 1:
            raise ValueError("Select exactly one process run for a timeline")
        frame = frame.sort_values("event_id")
        origin = frame.start_offset_s.min()
        end_ms = (frame.end_offset_s.max() - origin) * 1000
        initial_ms = (frame.head(10).end_offset_s.max() - origin) * 1000
        workers = [f"Geant4 {worker}" for worker in frame.worker_id]
        devices = [physical_devices[uuid] for uuid in frame.gpu_uuid]
        lanes = [f"Geant4 {worker}" for worker in sorted(frame.worker_id.unique())]
        lanes += sorted(set(devices))
        stages = [
            ("Geant4 preparation", workers, "start_offset_s", "gpu_submit_time_offset_s", "#1f77b4"),
            ("GPU queue wait", workers, "gpu_wait_start_time_offset_s", "gpu_wait_end_time_offset_s", "#aeb8c4"),
            ("GPU host service", devices, "gpu_service_start_offset_s", "gpu_service_end_offset_s", "#f28e2b"),
            ("Gate held for cleanup", devices, "gpu_service_end_offset_s", "gpu_release_time_offset_s", "#9467bd"),
        ]
        figure = go.Figure()
        for name, lane, begin, end, color in stages:
            start_ms = (frame[begin] - origin) * 1000
            stop_ms = (frame[end] - origin) * 1000
            figure.add_bar(
                name=name, legendgroup=name, orientation="h", y=lane, base=start_ms.tolist(),
                x=(stop_ms - start_ms).tolist(), width=0.55, marker_color=color,
                customdata=list(zip(
                    frame.event_id, frame.worker_id, frame.gpu_uuid, frame.num_gensteps,
                    frame.num_photons, frame.num_gpu_hits, frame.num_g4_hits, start_ms, stop_ms,
                )),
                hovertemplate=(
                    "Event %{customdata[0]} · Geant4 %{customdata[1]}<br>%{customdata[2]}<br>"
                    "%{customdata[3]:,} gensteps · %{customdata[4]:,} photons<br>"
                    "%{customdata[5]:,} GPU hits · %{customdata[6]:,} G4 hits<br>"
                    "%{customdata[7]:.3f}–%{customdata[8]:.3f} ms · duration %{x:.3f} ms"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        if warmup_events:
            warmup_end = frame.end_offset_s.nsmallest(warmup_events).max()
            figure.add_vline(x=(warmup_end - origin) * 1000, line_dash="dot", line_color="#484848")
        figure.update_layout(
            template="plotly_white", barmode="overlay", dragmode="pan", showlegend=True,
            height=345 + 40 * (len(lanes) - 2),
            legend=dict(orientation="h", y=1.08),
            margin=dict(l=55, r=25, t=50, b=45),
            xaxis=dict(
                title="Elapsed time (ms)", gridcolor="#e5e7eb",
                range=[0, initial_ms], minallowed=0, maxallowed=end_ms,
                rangeslider=dict(visible=True, range=[0, end_ms], thickness=0.18),
            ),
            yaxis=dict(categoryorder="array", categoryarray=lanes[::-1], fixedrange=True),
        )
        return figure

    return (make_timeline,)


@app.cell(hide_code=True)
def _(conditions, dual_runs, mo, workload_labels):
    _totals = dual_runs.groupby(["condition", "gpu_count"], as_index=False).raw_events.sum()
    _data = conditions.loc[conditions.gpu_count.eq(2)].merge(
        _totals, on=["condition", "gpu_count"], validate="one_to_one",
    )
    configuration_summary = _data.assign(
        workload=_data.condition.map(workload_labels),
        overlap=100 * _data.dual_service_overlap_fraction,
    )[[
        "workload", "gpu_count", "replications", "raw_events", "total_retained_events",
        "num_gensteps_mean", "num_photons_mean", "cpu_production_mean_ms", "gpu_service_mean_ms",
        "events_per_s", "overlap",
    ]].rename(columns={
        "workload": "Workload", "gpu_count": "GPUs", "replications": "Seed pairs",
        "raw_events": "Raw events", "total_retained_events": "Retained events",
        "num_gensteps_mean": "Gensteps/event", "num_photons_mean": "Photons/event",
        "cpu_production_mean_ms": "CPU prep (ms)",
        "gpu_service_mean_ms": "GPU service (ms)", "events_per_s": "Events/s",
        "overlap": "Both GPUs in service (%)",
    })
    mo.vstack([
        mo.md("""
        ## Configuration summary

        Each row includes all four seed pairs with two GPUs. For each run, throughput divides 64 completions
        by the time from the eighth to the final completion. Rates are geometric means across
        runs. Payload size and stage durations are means for the retained events. Simultaneous
        GPU service is the fraction of the completion window covered by both GPU service lanes.
        """),
        mo.ui.table(
            configuration_summary.round(2), selection=None, page_size=len(configuration_summary),
            show_column_summaries=False,
        ),
    ])
    return (configuration_summary,)


@app.cell(hide_code=True)
def _(configuration_frames, make_subplots):
    import numpy as np

    distribution_figures = []
    _colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    _dashes = ["solid", "dash", "dot", "dashdot"]
    for _condition, _frame in configuration_frames.items():
        _cpu_upper = 0.97 if _condition in ("argon_balanced", "argon_gpu_n100") else 0.99
        _figure = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "table"}, {"type": "table"}]],
            row_heights=[0.6, 0.4], vertical_spacing=0.17,
            subplot_titles=["CPU preparation + cleanup", "GPU host service"],
        )
        _cpu_ms = (
            _frame.gpu_submit_time_offset_s - _frame.start_offset_s
            + _frame.gpu_cleanup_end_offset_s - _frame.gpu_service_end_offset_s
        ) * 1000
        _gpu_ms = (_frame.gpu_service_end_offset_s - _frame.gpu_service_start_offset_s) * 1000
        _seeds = sorted(_frame.replication.unique())

        for _column, _times in enumerate((_cpu_ms, _gpu_ms), start=1):
            _valid = _times[np.isfinite(_times) & (_times > 0)]
            if _valid.empty:
                _low, _high = 0, 1
            else:
                _upper = _cpu_upper if _column == 1 else 0.99
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
            template="plotly_white", height=500,
            legend=dict(orientation="h", y=1.14),
            margin=dict(l=55, r=25, t=72, b=10),
        )
        distribution_figures.append(_figure)
    return (distribution_figures,)


@app.cell(hide_code=True)
def _(distribution_figures, dual_events, dual_runs, make_timeline, mo, workload_labels):
    _seed_sets = [
        set(dual_runs.loc[dual_runs.condition.eq(_condition), "replication"])
        for _condition in workload_labels
    ]
    _available_seeds = sorted(set.intersection(*_seed_sets))
    mo.stop(
        not _available_seeds,
        mo.callout("No seed pair is shared by all two-GPU configurations.", kind="warn"),
    )
    first_seed = _available_seeds[0]
    timeline_figures = []
    _blocks = []
    for (_condition, _label), _histogram in zip(
        workload_labels.items(), distribution_figures, strict=True,
    ):
        _match = dual_runs.loc[
            dual_runs.condition.eq(_condition)
            & dual_runs.replication.eq(first_seed)
        ]
        _run = _match.iloc[0]
        _frame = dual_events.loc[dual_events.run_id.eq(_match.run_id.item())]
        _figure = make_timeline(_frame, int(_run.warmup_events))
        timeline_figures.append(_figure)
        _heading = _label.replace(" · ", ", ")
        _blocks.append(mo.vstack([
            mo.md(f"### {_heading}, e-, 2 GPUs, seed pair {first_seed}"),
            mo.md(
                f"Run `{_run.run_id}` · {len(_frame)} events · "
                f"Geant4 / GPU seeds {int(_run.clhep_seed)} / {int(_run.gpu_seed)}. "
                f"Dotted line: **End of warm-up ({int(_run.warmup_events)} completed events)**."
            ),
            mo.ui.plotly(_figure),
            mo.ui.plotly(_histogram),
        ]))
    mo.vstack([
        mo.md("## Workload plots"),
        mo.md(
            "Each timeline comes from one process, with four Geant4 producer lanes above two physical GPU lanes. "
            "Hover over an interval to follow its event, payload size and physical GPU. "
            "The range slider spans the full run."
        ),
        mo.md(
            "Histograms pool the 64 retained events per run across all four two-GPU seed pairs. "
            "CPU time sums preparation and GPU-thread cleanup (the blue and purple intervals), excluding "
            "the grey handoff/queue wait. GPU time is host-observed service (the orange intervals). "
            "Each panel starts with the 1st to 99th percentile of positive, finite times pooled "
            "across seeds. As in the single-worker notebook, the CPU panels for balanced argon "
            "(65,536 × 1 keV) and GPU-heavy argon (100 × 1 MeV) use the 97th percentile "
            "at the upper end. Each panel adds 25% of that span to each side, stops at "
            "zero on the left, and uses 25 shared bins. Tables report shown/total counts and the mean, standard "
            "deviation, and CV before trimming. CV is 100 × standard deviation / mean. "
            "Click a seed in the legend to toggle it in both histogram panels."
        ),
        *_blocks,
    ])
    return first_seed, timeline_figures


@app.cell(hide_code=True)
def _(conditions, go, mo, pairs, workload_labels):
    speedups = go.Figure()
    _colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for _index, (_rep, _rows) in enumerate(pairs.groupby("replication")):
        speedups.add_scatter(
            name=f"Seed pair {int(_rep)}", mode="markers",
            x=_rows.speedup.tolist(), y=_rows.condition.map(workload_labels).tolist(),
            marker=dict(size=8, color=_colors[_index % len(_colors)]),
            customdata=list(zip(_rows.clhep_seed, _rows.gpu_seed)),
            hovertemplate="%{y}<br>Seeds %{customdata[0]} / %{customdata[1]}<br>%{x:.3f}×<extra>%{fullData.name}</extra>",
        )
    _dual = conditions.loc[conditions.gpu_count.eq(2)]
    for _column, _name, _symbol, _color in [
        ("paired_speedup", "Completion-window mean", "diamond", "#374151"),
        ("paired_whole_job_speedup", "Whole-job mean", "x", "#6b7280"),
    ]:
        speedups.add_scatter(
            name=_name, mode="markers", x=_dual[_column].tolist(),
            y=_dual.condition.map(workload_labels).tolist(),
            marker=dict(symbol=_symbol, size=11, color=_color),
            hovertemplate="%{y}<br>%{x:.3f}×<extra>%{fullData.name}</extra>",
        )
    speedups.add_vline(x=1, line_dash="dot", line_color="#484848")
    speedups.update_layout(
        template="plotly_white", height=465, margin=dict(l=55, r=25, t=72, b=45),
        legend=dict(orientation="h", y=1.14),
        xaxis=dict(title="Two-GPU throughput / matched one-GPU throughput", gridcolor="#e5e7eb"),
        yaxis=dict(
            categoryorder="array", categoryarray=list(workload_labels.values())[::-1],
            gridcolor="#e5e7eb",
        ),
    )
    mo.vstack([
        mo.md("""
        ## Throughput across seed pairs

        Each dot pairs one- and two-GPU runs with the same workload and seeds. The diamonds
        show geometric means of these ratios. Crosses use total process wall time, including
        initialization and final output. Values above one indicate a speedup with two GPUs.
        """),
        mo.ui.plotly(speedups),
    ])
    return (speedups,)


if __name__ == "__main__":
    app.run()
