# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "matplotlib", "numpy", "pandas"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Teerex: production ratio study v3")


@app.cell(hide_code=True)
async def _():
    import io
    import json
    import sys
    import zipfile
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch

    from teerex.analysis import plot_run_timeline, read_many_csv, timeline_summary
    from teerex.replay import replay

    if sys.platform == "emscripten":
        from pyodide.http import pyfetch

        evidence_dir = Path("/tmp/teerex-study-v3")
        evidence_dir.mkdir(exist_ok=True)
        _response = await pyfetch(str(mo.notebook_location() / "public" / "study.zip"))
        if not _response.ok:
            raise RuntimeError(f"Could not load study snapshot: HTTP {_response.status}")
        with zipfile.ZipFile(io.BytesIO(await _response.bytes())) as _archive:
            _archive.extractall(evidence_dir)
    else:
        evidence_dir = Path(__file__).resolve().parents[1] / "docs" / "study"
    phase_columns = ["cpu_pre_runtime_s", "gpu_runtime_s",
                     "cpu_post_locked_runtime_s", "cpu_post_unlocked_runtime_s"]
    phase_names = ["CPU generation", "GPU host service", "Gate-held cleanup", "Remaining CPU"]
    colors = ["#3478a3", "#e69f00", "#9774b8", "#69a88d"]
    cohort_names = {"drich": "dRICH", "pfrich": "pfRICH", "balanced": "Balanced argon",
                    "gpu_heavy_n10": "GPU-heavy argon: 10 × 10 MeV",
                    "gpu_heavy_n100": "GPU-heavy argon: 100 × 1 MeV"}
    return (Patch, Path, cohort_names, colors, evidence_dir, json, mo, np, pd,
            phase_columns, phase_names, plot_run_timeline, plt, read_many_csv, replay, timeline_summary)


@app.cell(hide_code=True)
def _(evidence_dir, json, np, pd, phase_columns):
    events = pd.read_csv(evidence_dir / "events.csv.gz")
    events["cohort"] = np.where(events["case"].eq("rich"), events.geometry_name, events["case"])
    events["cpu_demand_s"] = events[[phase_columns[i] for i in (0, 2, 3)]].sum(axis=1)
    events["gpu_share_pct"] = 100 * events.gpu_runtime_s / events[phase_columns].sum(axis=1)
    metrics = pd.read_csv(evidence_dir / "heldout-metrics.csv")
    condition_metrics = pd.read_csv(evidence_dir / "heldout-condition-metrics.csv")
    predictions = pd.read_csv(evidence_dir / "test-predictions.csv.gz")
    policy_results = pd.read_csv(evidence_dir / "policy-summary.csv")
    resources = pd.read_csv(evidence_dir / "resource-summary.csv")
    trace_checks = pd.read_csv(evidence_dir / "blocking-trace-validation.csv")
    frozen = json.loads((evidence_dir / "frozen-policies.json").read_text())
    manifest = json.loads((evidence_dir / "manifest.json").read_text())
    return (condition_metrics, events, frozen, manifest, metrics, policy_results,
            predictions, resources, trace_checks)


@app.cell(hide_code=True)
def _(events, manifest, mo):
    mo.md(f"""
    # Feeding a GPU from event-based CPU work

    **Teerex · production ratio study v3**

    How much CPU concurrency and work in flight can feed one GPU without
    excessive queues or completion latency? The new production sample contains
    **{len(events):,} retained events**, **{events.run_id.nunique()} processes**,
    and **{events.condition_id.nunique()} source conditions** across dRICH,
    pfRICH, balanced argon, and two GPU-heavy argon workloads.

    This analysis replaces the earlier dRICH-only fit and short argon pilots.
    Every timing summary, model check, and policy comparison below is rebuilt
    from `ratio-study-v3/all-production-events.csv`. The portable snapshot keeps
    the complete source CSV; its SHA256 begins `{manifest['source']['sha256'][:16]}`.

    **Read the results in three layers:** measured blocking phase timings,
    held-out duration predictions, and offline scheduling estimates. GPU host
    service includes transfers and synchronization. Its timing share does not
    measure device utilization, optical-kernel duty, or SM activity; this CSV
    contains no profiler evidence for those quantities.
    """)
    return


@app.cell(hide_code=True)
def _(cohort_names, events, mo, pd):
    sample_table = events.groupby("cohort", sort=False).agg(
        Events=("event_id", "size"), Processes=("run_id", "nunique"),
        Conditions=("condition_id", "nunique"),
    ).rename(index=cohort_names).reset_index(names="Workload")
    split_table = events.groupby(["split", "replication"], sort=False).agg(
        Events=("event_id", "size"), Processes=("run_id", "nunique")
    ).reset_index()
    mo.vstack([
        mo.md("""
        ## 1. What was measured?

        The RICH conditions use electrons or muons at **20 GeV/c**, with **10 or
        100 primaries/event**, in each of two geometries. The argon conditions
        use electrons at **1 keV kinetic energy × 65,536**, **10 MeV × 10**, or
        **1 MeV × 100**. Kinetic energy and momentum have different units and
        meanings; the RICH labels report momentum, the argon labels kinetic energy.

        Each condition has four independent replication processes and **500
        retained events/process**, IDs 5–504. The supplied export already marks
        every row as post-warm-up; no additional events or timing tails are
        removed here. All measured runs use blocking dispatch and one thread.
        Seeds, original offsets, geometry paths, run IDs, and condition IDs are
        preserved. Only one GPU UUID occurs in this export; its hardware model
        and CPU affinity are not recorded here.

        Replications **0/1 train** transforms, regressions, and condition-median
        baselines; **2 selects** scheduling policies; **3 evaluates** each frozen
        choice. Thousands of events do not replace independent process repeats:
        the model still has only two training replications per condition.
        """),
        mo.ui.table(sample_table, selection=None, show_column_summaries=False),
        mo.ui.table(split_table, selection=None, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _(events, pd):
    condition_order = list(events.condition_id.drop_duplicates())
    condition_summary = events.groupby("condition_id", sort=False).agg(
        Events=("event_id", "size"), Processes=("run_id", "nunique"),
        cpu_s=("cpu_demand_s", "median"), gpu_s=("gpu_runtime_s", "median"),
        photons=("num_photons", "median"), gpu_share=("gpu_share_pct", "median"),
    ).reset_index()
    condition_summary["CPU median (ms)"] = condition_summary.cpu_s * 1000
    condition_summary["GPU median (ms)"] = condition_summary.gpu_s * 1000
    condition_summary["GPU share median (%)"] = condition_summary.gpu_share
    condition_summary["Photons median"] = condition_summary.photons
    return condition_order, condition_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The workload ratio changes the scheduling problem

    Each event follows four dependent phases:

    ```text
    CPU generation → GPU host service → gate-held CPU cleanup → remaining CPU
                         └──────── exclusive GPU dispatch gate ────────┘
    ```

    CPU demand is generation plus both cleanup phases. The per-event GPU share is
    $d_{GPU}/(d_{pre}+d_{GPU}+d_{locked}+d_{unlocked})$.
    The median of event shares differs from the ratio of total or median times.
    All figures in this section describe the measured sample, including training,
    validation, and test replications; they do not choose a policy.
    """)
    return


@app.cell(hide_code=True)
def _(colors, condition_order, condition_summary, events, np, plt):
    ratio_figure, _axes = plt.subplots(1, 2, figsize=(12, 6.5), layout="constrained")
    _y = np.arange(len(condition_order))
    _axes[0].barh(_y - .18, condition_summary["CPU median (ms)"], .35,
                   label="CPU generation + cleanup", color=colors[0])
    _axes[0].barh(_y + .18, condition_summary["GPU median (ms)"], .35,
                   label="GPU host service", color=colors[1])
    _axes[0].set(xscale="log", xlabel="Median demand (ms, log scale)",
                 yticks=_y, yticklabels=condition_order)
    _axes[0].invert_yaxis()
    _axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.01))
    _shares = [events.loc[events.condition_id.eq(c), "gpu_share_pct"].to_numpy() for c in condition_order]
    _axes[1].boxplot(_shares, orientation="horizontal", positions=_y, widths=.6,
                     patch_artist=True, boxprops={"facecolor": "#dce8ef"},
                     flierprops={"markersize": 2, "alpha": .3})
    _axes[1].axvspan(40, 60, color=colors[3], alpha=.2, label="40–60% balance band")
    _axes[1].set(xlabel="Per-event GPU host-stage share (%)", xlim=(0, 101),
                 yticks=_y, yticklabels=[], ylim=(len(_y) - .5, -.5))
    _axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    for _axis in _axes:
        _axis.spines[["top", "right"]].set_visible(False)
        _axis.grid(axis="x", alpha=.2)
    ratio_figure
    return


@app.cell(hide_code=True)
def _(condition_summary, events, mo):
    _balanced = events.loc[events["case"].eq("balanced")]
    _inside = int(_balanced.gpu_share_pct.between(40, 60).sum())
    mo.vstack([
        mo.ui.table(condition_summary[["condition_id", "Events", "Processes", "Photons median",
            "CPU median (ms)", "GPU median (ms)", "GPU share median (%)"]].round(3),
            selection=None, show_column_summaries=False),
        mo.md(f"""
        Both dRICH and pfRICH are CPU-heavy in these conditions. pfRICH has a
        larger GPU share at multiplicity 10 because generation is shorter.
        The two GPU-heavy argon configurations spend roughly 94% of their stage
        time in GPU host service. Their nominal total kinetic energy is the
        same (100 MeV/event), but their track multiplicities and measured
        photon counts differ.

        The balanced argon sample has **{_inside:,}/{len(_balanced):,} events**
        inside the 40–60% band. All **{len(_balanced) - _inside} outside-band
        events remain included**. These controlled many-primary workloads
        characterize performance; stage timing does not establish physical
        accuracy or a realistic collision workload.
        """),
    ])
    return


@app.cell(hide_code=True)
def _(condition_order, mo):
    condition_control = mo.ui.dropdown(options=condition_order,
        value="argon|e-|T=0.001|n=65536", label="Measured condition")
    condition_control
    return (condition_control,)


@app.cell(hide_code=True)
def _(colors, condition_control, events, plt):
    _selected = events.loc[events.condition_id.eq(condition_control.value)]
    drift_figure, _axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, sharey=True,
                                       layout="constrained")
    for _axis, (_rep, _rows) in zip(_axes.flat, _selected.groupby("replication")):
        _rows = _rows.sort_values("event_id")
        for _column, _label, _color in [
            ("cpu_pre_runtime_s", "CPU generation", colors[0]),
            ("gpu_runtime_s", "GPU host service", colors[1]),
            ("cpu_post_runtime_s", "CPU cleanup", colors[2]),
        ]:
            _axis.plot(_rows.event_id, _rows[_column] * 1000, lw=.7, color=_color, label=_label)
        _axis.set(title=f"Replication {_rep}: {_rows.split.iloc[0]}", yscale="log",
                  xlabel="Original event ID", ylabel="Stage time (ms, log scale)")
        _axis.grid(alpha=.2)
        _axis.spines[["top", "right"]].set_visible(False)
    _axes[0, 0].legend(frameon=False, fontsize=8)
    drift_figure.suptitle(f"All 500 retained events/process · {condition_control.value}")
    drift_figure
    return


@app.cell(hide_code=True)
def _(cohort_names, mo):
    _intro = mo.md("""
    ## 3. Refit and check the duration models

    Five separate fits cover dRICH, pfRICH, balanced argon, and each GPU-heavy
    argon configuration. No liquid-argon timing is pooled into a RICH fit. Each
    fit uses the existing four-phase conjugate Bayesian log-duration baseline:
    log time is linear in transformed workload features, with a normal /
    inverse-gamma prior (`V0=10I`, `a0=2`, `b0=1`). The predictive law on log time
    is Student-t; exponentiating its location gives a duration median.

    CPU generation uses particle, geometry, momentum, and multiplicity. GPU and
    cleanup predictions additionally use photon/genstep counts available after
    generation. Scaling and condition medians use training rows only. In a
    single-condition argon fit, source features are constant; this checks repeated
    measurements of that condition and does not establish energy extrapolation.

    Compare Bayesian median, ordinary log regression, and training-condition
    median on **all held-out events**, not just the smaller scheduling streams.
    Predictive coverage and scheduling intervals remain conditional on the
    model and the small number of independent process replications.
    """)
    cohort_control = mo.ui.dropdown(options=list(cohort_names.values()), value="dRICH",
                                    label="Model and scheduling workload")
    mo.vstack([_intro, cohort_control])
    return (cohort_control,)


@app.cell(hide_code=True)
def _(cohort_control, cohort_names):
    selected_cohort = next(k for k, v in cohort_names.items() if v == cohort_control.value)
    return (selected_cohort,)


@app.cell(hide_code=True)
def _(colors, metrics, np, phase_columns, plt, selected_cohort):
    _test = metrics.loc[metrics.split.eq("test") & metrics.cohort.eq(selected_cohort)]
    model_figure, _axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for _offset, _method, _label, _color in zip(
        [-.25, 0, .25], ["bayesian", "ols_log", "condition_median"],
        ["Bayesian median", "OLS log regression", "Condition median"], [colors[0], colors[2], colors[3]],
    ):
        _values = _test.loc[_test.method.eq(_method)].set_index("phase").loc[phase_columns].mae_s
        _axes[0].bar(np.arange(4) + _offset, _values * 1000, .24, label=_label, color=_color)
    _axes[0].set(yscale="log", ylabel="Test MAE (ms, log scale)", xticks=np.arange(4),
                 xticklabels=["CPU pre", "GPU", "Locked post", "Remaining"])
    _axes[0].legend(frameon=False, fontsize=8)
    _bayes = _test.loc[_test.method.eq("bayesian")].set_index("phase")
    for _phase, _color in zip(phase_columns, colors):
        _axes[1].plot([50, 80, 95], _bayes.loc[_phase, ["coverage_50", "coverage_80", "coverage_95"]]
                      .to_numpy(float) * 100, "o-", label=_phase.replace("_runtime_s", ""), color=_color)
    _axes[1].plot([0, 100], [0, 100], "--", color=".5", lw=1)
    _axes[1].set(xlabel="Nominal predictive coverage (%)", ylabel="Observed test coverage (%)",
                 xlim=(45, 100), ylim=(0, 102))
    _axes[1].legend(frameon=False, fontsize=8)
    for _axis in _axes:
        _axis.spines[["top", "right"]].set_visible(False)
        _axis.grid(axis="y", alpha=.2)
    model_figure
    return


@app.cell(hide_code=True)
def _(condition_metrics, metrics, mo, selected_cohort):
    _rows = metrics.loc[metrics.cohort.eq(selected_cohort) & metrics.split.eq("test")]
    _best = _rows.loc[_rows.groupby("phase").mae_s.idxmin(), ["phase", "method", "mae_s"]]
    _groups = condition_metrics.loc[condition_metrics.cohort.eq(selected_cohort)
                                    & condition_metrics.split.eq("test")]
    _flagged = int((~_groups.coverage_95.between(.90, .99)).sum())
    mo.vstack([
        mo.md(f"**{_flagged}/{len(_groups)} condition/phase groups** fall outside the descriptive 90–99% band for nominal 95% coverage. This diagnostic does not certify calibration. Lowest test MAE by phase:"),
        mo.ui.table(_best, selection=None, show_column_summaries=False),
        mo.accordion({"All test errors and condition-level coverage": mo.vstack([
            mo.ui.table(_rows, selection=None, show_column_summaries=False),
            mo.ui.table(_groups, selection=None, page_size=8, show_column_summaries=False),
        ])}),
    ])
    return


@app.cell(hide_code=True)
def _(mo, phase_names):
    prediction_phase = mo.ui.dropdown(options=phase_names, value="GPU host service", label="Prediction phase")
    prediction_phase
    return (prediction_phase,)


@app.cell(hide_code=True)
def _(colors, np, phase_columns, phase_names, plt, prediction_phase, predictions, selected_cohort):
    _phase = phase_columns[phase_names.index(prediction_phase.value)]
    _rows = predictions.loc[predictions.cohort.eq(selected_cohort) & predictions.phase.eq(_phase)]
    prediction_figure, _axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for _axis, _column, _label, _color in zip(_axes, ["predicted_s", "median_s"],
        ["Bayesian median and 95% interval", "Training-condition median"], [colors[0], colors[3]]):
        if _column == "predicted_s":
            _axis.vlines(_rows.observed_s * 1000, _rows.low_95_s * 1000, _rows.high_95_s * 1000,
                          color=_color, alpha=.10, lw=.6)
        _axis.scatter(_rows.observed_s * 1000, _rows[_column] * 1000, s=5, alpha=.35, color=_color)
        _low = max(1e-7, min(_rows.observed_s.min(), _rows[_column].min()) * 1000 * .7)
        _high = max(_rows.observed_s.max(), _rows[_column].max()) * 1000 * 1.4
        _axis.plot([_low, _high], [_low, _high], "--", color=".5", lw=1)
        _axis.set(xscale="log", yscale="log", xlabel="Observed (ms)", ylabel="Predicted (ms)",
                  title=_label, xlim=(_low, _high))
        _axis.spines[["top", "right"]].set_visible(False)
    prediction_figure.suptitle(f"{prediction_phase.value} · {len(_rows):,} held-out events")
    prediction_figure
    return


@app.cell(hide_code=True)
def _(manifest, mo):
    mo.md(f"""
    ## 4. Select on validation, evaluate on test

    The scheduler releases a finite event stream at time zero. CPU generation
    and both dependent cleanup phases share a finite worker pool. One exclusive
    GPU gate stays held through locked cleanup, including any wait for a cleanup
    worker. Ready cleanup has priority; running CPU work cannot be preempted.

    - **Blocking:** one event completes before another is admitted.
    - **Synchronous:** each worker retains its event slot while waiting for the GPU.
    - **Asynchronous:** workers return to the pool after generation and are
      reacquired for cleanup; the in-flight limit bounds unfinished events.

    Each workload independently compares **85 configurations**: blocking, plus
    synchronous/async with 1/2/4/8/16/20 workers and 1/2/4/8/16/32/64 in flight.
    Candidates share **{manifest['draws']} posterior predictive draws** on a
    **256-event validation stream**, weighted equally across that workload's
    conditions. Choose least workers, then least in flight, within 5% of the
    best validation median makespan. Test outcomes never choose the policy.

    The frozen choice and blocking control are evaluated on a separate,
    equally weighted 256-event test stream. The figures report simulated
    speedups relative to simulated blocking on the same draws. These are
    finite-stream estimates, not measured executor speedups. CPU batches are
    one event; no cross-event photon aggregation is modeled.
    """)
    return


@app.cell(hide_code=True)
def _(frozen, np, plt, policy_results, selected_cohort):
    _rows = policy_results.loc[policy_results.cohort.eq(selected_cohort) & policy_results.split.eq("validation")]
    grid_figure, _axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    _maximum = max(1.01, _rows.simulated_speedup_median.max())
    _chosen = frozen[selected_cohort]["configuration"]
    for _axis, _policy in zip(_axes, ["synchronous", "async"]):
        _grid = _rows.loc[_rows.policy.eq(_policy)].pivot(index="workers", columns="max_inflight",
                                                        values="simulated_speedup_median")
        _image = _axis.imshow(_grid, origin="lower", aspect="auto", cmap="viridis", vmin=1, vmax=_maximum)
        _axis.set(xticks=np.arange(len(_grid.columns)), xticklabels=_grid.columns,
                  yticks=np.arange(len(_grid.index)), yticklabels=_grid.index,
                  xlabel="Maximum events in flight", ylabel="CPU workers", title=_policy.capitalize())
        for _i, _worker in enumerate(_grid.index):
            for _j, _cap in enumerate(_grid.columns):
                _value = _grid.loc[_worker, _cap]
                _axis.text(_j, _i, f"{_value:.1f}", ha="center", va="center", fontsize=8,
                           color="white" if _value < .65 * _maximum else "black")
        if _policy == _chosen["policy"]:
            _axis.scatter([list(_grid.columns).index(_chosen["max_inflight"])],
                          [list(_grid.index).index(_chosen["workers"])], s=620, marker="s",
                          facecolors="none", edgecolors="#e69f00", linewidths=2)
    grid_figure.colorbar(_image, ax=_axes, label="Median simulated validation speedup", shrink=.8)
    grid_figure
    return


@app.cell(hide_code=True)
def _(cohort_names, frozen, pd, policy_results, resources):
    _records = []
    for _cohort, _policy in frozen.items():
        _config = _policy["configuration"]
        _row = policy_results.loc[policy_results.cohort.eq(_cohort) & policy_results.split.eq("test")
            & policy_results.configuration.eq(_config["configuration"])
            & policy_results["mode"].eq("independent") & policy_results.cpu_slowdown.eq(1)].iloc[0]
        _resource = resources.loc[resources.cohort.eq(_cohort)
                                 & resources.configuration.eq(_config["configuration"])].iloc[0]
        _records.append({"Workload": cohort_names[_cohort], "Policy": _config["policy"],
            "Workers": _config["workers"], "In flight": _config["max_inflight"],
            "Predicted makespan (s)": _row.makespan_median_s,
            "Predicted speedup": _row.simulated_speedup_median,
            "Observed-demand replay (s)": _resource.makespan_s,
            "GPU service / span (%)": 100 * _resource.gpu_service_s / _resource.makespan_s,
            "GPU gate / span (%)": 100 * _resource.gpu_gate_occupied_s / _resource.makespan_s})
    frozen_summary = pd.DataFrame(_records)
    return (frozen_summary,)


@app.cell(hide_code=True)
def _(frozen_summary, mo):
    mo.vstack([
        mo.md("### Frozen policies on separate test streams"),
        mo.ui.table(frozen_summary.round(3), selection=None, show_column_summaries=False),
        mo.md("The last three columns replay fixed **observed test demands** under the frozen choice. They remain modeled schedules. GPU gate occupancy includes cleanup and cleanup-worker waits; it can greatly exceed GPU host-service occupancy."),
    ])
    return


@app.cell(hide_code=True)
def _(colors, frozen, mo, np, plt, policy_results, selected_cohort):
    _config = frozen[selected_cohort]["configuration"]
    _rows = policy_results.loc[policy_results.cohort.eq(selected_cohort) & policy_results.split.eq("test")
        & policy_results.configuration.eq(_config["configuration"])].sort_values(["mode", "cpu_slowdown"])
    _labels = ["Observed demands" if r.mode == "observed" else "Joint residual blocks" if r.mode == "joint_block"
               else "Nominal posterior" if r.cpu_slowdown == 1 else f"CPU demand ×{r.cpu_slowdown:g}"
               for r in _rows.itertuples()]
    sensitivity_figure, _axis = plt.subplots(figsize=(11, 3.8), layout="constrained")
    _median = _rows.makespan_median_s.to_numpy()
    _axis.bar(np.arange(len(_rows)), _median, color=colors[0], width=.6)
    _axis.errorbar(np.arange(len(_rows)), _median,
        yerr=np.vstack([_median - _rows.makespan_low_s, _rows.makespan_high_s - _median]),
        fmt="none", color=".2", capsize=4)
    _axis.set(xticks=np.arange(len(_rows)), xticklabels=_labels, ylabel="Test makespan (s)",
              title=f"{selected_cohort}: {_config['configuration']} · fixed choice, changed demands")
    _axis.spines[["top", "right"]].set_visible(False)
    _axis.grid(axis="y", alpha=.2)
    mo.vstack([sensitivity_figure, mo.md("Whiskers span the 2.5–97.5% predictive quantiles. CPU ×1.5/×2 are uncalibrated contention scenarios applied when CPU workers exceed one; one-worker choices stay unchanged. Joint blocks resample up to eight adjacent training residual tuples and share coefficient draws; they probe dependence without correcting conditional bias.")])
    return


@app.cell(hide_code=True)
def _(events, frozen):
    replay_sources = {condition: rows.sort_values("event_id") for condition, rows in
                      events.loc[events.split.eq("test")].groupby("condition_id", sort=False)}
    _indexed = events.set_index(["run_id", "event_id"])
    for _cohort, _policy in frozen.items():
        replay_sources[f"{_cohort}: equal-condition test mixture"] = _indexed.loc[
            [tuple(key) for key in _policy["streams"]["test"]]].reset_index()
    return (replay_sources,)


@app.cell(hide_code=True)
def _(mo, replay_sources):
    _intro = mo.md("""
    ## 5. Explore workers, queues, and dependent cleanup

    This live replay uses fixed measured **test** demands. Controls change the
    schedule without refitting or changing the frozen policies above. All
    events are released at time zero; completion latency includes admission
    backlog. Admission-to-completion latency is shown separately. Queued
    photons are a work proxy without a measured VRAM conversion.
    """)
    workload_control = mo.ui.dropdown(options=list(replay_sources), value="argon|e-|T=0.001|n=65536",
                                      label="Test demand source")
    event_control = mo.ui.slider(start=4, stop=256, step=4, value=32, label="Events", show_value=True)
    worker_control = mo.ui.slider(start=1, stop=20, value=4, label="CPU workers", show_value=True)
    inflight_control = mo.ui.slider(start=1, stop=64, value=8, label="Maximum in flight", show_value=True)
    slowdown_control = mo.ui.dropdown(options=[1.0, 1.5, 2.0], value=1.0, label="Parallel CPU multiplier")
    batch_control = mo.ui.dropdown(options=[1, 2, 4, 8], value=1, label="Synchronous worker batch")
    lane_policy = mo.ui.dropdown(options=["blocking", "synchronous", "async"], value="async", label="Timeline policy")
    mo.vstack([_intro, workload_control, event_control, mo.hstack([worker_control, inflight_control]),
               mo.hstack([slowdown_control, batch_control]), lane_policy])
    return (batch_control, event_control, inflight_control, lane_policy, slowdown_control,
            worker_control, workload_control)


@app.cell(hide_code=True)
def _(batch_control, event_control, inflight_control, pd, phase_columns, replay,
      replay_sources, slowdown_control, worker_control, workload_control):
    replay_sample = replay_sources[workload_control.value].head(event_control.value).copy()
    replay_timelines = {}
    _summary = []
    for _policy in ["blocking", "synchronous", "async"]:
        _workers = 1 if _policy == "blocking" else worker_control.value
        _cap = 1 if _policy == "blocking" else inflight_control.value
        _factor = slowdown_control.value if _workers > 1 else 1.0
        _events = [{"event_id": i, "num_photons": int(row["num_photons"]),
                    **{phase: float(row[phase]) * (1 if phase == "gpu_runtime_s" else _factor)
                       for phase in phase_columns}} for i, row in enumerate(replay_sample.to_dict("records"))]
        _timeline = replay(_events, workers=_workers, max_inflight=_cap, policy=_policy,
                           event_batch_size=batch_control.value if _policy == "synchronous" else 1)
        replay_timelines[_policy] = _timeline
        _span = float(_timeline.end_time_offset_s.max())
        _summary.append({"Policy": _policy, "Events": len(_timeline), "Makespan (s)": _span,
            "P95 completion (s)": _timeline.latency_s.quantile(.95),
            "P95 admitted latency (s)": _timeline.service_latency_s.quantile(.95),
            "CPU capacity busy (%)": 100 * _timeline.cpu_runtime_s.sum() / (_workers * _span),
            "GPU service / span (%)": 100 * _timeline.gpu_runtime_s.sum() / _span,
            "GPU gate / span (%)": 100 * _timeline.gpu_gate_runtime_s.sum() / _span})
    replay_summary = pd.DataFrame(_summary)
    replay_summary["Speedup vs blocking"] = replay_summary["Makespan (s)"].iloc[0] / replay_summary["Makespan (s)"]
    return replay_sample, replay_summary, replay_timelines


@app.cell(hide_code=True)
def _(mo, replay_summary):
    mo.ui.table(replay_summary.round(3), selection=None, show_column_summaries=False)
    return


@app.cell(hide_code=True)
def _(Patch, colors, plt):
    def worker_lanes(frame, *, title, measured=False):
        workers = sorted(set(frame.cpu_pre_worker_id) | set(frame.cpu_post_worker_id))
        origin = float(frame.cpu_pre_start_time_offset_s.min()) if measured else 0.0
        figure, axis = plt.subplots(figsize=(11, max(3.2, 1.8 + .32 * len(workers))))
        cpu_y = {worker: index + 1 for index, worker in enumerate(workers)}
        for row in frame.itertuples():
            for worker, start, end, color in [
                (row.cpu_pre_worker_id, row.cpu_pre_start_time_offset_s, row.cpu_pre_end_time_offset_s, colors[0]),
                (row.cpu_post_worker_id, row.cpu_post_start_time_offset_s, row.cpu_post_end_time_offset_s, colors[2]),
            ]:
                if end > start:
                    axis.broken_barh([(start - origin, end - start)], (cpu_y[worker] - .25, .5), facecolors=color)
            axis.broken_barh([(row.gpu_start_time_offset_s - origin,
                              row.gpu_release_time_offset_s - row.gpu_start_time_offset_s)],
                             (-.32, .64), facecolors="#d7d7d7")
            axis.broken_barh([(row.gpu_start_time_offset_s - origin,
                              row.gpu_end_time_offset_s - row.gpu_start_time_offset_s)],
                             (-.2, .4), facecolors=colors[1])
        axis.set(yticks=[0] + list(cpu_y.values()), yticklabels=["GPU / gate"] + [f"CPU {w}" for w in workers],
                 xlabel="Seconds since first selected event" if measured else "Seconds since common release",
                 title=title, xlim=(0, float(frame.cpu_post_end_time_offset_s.max()) - origin))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", alpha=.2)
        axis.legend(handles=[Patch(color=colors[0], label="CPU generation"), Patch(color=colors[2], label="CPU cleanup"),
                             Patch(color=colors[1], label="GPU host service"), Patch(color="#d7d7d7", label="Gate occupied")],
                    loc="upper left", bbox_to_anchor=(0, -.2), ncol=4, frameon=False, fontsize=8)
        figure.tight_layout()
        return figure
    return (worker_lanes,)


@app.cell(hide_code=True)
def _(lane_policy, replay_timelines, worker_lanes):
    worker_lanes(replay_timelines[lane_policy.value], title=f"Offline {lane_policy.value} replay")
    return


@app.cell(hide_code=True)
def _(mo, replay_sample, worker_lanes):
    if replay_sample.run_id.nunique() == 1:
        _measured = replay_sample.head(16).copy()
        _measured["cpu_pre_worker_id"] = 0
        _measured["cpu_post_worker_id"] = 0
        mo.output.replace(mo.vstack([
            mo.md("### Original measured blocking timeline\nThe first 16 selected events retain their original inter-event gaps. Only the plotting origin is subtracted."),
            worker_lanes(_measured, measured=True, title="Measured baseline: one original process"),
        ]))
    else:
        mo.output.replace(mo.callout("This mixture combines independent process clocks. Select one condition to inspect an original measured timeline.", kind="info"))
    return


@app.cell(hide_code=True)
def _(manifest, mo, trace_checks):
    mo.md(f"""
    ## 6. What this establishes, and what comes next

    The longer production sample now permits held-out checks and independent
    policy selection in all five workload groups. A CPU-heavy condition can
    benefit from many producers; GPU-heavy workloads have less serial CPU work
    to hide. Cleanup and GPU-gate occupancy remain separate from useful transport
    time, so a busy gate alone does not establish a fully fed GPU.

    Next, validate frozen choices in an actual charged asynchronous executor,
    measure concurrent service-time changes, and improve condition-level model
    calibration. Cleanup reservations, larger worker batches, cross-event GPU
    aggregation, memory limits, and latency budgets need their own experiments.
    This export contains no asynchronous hardware measurements or device traces.

    ## 7. Provenance and reproduction

    The snapshot contains the complete original CSV, model parameters, training
    residual diagnostics, held-out predictions, condition metrics, frozen
    policies with exact stream identities, and trace-accounting checks. The
    maximum reconstructed blocking boundary discrepancy is
    **{trace_checks.max_event_boundary_error_s.max() * 1e9:.2f} ns** across
    **{len(trace_checks)} processes**; submission overhead and inter-event gaps
    are accounted for separately from phase service.

    Source SHA256: `{manifest['source']['sha256']}`

    Rebuild from the supplied export, then verify all bundled file hashes:

    ```bash
    PYTHONPATH=. uv run python scripts/refresh_ratio_study.py \\
      ../teerex-study/ratio-study-v3/all-production-events.csv
    PYTHONPATH=. uv run python scripts/refresh_ratio_study.py --verify
    uv run marimo edit notebooks/simload.py
    ```

    `docs/study/README.md` describes the artifact files and exact split rules.
    `docs/bayesian-dispatch.md` defines the model and scheduler.
    `docs/simload.md` documents the original synthetic workload and its CUDA
    requirements. This analysis notebook needs neither CUDA nor the raw archive.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    extra_paths = mo.ui.text_area(value="", label="Optional: one timeline CSV path per line", rows=3, full_width=True)
    mo.accordion({"Inspect another timeline": mo.vstack([
        mo.md("Relative paths use the process working directory. Use unique scenario labels for independent clocks. These inputs affect only this viewer."), extra_paths,
    ])})
    return (extra_paths,)


@app.cell(hide_code=True)
def _(Path, extra_paths, mo, read_many_csv):
    _paths = [Path(line.strip()).expanduser() for line in extra_paths.value.splitlines() if line.strip()]
    mo.stop(not _paths)
    _missing = [str(path) for path in _paths if not path.is_file()]
    mo.stop(bool(_missing), mo.callout("Missing CSV files: " + ", ".join(_missing), kind="warn"))
    try:
        extra_events = read_many_csv(_paths)
        if not {"scenario", "event_id", "cpu_start_time_offset_s", "gpu_start_time_offset_s"}.issubset(extra_events):
            raise ValueError("The CSV does not contain the shared timeline columns.")
    except (ValueError, OSError) as _error:
        mo.stop(True, mo.callout(str(_error), kind="warn"))
    return (extra_events,)


@app.cell(hide_code=True)
def _(extra_events, mo):
    _labels = sorted(extra_events.scenario.dropna().unique())
    extra_scenarios = mo.ui.multiselect(options=_labels, value=_labels[:3], label="Timeline scenarios")
    extra_shared_scale = mo.ui.checkbox(value=False, label="Share time scale")
    mo.vstack([extra_scenarios, extra_shared_scale])
    return extra_scenarios, extra_shared_scale


@app.cell(hide_code=True)
def _(extra_events, extra_scenarios, extra_shared_scale, mo, plot_run_timeline, timeline_summary):
    mo.stop(not extra_scenarios.value)
    _selected = extra_events.loc[extra_events.scenario.isin(extra_scenarios.value)].copy()
    _figure, _, _timeline = plot_run_timeline(_selected, extra_scenarios.value, sharex=extra_shared_scale.value)
    mo.vstack([mo.ui.table(timeline_summary(_selected), selection=None, show_column_summaries=False), _figure,
               mo.ui.table(_timeline, selection=None, page_size=10, show_column_summaries=False)])
    return


if __name__ == "__main__":
    app.run()
