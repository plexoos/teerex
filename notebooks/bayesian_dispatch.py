# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.25.2", "numpy", "pandas", "scipy", "matplotlib"]
# ///
"""Train a phase model, adapt to an unknown machine, and inspect allocations."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Bayesian CPU/GPU dispatch")


@app.cell(hide_code=True)
def _():
    import hashlib
    import json
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Patch

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from teerex import adaptive as ad
    from teerex.replay import replay

    data_path = _root / "docs" / "study" / "events.csv.gz"
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10})
    phase_labels = ["CPU generation", "GPU host service", "Gate-held cleanup", "Remaining CPU"]
    regime_labels = {"measured": "Held-out measured replay", "transfer": "Synthetic new machine",
                     "stress": "Synthetic misspecification"}
    policy_colors = {"fixed": "#82919d", "nominal": "#3279a8", "mean": "#d38723",
                     "bayes": "#15846b", "oracle": "#8d69a7"}
    return (Patch, Path, ad, data_path, hashlib, json, mo, np, pd,
            phase_labels, plt, policy_colors, regime_labels, replay)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Learn the machine. Allocate the CPUs. Keep the GPU supplied.

    A simulation arrives with source settings we can inspect, but we do not yet
    know its phase runtimes on this machine. **How many CPU workers should we
    allocate, and how many events should we allow in flight?** We want short
    completion time without paying for CPUs that add little throughput.

    This notebook trains on the current Simphony `simg4ox` event measurements,
    maintains a Bayesian belief about a new execution environment, and uses that
    belief to choose allocations. It recomputes its fits and comparisons when run.
    No CUDA device is required for this experiment.

    **The scope:** a known family of event workloads, one GPU dispatch gate, and
    up to eight interchangeable CPU workers. We discover capacity before making
    decisions; machine *speed* and CPU contention are uncertain. Completely new
    physics configurations need calibration. Multiple GPUs and competing external
    jobs are extensions, not demonstrated capabilities.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Formulate the decision before fitting the model

    Event $i$ has four dependent service demands, in seconds:

    $$D_i=(D_{i,\mathrm{pre}},D_{i,\mathrm{gpu}},D_{i,\mathrm{lock}},D_{i,\mathrm{post}}).$$

    **CPU generation → GPU host service → CPU cleanup under the GPU gate → remaining CPU.**
    GPU service includes host-observed transfers and synchronization. The gate remains
    occupied until locked cleanup finishes, including any wait for a free CPU worker.
    The two cleanup phases use the same worker contiguously. CPU workers are released
    while events wait for GPU service; ready cleanup takes priority over new generation.

    At batch boundary $b$, select $a_b=(c_b,q_b)$: CPU workers and maximum unfinished
    events. CPU phases use one worker each, the GPU gate is exclusive, and phases are
    nonpreemptive. Source order is FIFO; GPU submission order is CPU-generation completion
    order. The current batch's labels are visible, but its durations and future batches
    are hidden. Drain this batch before reallocating resources for the next one.

    Let $T_b(a_b,D_b)$ be its simulated makespan. The user-selected objective is

    $$
    L_b(a_b,D_b)=T_b(a_b,D_b)+\lambda\underbrace{c_bT_b(a_b,D_b)}_{\text{allocated core-seconds}},
    \qquad \min_\pi\;\mathbb E^\pi\left[\sum_b L_b\right].
    $$

    Here $\lambda$ converts allocated core-seconds to the same cost units as elapsed
    seconds. At $\lambda=0.05$, reserving four CPUs for one second costs $1.2$ equivalent
    seconds. Reservation cost includes idle CPU capacity; it is **not CPU busy time**.
    One GPU is reserved throughout, so a GPU reservation price could be folded into
    the elapsed-time coefficient. There is no calibrated dollar or energy price here.

    The CPU/GPU split is a property of the physics and execution path, not a dial that
    moves arbitrary CPU work onto the GPU. We choose concurrency and admission capacity.
    A CPU-only fallback would require paired implementations and measurements that we
    do not currently have. This matches the separation of optical GPU work and other
    CPU physics described in the [Geant4 CaTS documentation](https://geant4.web.cern.ch/docs/advanced_examples_doc/example_cats.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A trainable source model and an uncertain machine

    For known source condition $k$, let $Y_i=\log(D_i/(1\,\mathrm{s}))$ be the four-vector
    of log demands. The source model is

    $$Y_{ir}=\mu_k+u_{kr}+\epsilon_{ir},\quad
      u_{kr}\sim\mathcal N(0,R_k),\quad\epsilon_{ir}\sim\mathcal N(0,\Sigma_k).$$

    Fit $\Sigma_k$ from within-run residuals and the covariance of run means from
    replications 0/1. Use a weak proper prior $\mu_k\sim\mathcal N(m_0,V_0)$ with
    $m_0=\log(0.01)\mathbf1$ and $V_0=9I$. With $J_k$ run means and regularized
    run-mean covariance $\widehat R_k$, the training update is

    $$V_k=(V_0^{-1}+J_k\widehat R_k^{-1})^{-1},\qquad
      m_k=V_k(V_0^{-1}m_0+J_k\widehat R_k^{-1}\overline Y_k).$$

    These are **empirical-Bayes** fits: covariance estimates are plugged in, not given
    their own posterior. We shrink each sample covariance 20% toward its diagonal,
    and add log-scale standard-deviation floors of 0.03 for run means and 0.02 for
    event residuals. These are explicit regularizers, not hardware facts. Two training
    runs per condition are very little evidence for between-run variation or tails.
    We retain the regularized covariance of run means as a proxy for a new run's
    persistent shift covariance; it also contains finite-run sampling noise. A new
    source-run prediction therefore uses covariance $\Sigma_k+V_k+\widehat R_k$.

    For a new execution environment, introduce persistent CPU and GPU log multipliers
    $z_C,z_G$ and uncertain CPU contention $\alpha$. Define

    $$h(c,\alpha)=1+\alpha(c-1),$$

    $$Y_i(a)=m_{k_i}+H_{k_i}\theta+\ell(c,\alpha)+\epsilon_i,$$

    where $\theta=(z_C,z_G,\delta_1,\ldots,\delta_K)$, each $\delta_k$ is a four-vector
    combining baseline uncertainty and a new run's persistent shift, with covariance
    $V_k+\widehat R_k$, and $H_k$ selects that offset,
    adds $z_C$ to the three CPU phases, and adds $z_G$ to GPU service. The vector
    $\ell$ adds $\log h(c,\alpha)$ to CPU phases and zero to GPU service.

    Start with independent $z_C,z_G\sim\mathcal N(0,0.7^2)$ and a uniform prior over
    $\alpha\in\{0,0.05,0.15,0.3,0.6\}$. Multiplier 2 means twice the source duration.
    The machine prior and contention law are **assumptions**: this dataset contains
    no hardware or worker-count sweep that could learn them. They describe effective
    execution corrections; machine speed and an unmodeled persistent workload shift
    cannot generally be identified separately.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bayesian updates and predictions

    Conditional on each contention hypothesis, the parameter posterior is Gaussian,
    $\theta\mid\alpha,H_b\sim\mathcal N(m_\alpha,V_\alpha)$, with mixture weight $w_\alpha$.
    A completed batch supplies a mean log residual $\bar y_k$ for each present condition.
    Its observation covariance is $Q_k=\widehat\Sigma_k/n_{\mathrm{eff}}$, using
    $n_{\mathrm{eff}}=\min(n_k,4)$. This cap tempers the likelihood for contiguous,
    potentially correlated events; it is not an estimated autocorrelation correction.

    $$S_\alpha=H_kV_\alpha H_k^T+Q_k,\quad
      e_\alpha=\bar y_k-\ell(c,\alpha)-H_km_\alpha,$$

    $$w_\alpha^+\propto w_\alpha\,\mathcal N(e_\alpha;0,S_\alpha),\quad
      K_\alpha=V_\alpha H_k^TS_\alpha^{-1},$$

    $$m_\alpha^+=m_\alpha+K_\alpha e_\alpha,\quad
      V_\alpha^+=V_\alpha-K_\alpha H_kV_\alpha.$$

    This is exact conditioning for the stated Gaussian-mixture likelihood, conditional
    on its plug-in covariances. The implementation uses the numerically safer Joseph
    covariance update. Since $h(1,\alpha)=1$, **observations exclusively at one worker
    cannot identify contention** under the independent prior. After higher-worker
    observations couple speed and contention, one-worker calibration can help separate
    them. At a fixed higher count, speed and contention remain confounded; allocation
    variation makes them identifiable within the model.

    A future event has a mixture of multivariate lognormal predictions. For phase $p$,

    $$\mathbb E[D_{ip}\mid H_b,a]=\sum_\alpha w_\alpha
      \exp\!\left(m_{k_i,p}+[H_{k_i}m_\alpha]_p+\ell_p(c,\alpha)
      +\tfrac12[H_{k_i}V_\alpha H_{k_i}^T+\Sigma_{k_i}]_{pp}\right)\mathrm{s}.$$

    This mean is finite. Parameter draws are shared by every event in a simulated
    batch, while correlated phase noise varies by event. Baseline uncertainty persists
    across batches. It is not repeatedly injected as independent event noise.

    **Information boundary:** input condition and batch size are known at allocation;
    generated photons, hits, phase times, and future condition labels are not supplied
    to the policy. Only the chosen allocation's completed phase times update it.
    Because decisions occur after a full drain, no unfinished duration is incorrectly
    treated as a completed observation. A continuous controller would need explicit
    survival likelihoods and a different state representation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Turn the posterior into an allocation

    Candidate allocations use $c\in\{1,2,4,8\}$ and $q\in\{c,2c\}$. Draw a contention
    hypothesis, one parameter vector, and event noise from the current belief. Replay
    each candidate on the **same sampled randomness**. Choose

    $$\widehat a_b=\arg\min_{a\in\mathcal A}\frac1S\sum_{s=1}^S
      L_b(a,D_b^{(s)}(a)).$$

    Different worker counts imply different demands when contention is present.
    Recompute the decision after observing the chosen batch. This is a **one-batch,
    posterior expected-cost controller**, not a globally optimal Bayes-adaptive policy.
    It does not explicitly buy information, forecast arrivals, or search event orderings.
    A fuller controller would add the expected value of the next belief state:

    $$V(b,H)=\min_a\mathbb E[L_b(a,D_b)+V(b+1,H\cup O_b)\mid H,a].$$

    Short lookahead and deliberate calibration probes are sensible next steps only if
    their expected future savings pay for their execution and computation costs.

    We also test **posterior mean demands**: insert the analytic mean above into the
    same scheduler once per candidate. Since scheduling is nonlinear,
    $\mathbb E[T(D)]\ne T(\mathbb E[D])$. Sampling can matter, but its extra effort must
    earn its place in an actual policy comparison.
    """)
    return


@app.cell(hide_code=True)
def _(ad, data_path, hashlib, json):
    events = ad.load_events(data_path)
    source_model = ad.fit_source(events)
    source_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    _manifest_path = data_path.parent / "manifest.json"
    legacy_manifest = json.loads(_manifest_path.read_text()) if _manifest_path.is_file() else {}
    return events, legacy_manifest, source_model, source_sha


@app.cell(hide_code=True)
def _(ad, events, pd):
    _rows = []
    for _condition, _group in events.groupby("condition_id", sort=True):
        _cpu = _group[[ad.PHASES[i] for i in (0, 2, 3)]].sum(axis=1)
        _rows.append({"Condition": _condition, "Events": len(_group), "Runs": _group.run_id.nunique(),
                      "Median CPU ms": 1000*_cpu.median(), "Median GPU ms": 1000*_group.gpu_runtime_s.median(),
                      "Median CPU/GPU": (_cpu/_group.gpu_runtime_s).median(),
                      "Photon CV %": 100*_group.num_photons.std()/_group.num_photons.mean(),
                      "CPU CV %": 100*_cpu.std()/_cpu.mean(),
                      "GPU CV %": 100*_group.gpu_runtime_s.std()/_group.gpu_runtime_s.mean()})
    audit_table = pd.DataFrame(_rows)
    split_table = events.groupby("replication").agg(events=("event_id", "size"), runs=("run_id", "nunique")).reset_index()
    split_table["Role"] = split_table.replication.map({0: "Training", 1: "Training", 2: "Validation / fixed-policy selection", 3: "Final prediction and replay"})
    return audit_table, split_table


@app.cell(hide_code=True)
def _(audit_table, events, legacy_manifest, mo, source_sha, split_table):
    _legacy_count = legacy_manifest.get("events")
    _legacy_label = f"{_legacy_count:,}" if isinstance(_legacy_count, int) else "unknown"
    _snapshot_note = (
        f"The current CSV has {len(events):,} events; the legacy manifest says "
        f"{_legacy_label}. This notebook reads the CSV and trains anew. "
        "It does not use the old saved fits, metrics, or policy choices."
        if legacy_manifest.get("events") != len(events) else
        "The CSV event count agrees with the saved manifest; this notebook still fits its own model."
    )
    mo.vstack([
        mo.md(f"""
        ## 3. What evidence do we actually have?

        **{len(events):,} events · {events.run_id.nunique()} processes ·
        {events.condition_id.nunique()} conditions · {events.gpu_uuid.nunique()} recorded GPU UUID ·
        {events.num_threads.nunique()} observed CPU-thread setting.**

        The current data contain electron workloads in pfRICH, dRICH, and liquid argon.
        Every condition has four runs with 500 retained events each. Warm-up events
        were already excluded; this notebook does no additional tail trimming.
        Timing and count summaries below are descriptive only; test rows do not fit the model.
        """),
        mo.callout(_snapshot_note, kind="warn"),
        mo.ui.table(audit_table.round(3), selection=None, show_column_summaries=False),
        mo.md("Low photon-count variation does not imply stable timings: CPU timing tails are already substantial in some conditions. Keep those measured tails. Extra synthetic variability should target missing workload and hardware regimes, not merely inflate every column."),
        mo.ui.table(split_table, selection=None, show_column_summaries=False),
        mo.md(f"Input archive SHA-256: `{source_sha}`. Whole runs stay in one split; no random event-level train/test split."),
    ])
    return


@app.cell(hide_code=True)
def _(audit_table, mo, plt):
    _fig, _ax = plt.subplots(figsize=(9, 4.2))
    for _i, _row in audit_table.iterrows():
        _ax.scatter(_row["Median CPU ms"], _row["Median GPU ms"], s=65)
        _ax.annotate(_row.Condition, (_row["Median CPU ms"], _row["Median GPU ms"]),
                     xytext=(6, 5 if _i % 2 else -12), textcoords="offset points", fontsize=8)
    _ax.plot([.1, 500], [.1, 500], "--", color="#aab3bb", linewidth=1, label="Equal CPU and GPU demand")
    _ax.set(xscale="log", yscale="log", xlabel="Median CPU demand / event (ms)",
            ylabel="Median GPU host service / event (ms)", title="Wide resource ratios, narrow source settings")
    _ax.legend(loc="upper left", fontsize=8)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("Below the diagonal, CPU work dominates; above it, GPU service dominates. The axes show service demands, not utilization or concurrent throughput.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Enough to start; insufficient to claim generalization

    | We can learn or check now | We cannot infer from these rows |
    |---|---|
    | Phase costs for seven observed source conditions | A continuous response surface for arbitrary energies, particles, multiplicities, or geometries |
    | Within-run phase correlation, timing tails, and run shifts | Worker-count contention, GPU concurrency, scheduling overhead, or memory limits |
    | Held-out prediction on another seed run of the same setup | Transfer to another CPU/GPU, driver, affinity, or clock regime |
    | Counterfactual ordering under an explicit resource simulator | A measured hardware speedup from an asynchronous executor |

    A **lookup likelihood plus Bayesian machine adaptation** is deliberately appropriate
    to this small condition library. A hierarchical regression over source and hardware
    features is the better eventual model, after collecting coverage across those features.
    A flexible neural surrogate would currently have little independent diversity to learn
    from and could disguise extrapolation as a confident prediction.

    Source settings are known at dispatch in this demonstration. If even those settings
    are unavailable, begin with a broad workload-mixture prior and a short measured pilot;
    no method can recover an arbitrary CPU/GPU payload from no information. If a new
    condition is absent from the training library, this implementation rejects it and
    asks for calibration rather than silently borrowing an unrelated condition.
    """)
    return


@app.cell(hide_code=True)
def _(ad, events, pd, source_model):
    diagnostics = pd.concat([ad.source_diagnostics(source_model, events, _r) for _r in (2, 3)], ignore_index=True)
    return (diagnostics,)


@app.cell(hide_code=True)
def _(ad, diagnostics, mo, np, phase_labels, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for _ax, _rep, _title in zip(_axes, (2, 3), ("Validation: replication 2", "Test: replication 3")):
        _table = diagnostics.loc[diagnostics.replication.eq(_rep)].pivot(index="condition", columns="phase", values="coverage90")[list(ad.PHASES)]
        _ax.imshow(_table.to_numpy(), vmin=.6, vmax=1., cmap="YlGnBu", aspect="auto")
        for _i in range(len(_table)):
            for _j in range(4):
                _value = _table.iloc[_i, _j]
                _ax.text(_j, _i, f"{100*_value:.1f}%", ha="center", va="center",
                         color="white" if _value > .85 else "#1a2b3a", fontsize=9)
        _ax.set_xticks(np.arange(4), phase_labels, rotation=25, ha="right")
        _ax.set_yticks(np.arange(len(_table)), _table.index, fontsize=8)
        _ax.set_title(_title)
    _fig.tight_layout()
    mo.vstack([
        mo.md("""
        ## 4. Train and check predictions before judging decisions

        **Each cell below is empirical coverage of a nominal 90% predictive interval.**
        These forecasts are made before seeing the held-out run and concern the source
        machine, so the broad new-machine prior is not added. Each cell uses all 500
        held-out events in its condition. The rows within a run are correlated: these
        are descriptive coverages, not 500 independent calibration trials.

        Calibration varies by condition and phase. A pooled average can conceal a badly
        modeled run. The model is not tuned after looking at replication 3. Validation
        selects only the fixed resource baseline; covariance floors, priors, seeds, and
        candidate actions are predeclared demonstration settings.
        """), _fig,
        mo.accordion({"Held-out errors and interval widths": mo.ui.table(diagnostics.round(4),
                     selection=None, page_size=14, show_column_summaries=False)}),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    experiment_controls = mo.md("""
    {cpu_price}

    {draws}

    {repeats}

    {prior_sd}
    """).batch(**{
        "cpu_price": mo.ui.slider(0., .3, step=.01, value=.05, label="CPU allocation price λ"),
        "draws": mo.ui.dropdown({"16 draws": 16, "24 draws": 24, "64 draws": 64}, value="24 draws", label="Posterior rollouts"),
        "repeats": mo.ui.dropdown({"8 seeds": 8, "16 seeds": 16}, value="8 seeds", label="Matched repeats"),
        "prior_sd": mo.ui.dropdown({"Narrow: 0.35": .35, "Default: 0.70": .7, "Broad: 1.20": 1.2}, value="Default: 0.70", label="Log machine prior SD"),
    }).form(submit_button_label="Re-run matched experiments", show_clear_button=False)
    mo.vstack([
        mo.md("""
        ## 5. Three worlds, identical policies and paired event blocks

        1. **Measured replay:** contiguous test-run blocks, unchanged service demands;
           no contention is imposed. Concurrency is still simulated, not measured.
        2. **Synthetic new machine:** the same blocks, multiplied by fixed CPU/GPU
           factors independently log-uniform from 0.5 to 2.5. Contention is uniform
           from 0.03 to 0.30, usually between model-grid points. These hidden parameters
           are sampled once per seed and shared by all policies.
        3. **Synthetic misspecification:** the new machine plus joint event lognormal
           variation (log SD 0.35), 8% fivefold bursts, and superlinear contention
           `1 + α(c − 1)^1.35`. The controller retains its original likelihood and linear
           contention law. These are stress assumptions, not inferred physics.

        Each run contains seven drained batches of 24 events: one batch per condition
        in a seeded shuffled order. A batch is homogeneous and takes a contiguous block
        from replication 3. This is a workload-switching benchmark; geometry/process
        startup and reconfiguration costs are excluded. Actual future batch labels are
        hidden from the controller. Policies see the same visible batches and hidden
        event difficulties, while their allocations and observations may differ.

        Defaults use seeds **110–117**, declared before inspection. Repeats vary sampled
        blocks/order and, in synthetic cases, the machine. They are not eight independently
        measured machines. The form reruns all policies, including validation selection,
        with the same controls; source-model fitting remains unchanged.
        """), experiment_controls,
    ])
    return (experiment_controls,)


@app.cell(hide_code=True)
def _(ad, events, experiment_controls, source_model):
    settings = experiment_controls.value or {"cpu_price": .05, "draws": 24, "repeats": 8, "prior_sd": .7}
    suite = ad.run_suite(events, source_model, seeds=tuple(range(110, 110+settings["repeats"])),
                        cpu_price=settings["cpu_price"], draws=settings["draws"], prior_sd=settings["prior_sd"])
    return settings, suite


@app.cell(hide_code=True)
def _(mo, suite):
    mo.md(f"""
    ### What is a fair comparison?

    | Policy | What it knows and changes |
    |---|---|
    | Fixed (validation) | One allocation, **{suite['fixed'].label}**, selected from replication 2 with the same objective |
    | Source model only | Chooses an allocation for each visible condition, using source predictions; assumes no machine shift or contention; never updates |
    | Adaptive mean demands | Updates the full Bayesian belief, then replays analytic mean demands once per candidate |
    | Bayesian expected cost | Uses the same Bayesian update, but averages cost across joint posterior draws |
    | Hindsight lower bound | Knows every realized duration and the true allocation-response law; chooses the cheapest candidate for each batch |

    The last policy is an evaluation bound, not a deployable competitor. It has the
    same finite action set and batch barriers. It is not a bound on every possible
    continuous or multi-GPU scheduler. All reported improvements include the early
    batches when the adaptive policies have not learned the machine yet.
    """)
    return


@app.cell(hide_code=True)
def _(ad, mo, plt, policy_colors, regime_labels, suite):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.3))
    _markers = {"fixed": "o", "nominal": "s", "mean": "^", "bayes": "D", "oracle": "x"}
    for _ax, (_regime, _title) in zip(_axes, regime_labels.items()):
        _part = suite["metrics"].loc[suite["metrics"].regime.eq(_regime)]
        _pivot = _part.pivot(index="seed", columns="policy", values="objective")
        for _p, _policy in enumerate(ad.POLICIES):
            _relative = _pivot[_policy]/_pivot.fixed
            _ax.scatter(_relative, _pivot.index + (_p-2)*.12, color=policy_colors[_policy],
                        marker=_markers[_policy], label=ad.POLICIES[_policy], s=35, alpha=.85)
        _ax.axvline(1., color="#98a4ae", linewidth=1)
        _ax.set(title=_title, xlabel="Objective / fixed-policy objective", ylabel="Seed")
        _ax.invert_yaxis()
    _axes[0].legend(fontsize=7, loc="best")
    _fig.tight_layout()
    mo.vstack([
        mo.md("""
        ## 6. Did the decisions help?

        **Lower cost is better. Compare policies within each seed.** A value below 1
        improves on the validation-selected fixed allocation. Vertical offsets separate
        symbols only. A better runtime fit is not itself evidence of a better allocation.
        """), _fig,
        mo.md("**Paired savings from Bayesian expected cost.** Positive favors Bayesian expected cost; negative favors the reference. Intervals are descriptive 95% Student-t intervals for mean seed-wise percentage savings. With few synthetic/bootstrapped scenarios, they do not establish a hardware-level guarantee."),
        mo.ui.table(suite["paired"].round(3), selection=None, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _(ad, mo, suite):
    _metrics = suite["metrics"].copy()
    _table = _metrics.groupby(["regime", "policy"], sort=False)[
        ["makespan", "core_seconds", "objective", "planning_seconds", "elapsed_with_planning"]].mean().reset_index()
    _table["policy"] = _table.policy.map(ad.POLICIES)
    _comparisons = suite["paired"].set_index(["regime", "reference"])
    _vs_mean = _comparisons.loc[("transfer", ad.POLICIES["mean"]), "saving_pct"]
    _vs_source = _comparisons.loc[("transfer", ad.POLICIES["nominal"]), "saving_pct"]
    mo.vstack([
        mo.md(f"""
        ### Separate learning, uncertainty integration, and computing cost

        On the synthetic transfer scenarios at these settings, posterior expected-cost
        planning changes the objective by **{_vs_source:+.2f}% savings versus the source
        model** and **{_vs_mean:+.2f}% savings versus adaptive mean demands**.
        Read these together with their paired intervals above. If the latter is zero
        or negative, the added rollouts have not justified themselves in this experiment.

        Bayesian *learning* is shared by both adaptive policies. A cheap posterior-mean
        controller may be the useful result even when full predictive integration is not.
        Improvements over a fixed allocation also reflect known workload ratios and
        should not all be credited to learning an unfamiliar machine.

        The table separates simulation seconds, reserved core-seconds, and local Python
        planning/update time. `elapsed_with_planning` adds measured controller time to
        makespan under a serial-controller assumption. Objective comparisons above exclude
        that overhead; CPU reservation during planning and executor overhead are not modeled.
        Small millisecond events may need much larger batches to amortize a controller.
        Planning time varies with the computer running this notebook.
        """),
        mo.ui.table(_table.round(4), selection=None, page_size=15, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, regime_labels, suite):
    inspect_regime = mo.ui.dropdown({label: key for key, label in regime_labels.items()},
                                   value="Synthetic new machine", label="World to inspect")
    inspect_seed = mo.ui.dropdown({str(seed): int(seed) for seed in sorted(suite["metrics"].seed.unique())},
                                 value=str(suite["metrics"].seed.min()), label="Seed")
    inspect_batch = mo.ui.slider(0, 6, value=2, step=1, label="Decision / batch index")
    mo.vstack([mo.md("## 7. Inspect one decision and its learning history"),
               mo.hstack([inspect_regime, inspect_seed, inspect_batch], justify="start")])
    return inspect_batch, inspect_regime, inspect_seed


@app.cell(hide_code=True)
def _(inspect_batch, inspect_regime, inspect_seed, suite):
    selected_world = suite["worlds"][(inspect_regime.value, inspect_seed.value)]
    selected_result = suite["results"][(inspect_regime.value, inspect_seed.value, "bayes")]
    selected_step = selected_result["history"][inspect_batch.value]
    return selected_result, selected_step, selected_world


@app.cell(hide_code=True)
def _(mo, plt, selected_step, settings):
    _scores = selected_step["scores"]
    _fig, _ax = plt.subplots(figsize=(10, 3.8))
    _ax.barh(_scores.action, _scores.cost, xerr=2*_scores.mc_se,
             color=["#15846b" if chosen else "#acbcc7" for chosen in _scores.selected], capsize=3)
    _ax.invert_yaxis()
    _ax.set(xlabel="Predicted cost (equivalent seconds)", title="Candidate costs before observing this batch")
    _fig.tight_layout()
    mo.vstack([
        mo.md(f"""
        **Batch {selected_step['batch']}: `{selected_step['condition']}`.**
        Chosen allocation: **{selected_step['action'].label}**.
        The belief has seen **{selected_step['before']['observed_batches']} completed batches**.
        Realized cost, revealed after selection: **{selected_step['objective']:.4f}** equivalent seconds.

        Bars average {settings['draws']} matched posterior futures. Whiskers show **two
        Monte Carlo standard errors**, not predictive intervals or uncertainty about the
        real world's optimum. The table also gives the paired standard error of each
        candidate's difference from the selected candidate. Near ties are not strong evidence.
        """), _fig,
        mo.ui.table(_scores.round(5), selection=None, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo, np, plt, selected_result, selected_world):
    _history = selected_result["history"]
    _states = [_history[0]["before"]] + [entry["after"] for entry in _history]
    _x = np.arange(len(_states))
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for _ax, _name in zip(_axes[:2], ("cpu", "gpu")):
        _ax.plot(_x, [state[f"{_name}_mean"] for state in _states], "o-", color="#15846b")
        _ax.fill_between(_x, [state[f"{_name}_lo"] for state in _states],
                         [state[f"{_name}_hi"] for state in _states], color="#15846b", alpha=.18)
        _ax.axhline(selected_world[_name], color="#aa6030", linestyle="--", label="Evaluator's hidden multiplier")
        _ax.set(title=f"{_name.upper()} multiplier at one worker", xlabel="Completed batches", ylabel="Duration multiplier")
        _ax.legend(fontsize=7)
    _alpha_weights = np.array([state["alpha_weights"] for state in _states])
    for _j, _alpha in enumerate((0., .05, .15, .3, .6)):
        _axes[2].plot(_x, _alpha_weights[:, _j], label=f"α={_alpha:g}")
    _axes[2].set(title="Contention hypothesis weights", xlabel="Completed batches", ylabel="Posterior mass", ylim=(0, 1))
    _axes[2].legend(fontsize=7)
    _fig.tight_layout()
    mo.vstack([_fig, mo.md(f"""
    Bands are 90% credible intervals for persistent multipliers, conditional on the
    model and prior. They are not next-event prediction intervals. Hidden synthetic
    contention is **{selected_world['alpha']:.3f}**; the policy never receives it.
    In the stress case the model's contention law is deliberately wrong, so a sharp
    posterior can still be wrong. Narrow uncertainty alone is not success.
    """)])
    return


@app.cell(hide_code=True)
def _(Patch, ad, mo, plt, replay, selected_step):
    _action = selected_step["action"]
    _rows = [dict(event_id=i, **dict(zip(ad.PHASES, values)))
             for i, values in enumerate(selected_step["observed"])]
    _timeline = replay(_rows, workers=_action.workers, max_inflight=_action.inflight, policy="async")
    _fig, _ax = plt.subplots(figsize=(12, max(2.7, .45*(_action.workers+1))))
    for _row in _timeline.itertuples():
        for _start, _end, _lane, _color in (
            (_row.cpu_pre_start_time_offset_s, _row.cpu_pre_end_time_offset_s, _row.cpu_pre_worker_id, "#3279a8"),
            (_row.cpu_post_start_time_offset_s, _row.cpu_post_end_time_offset_s, _row.cpu_post_worker_id, "#87bbd5"),
            (_row.gpu_start_time_offset_s, _row.gpu_end_time_offset_s, _action.workers, "#dc902c"),
        ):
            _ax.broken_barh([(_start, _end-_start)], (_lane-.3, .6), facecolors=_color)
        _ax.broken_barh([(_row.gpu_end_time_offset_s, _row.gpu_release_time_offset_s-_row.gpu_end_time_offset_s)],
                        (_action.workers-.3, .6), facecolors="#d2b6da", hatch="//")
    _ax.set_yticks(list(range(_action.workers+1)), [f"CPU {i}" for i in range(_action.workers)] + ["GPU gate"])
    _ax.set(xlabel="Seconds from this batch's start", title="Realized phase schedule for the selected allocation")
    _ax.legend(handles=[Patch(color="#3279a8", label="Generation"), Patch(color="#87bbd5", label="Cleanup"),
                        Patch(color="#dc902c", label="GPU host service"), Patch(facecolor="#d2b6da", hatch="//", label="Gate held after GPU service")],
               fontsize=8, loc="upper right")
    _fig.tight_layout()
    mo.vstack([_fig, mo.md("This is counterfactual replay of the realized demands, not a measured concurrent trace. The hatched interval includes any wait for a cleanup worker and gate-held cleanup. CPU idle capacity is still charged under the reservation objective.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Would more variable synthesized events help?

    **Yes, for testing the controller. More independent, structured measurements are
    needed for training a transferable physical runtime model.** Repeating narrow
    source settings many more times does not identify effects that never vary.

    | Experiment | What to vary or record | What it would resolve |
    |---|---|---|
    | Source-physics coverage | Log-spaced multiplicity and energy; particles, geometry, optical yield; repeated seeds | Workload effects and nonlinear scaling beyond seven condition labels |
    | Randomized allocation sweep | Repeat the same input families at 1/2/4/8 CPU workers and several WIP caps, randomizing order | CPU saturation, contention, gate stalls, and the causal response to our actions |
    | Hardware transfer | Several CPU/GPU combinations; model/UUID, affinity, clocks, memory, driver and build identifiers | Hierarchical machine effects and credible leave-one-machine-out evaluation |
    | Executor traces | Service vs queue wait, GPU gate release, actual CPU occupancy, transfer time, peak memory, batch size | Whether the replay semantics and feasible allocations match the real executor |
    | Mixed and bursty jobs | Arrival timestamps, job source metadata, workload transitions, startup costs, long tails | Continuous dispatch, admission control, queue latency, and lookahead value |

    A useful synthetic generator should preserve the **joint** CPU/GPU/cleanup demand,
    include bursts and rare expensive events, separate persistent machine effects from
    event noise, and vary contention independently of workload. Draw one hidden machine
    per experiment and give every policy matched event randomness. Do not simply add
    independent white noise to four timing columns and treat it as new training evidence.

    For real `simg4ox` collection, vary primary multiplicity and energy (and other supported
    physics settings) at generation time, retaining seeds and settings per event/run.
    First check that these controls produce the intended CPU/GPU-demand range. Changing
    photon yield may alter the physical workload, so record that change rather than
    pretending it represents another sample of the same physics configuration.

    A particularly useful next hardware study is **the same workload at multiple randomized
    CPU/WIP allocations on at least two machines**, followed by a machine held out entirely
    from fitting. The present dataset's biggest missing axis is response to an allocation,
    not its total number of events. Synthetic stress tests can prioritize that campaign;
    they cannot substitute for it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. What this iteration establishes

    There is a concrete loop here: **fit source demands → keep a machine belief → predict
    allocation consequences → choose → execute a batch → observe → update**. Its decisions,
    predictive costs, posterior changes, and paired realized outcomes are inspectable.
    Both adaptive policies are Bayesian; predictive sampling is an optional planning layer.

    The current data support an initial likelihood for the observed workload library and
    an honest held-out source check. They do not establish portable CPU/GPU scaling or an
    asynchronous hardware speedup. The demonstration can improve, tie, or regress, especially
    when its contention law or tail model is wrong. Prefer the cheaper controller when its
    measured decision quality is equivalent.

    Before production use, expand run and machine coverage, check phase prediction by
    condition, validate action-dependent runtimes in the executor, and measure decision
    overhead. Explore richer models only against strong simple baselines. This sequence
    follows the iterative model-checking approach described in
    [Bayesian Workflow](https://sites.stat.columbia.edu/gelman/research/unpublished/Bayesian_Workflow_article.pdf).

    **Reproduce locally:** from the Teerex checkout, run
    `uv run marimo edit notebooks/bayesian_dispatch.py`.
    The numerical engine is `teerex/adaptive.py`; it imports no external project and uses
    NumPy/SciPy, not a GPU backend. Existing study artifacts are not rewritten.
    """)
    return


if __name__ == "__main__":
    app.run()
