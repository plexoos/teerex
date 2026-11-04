# Bayesian durations and CPU/GPU dispatch

Teerex separates two questions: how long an event's phases might take, and how a
resource-constrained scheduler would arrange those phases. `teerex.bayes` supplies
a small conjugate regression; `teerex.study` builds duration models and compares
policies; `teerex.replay` constructs deterministic schedules from supplied demands.
The [study report](study.md) gives the measured context and current findings.

## Four dependent phases

A measured charged event follows this order:

```text
CPU generation ──> GPU host service ──> CPU cleanup under gate ──> remaining CPU work
      pre                 gpu                    locked                  unlocked
                          |<──── exclusive GPU dispatch gate ────>|
```

The corresponding CSV demands are `cpu_pre_runtime_s`, `gpu_runtime_s`,
`cpu_post_locked_runtime_s`, and `cpu_post_unlocked_runtime_s`. GPU host service
includes transfers and synchronization. The exclusive dispatch gate stays held
through collection/reset after device completion. Cleanup depends on the GPU
result. Synthetic Simload's independent CPU-post work has different semantics.

In an asynchronous replay, a completed GPU operation may wait for a CPU worker
before cleanup starts. That wait also holds the gate. Thus GPU service time,
gate service demand (GPU plus locked cleanup), and occupied gate time are three
different quantities.

## The duration model

For each phase and work class, the implementation fits

$$
y_i=\log\!\bigl(\max(d_i,10^{-9}\,\mathrm{s})/(1\,\mathrm{s})\bigr),\qquad
 y_i=x_i^\top\beta+\epsilon_i,\qquad
 \epsilon_i\mid\sigma^2\sim\mathcal N(0,\sigma^2).
$$

The proper normal/inverse-gamma prior is

$$
\beta\mid\sigma^2\sim\mathcal N(m_0,\sigma^2V_0),\qquad
\sigma^2\sim\operatorname{InvGamma}(a_0,b_0),
$$

with defaults $m_0=0$, $V_0=10I$, $a_0=2$, $b_0=1$. The inverse-gamma density uses
$\exp(-b_0/\sigma^2)$. This broad prior has a log-location of zero, corresponding
to one second; it is a baseline to check against each timing regime.

For design matrix $X$ and log durations $y$, the conjugate update is

$$
V_n=(V_0^{-1}+X^\top X)^{-1},\quad
m_n=V_n(V_0^{-1}m_0+X^\top y),\quad a_n=a_0+n/2,
$$

$$
b_n=b_0+\tfrac12\left[\lVert y-Xm_n\rVert^2+
(m_n-m_0)^\top V_0^{-1}(m_n-m_0)\right].
$$

The code uses QR on an augmented least-squares system to avoid forming the normal
equations. A new log duration has a Student-t predictive distribution with
$2a_n$ degrees of freedom, location $x^\top m_n$, and scale
$\sqrt{(b_n/a_n)(1+x^\top V_nx)}$. This scale is not its standard deviation.

Exponentiation preserves quantiles: the duration median is
$\exp(x^\top m_n)$ seconds, and prediction intervals exponentiate the log-scale
Student-t bounds. The implementation does not use an exponential Student-t mean;
that expectation is not finite. Scheduling uses finite posterior samples.

### Features and information available to the scheduler

| Phase | Allowed predictors |
| --- | --- |
| CPU generation | Particle and geometry categories; `log1p(momentum_gev_c)` and `log1p(multiplicity)` |
| GPU and both cleanup phases | Those predictors plus `log1p(num_photons)` and `log1p(num_gensteps)` |

The transform supplies an intercept, standardizes numeric features using training
means/scales, and retains category indicators. A proper prior accommodates
collinearity. Unknown categories are rejected. CPU-pre cannot use generated counts;
they become available at GPU submission. Hits, future durations, and clock values
never enter the feature matrix.

CPU-pre has one class. The other phases split zero-photon and positive-work
rows; unseen classes require more training data. Zero-work events remain in the
study and recorded zero durations use the timer floor on the log scale.

### Coherent draws and diagnostics

`BayesianLinearRegression.sample_predictive(X, n_draws, rng)` draws one variance
and coefficient vector for the whole stream in each draw, then conditionally
independent observation noise. It returns `(draw, event)` in the supplied target
units. `DurationModels.sample()` exponentiates the four log-duration models and
returns `(draw, event, phase)`.

The primary model treats phases independently. The `joint_block` sensitivity
resamples joint training residual tuples in blocks of up to eight consecutive
rows within a run/work class, alongside parameter draws. It is a sensitivity
scenario, not a calibrated correlated posterior. The study also checks validation
results with $V_0=I$ and $100I$, pooled and per-condition coverage, phase residual
correlations, lag-one dependence, and drift. Ordinary log regression and
training-condition medians provide point-prediction baselines.

`to_dict()` / `from_dict()` persist a fitted `BayesianLinearRegression`, including
prior and posterior parameters. The study's model JSON also stores transforms,
class models, OLS coefficients, and condition medians.

## Offline replay

```python
from teerex.replay import replay

rows = [{
    "event_id": 0, "num_photons": 100_000,
    "cpu_pre_runtime_s": 0.2, "gpu_runtime_s": 0.1,
    "cpu_post_locked_runtime_s": 0.01,
    "cpu_post_unlocked_runtime_s": 0.001,
}]
timeline = replay(rows, workers=4, max_inflight=8, policy="async")
```

Every input needs a unique nonnegative integer event ID and four finite,
nonnegative demands. `num_photons` is optional. All events are released at time
zero; this is a finite job, without an external arrival process.

| Policy | CPU ownership and dispatch |
| --- | --- |
| `blocking` | One admitted event; phases run serially. |
| `synchronous` | Each event retains a CPU worker during queue/GPU waits and cleanup. GPU acquisition follows input order. Active events are bounded by `min(workers, max_inflight)`. |
| `async` | CPU worker is released after generation and reacquired for cleanup. GPU queue is FIFO by submission; input order breaks ties. Admission is bounded by `max_inflight`. |

All CPU phases use one shared finite worker pool. Ready cleanup takes priority
over new CPU-pre work; phases are nonpreemptive. Locked and unlocked cleanup
run contiguously on the same worker; the gate releases between them. An async
CPU worker can still be busy when GPU completion makes cleanup ready.

`event_batch_size=N` is supported only for synchronous replay. A worker reserves
a contiguous input chunk and executes its events serially, retaining chunk
ownership. Reserved future events do not count toward WIP; GPU acquisition still
follows global input order. This models Geant4 worker batching. It does not combine
photons from several events into one GPU launch. The study CLI currently searches
unit-event batches only.

Outputs include worker assignments and offsets, queue time, cleanup-worker wait,
gate occupation, and completion latency. `latency_s` includes admission backlog
from the common release time; `service_latency_s` begins at admission. The replay's
`gpu_wait_runtime_s` is submit-to-completion elapsed time, not a measurement of an
async CPU worker blocking. CPU envelopes include waits: use the separate pre/post
intervals for CPU busy lanes.

Service demands remain fixed across resource counts. CPU contention, executor
overheads, memory capacity, and changed GPU batch efficiency need additional
measurements or explicit sensitivity assumptions. Queued photons are a workload
proxy, not a VRAM estimate.

## Study CLI and selection procedure

`study` fits duration models to a complete measured-run archive, compares
dispatch policies by offline replay, and writes a report and supporting artifacts.
Use `uv run study --help` for argument help and examples. To run the default
49-candidate grid with 64 predictive draws:

```bash
uv run study --root /path/to/teerex-study/drich-study-v2
```

| Argument | Meaning and accepted values | Default |
| --- | --- | --- |
| `--root ROOT_DIR` | Required archive directory containing `runs/<run_id>/`. Each complete `physics_production` run needs `run.json`, `events.csv`, `events.manifest.json`, and `simg4ox.stdout.log`. Replications 0 and 1 train the models, 2 selects the policy, and 3 evaluates the frozen choice. Every measured condition must occur in the training, validation, and test splits. New derived artifacts are written beneath this same root. | Required |
| `--draws N` | Positive integer giving the number of posterior predictive draws per replayed stream. Each draw samples event phase durations; all policy candidates share the validation draws so comparisons use the same sampled demands. More draws reduce Monte Carlo noise in policy summaries and increase runtime. This changes the sampling budget while keeping the measured inputs and model fit fixed. | `64` |
| `--workers COUNT [COUNT ...]` | Space-separated, distinct positive integers defining candidate CPU worker-pool sizes in the simulated scheduler. CPU pre-processing and post-processing share this pool. This sets replay concurrency rather than the number of processes running the analysis. Values replace the default grid. | `1 2 4 8` |
| `--inflight-limits EVENTS [EVENTS ...]` | Space-separated, distinct positive integers defining candidate limits on admitted but unfinished events. An event occupies a slot through CPU pre-processing, GPU queueing and service, and both CPU post-processing phases. Async replay can admit more events than CPU workers; synchronous replay is additionally limited by the worker count. The unit is events, with no implied photon count or GPU-memory limit. Values replace the default grid. | `1 2 4 8 16 32` |

Resource lists are sorted before evaluation. Every worker count is paired with
every in-flight limit for both synchronous and async policies, and a separate
one-worker, one-event blocking control is always included. With `W` worker values
and `Q` in-flight values, the CLI evaluates `1 + 2 * W * Q` configurations, including
configurations whose effective concurrency is identical. All candidates use one
event per CPU batch.

For a quick exploratory run, reduce both the draw count and the grid:

```bash
uv run study --root /path/to/teerex-study/drich-study-v2 --draws 8 \
  --workers 1 2 --inflight-limits 1 2 4
```

This evaluates 13 configurations with noisier Monte Carlo estimates. Posterior
probabilities change in steps of `1 / draws`: 64 draws give a resolution of about
1.6 percentage points, so even the default is an initial analysis rather than
precise tail-probability estimation. Increasing `--draws` improves sampling
precision; model adequacy still depends on held-out calibration and sensitivity
checks. Analytic per-event predictive medians and intervals do not depend on this
draw count.

The production-v3 notebook uses a wider grid than the archive CLI defaults.
Rebuild its five separate fits directly from the combined production CSV:

```bash
PYTHONPATH=. uv run python scripts/refresh_ratio_study.py \
  ../teerex-study/ratio-study-v3/all-production-events.csv --draws 64
```

| Setting | Archive CLI default | Production-v3 notebook |
| --- | --- | --- |
| Posterior draws | 64 | 64 |
| CPU worker grid | 1, 2, 4, 8 | 1, 2, 4, 8, 16, 20 |
| In-flight grid | 1, 2, 4, 8, 16, 32 | 1, 2, 4, 8, 16, 32, 64 |
| Candidate count | 49 | 85 |
| CPU event batch | 1 | 1 |
| Training replications | 0, 1 | 11,000 retained rows across five fits |
| Validation replication | 2 | 5,500 retained rows across five fits |
| Test replication | 3 | 5,500 retained rows across five fits |
| Analysis seed | 20261031 | 20261031 |

For each grid point the CLI evaluates synchronous and async scheduling, plus one
blocking control. It compares all candidates on shared validation predictive draws.
Among policies within 5% of the best posterior median makespan, it selects the
least workers, then least in-flight capacity, then policy name. It freezes that
choice before final test prediction/replay. This is a finite-grid recommendation.

The archive CLI includes homogeneous streams (cap 128) and a shuffled
equal-condition mixture (cap 256, seed 20261031). The production-v3 refresh uses
one 256-event mixture per workload and split: 64 events from each of the four
conditions in a RICH geometry, or 256 from the single condition in an argon
workload. “Balanced mixture” in artifact fields means equal condition weighting;
the balanced argon resource regime is a separate concept. Every held-out event,
rather than only these scheduling subsets, contributes to prediction checks.

## Inputs, integrity, and outputs

Each input is `runs/<run_id>/` containing `events.csv`, `events.manifest.json`,
`simg4ox.stdout.log`, and `run.json`. Metadata must identify a complete
`physics_production` run, particle, momentum, multiplicity, replication, event
count, and warm-up count. When recorded, configuration/macro/CSV hashes and seeds
are verified. Timing boundaries and phase sums must agree, event IDs must be
unique within a run, and independent replications must use different recorded
CPU seeds and GPU streams. Invalid complete runs fail ingestion.

Only training rows fit transforms and models; warm-up rows remain in the raw audit
dataset but leave the splits. Final test outcomes do not fit or select the policy.
Every invocation creates a fresh `analysis-<timestamp>` under `datasets/`, `models/`,
`figures/`, and `reports/`, preserving raw inputs. Outputs include split/run indexes,
held-out predictions and metrics, observed and posterior policy comparisons,
sensitivities, a frozen policy, model parameters, standalone plots, and a report.

The [portable evidence](study/README.md) supports the notebook and replay. Its
combined CSV is accepted by `scripts/refresh_ratio_study.py`, which validates
timing identities, seeds, and replication splits and overwrites derived evidence
in `docs/study/`. It is not a full run archive accepted by the `study --root`
CLI; that workflow still requires the original run tree and its metadata.
