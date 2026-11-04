# CPU/GPU batching and dispatch: production v3

The [main notebook](../notebooks/simload.py) now analyzes the supplied
`teerex-study/ratio-study-v3/all-production-events.csv`: 22,000 retained events,
44 independent processes, 11 source conditions, four replications/condition,
and 500 retained events/process. The [portable evidence](study/README.md)
contains the complete CSV and rebuilt predictions, policy searches, and provenance.
This replaces the notebook's earlier dRICH-only fit and short argon pilots.

## Measured sample and ratios

| Workload | Source conditions | Retained events | Median event GPU-stage share |
| --- | --- | ---: | --- |
| dRICH | e−/μ−, 20 GeV/c, 10/100 primaries | 8,000 | 1.6–7.1% by condition |
| pfRICH | e−/μ−, 20 GeV/c, 10/100 primaries | 8,000 | 6.5–26.1% by condition |
| Balanced argon | 65,536 electrons × 1 keV kinetic energy | 2,000 | 57.2% |
| GPU-heavy argon, n10 | 10 electrons × 10 MeV kinetic energy | 2,000 | 93.9% |
| GPU-heavy argon, n100 | 100 electrons × 1 MeV kinetic energy | 2,000 | 94.3% |

Per-event GPU share divides GPU host service by the sum of CPU generation, GPU
service, gate-held cleanup, and remaining CPU time. This is distinct from a ratio
of medians, device-kernel duty, or SM utilization. Host service includes transfers
and synchronization. The supplied CSV contains no device profiles, GPU model,
or CPU affinity; older profiler findings are not attached to these new runs.

The export already excludes warm-up events. IDs 5–504 remain in every process;
no additional trimming is applied. Of the 2,000 balanced events, 1,975 fall in
the 40–60% timing-share band; all 25 outside-band rows are retained. The notebook
shows all events in four process-level time series to expose drift and tails.
Original offsets, seeds, metadata, and independent process clocks are preserved.

The two GPU-heavy conditions have the same nominal 100 MeV total kinetic energy
per event, but different track multiplicities and photon counts. Their median
photon yields are approximately 2.62 million and 2.76 million. The balanced
condition yields about 2.27 million. These controlled many-primary samples
characterize performance and do not establish physical accuracy or represent
realistic collision events.

## Duration fits and validation

Five separate four-phase Bayesian log-duration fits cover dRICH, pfRICH, balanced
argon, and each GPU-heavy condition. The existing normal/inverse-gamma prior,
training-only transforms, feature-availability rules, and condition-median/OLS
baselines are retained. Constant source features in each single-condition argon
fit do not establish extrapolation across energy or multiplicity.

Replications 0/1 train (11,000 events), replication 2 selects policies (5,500),
and replication 3 tests frozen choices (5,500). Every held-out event contributes
to duration metrics; only smaller, explicitly identified streams feed scheduling.
The sample still has two training replication clusters per condition.

Nominal 95% test coverage ranges from 93.1–98.5% for dRICH, 94.6–97.4% for pfRICH,
and 96–100% across the argon fits. Overcoverage and condition-level failures
remain visible in the notebook. The baseline is not declared calibrated or best
in every phase; test MAE compares it with ordinary log regression and condition
medians. Training residual diagnostics and joint-block sensitivity are saved.

## Validation-selected policies on held-out test streams

The search evaluates 85 configurations per workload: blocking plus synchronous
and async combinations of 1/2/4/8/16/20 CPU workers and 1/2/4/8/16/32/64 in-flight
events. All candidates share 64 predictive draws on a 256-event validation
stream. Selection chooses least workers, then least in flight, within 5% of
best validation median makespan, then freezes the choice before test evaluation.

| Workload | Frozen policy | Workers / in flight | Test median makespan | Median simulated speedup | Observed-demand replay |
| --- | --- | --- | ---: | ---: | ---: |
| dRICH | async | 20 / 32 | 1.326 s | 17.34× | 1.402 s |
| pfRICH | async | 16 / 16 | 0.168 s | 6.77× | 0.175 s |
| Balanced argon | async | 1 / 2 | 64.838 s | 1.67× | 64.282 s |
| GPU-heavy argon, n10 | async | 1 / 1 | 76.802 s | 1.00× | 76.596 s |
| GPU-heavy argon, n100 | async | 1 / 1 | 75.394 s | 1.00× | 75.118 s |

Each row uses its own 256-event test stream. RICH streams contain 64 events from
each condition, deterministically shuffled; argon streams contain 256 events
from one condition. Durations and workload compositions differ across rows,
so their makespans are not a common-work benchmark. Speedups compare the frozen
policy with blocking on the same workload and shared draws.

One-worker/one-in-flight async executes serially. In the GPU-heavy cases the
least-concurrency selection rule accepts it within 5% of the best validation
candidate; this does not claim that every larger configuration has exactly zero
gain. Balanced argon benefits from overlapping generation with another event's
GPU service even with one shared CPU worker.

**All policy results are offline schedules, including observed-demand replay.**
They are not measured asynchronous hardware speedups. CPU demand ×1.5/×2 reduces
the nominal dRICH estimate to about 11.94×/9.09× and pfRICH to 5.77×/5.00×. These
are uncalibrated contention scenarios, applied only when CPU workers exceed one;
one-worker choices are unchanged. Predictive intervals are conditional on the
model, and joint residual blocks do not correct conditional model bias.

## Boundaries and next experiments

CPU generation and both cleanup phases share the finite worker pool. The GPU
gate remains occupied through locked cleanup and any cleanup-worker wait.
Resource summaries separate occupied gate time from GPU service time. Completion
latency includes the time-zero admission backlog; admitted latency is separate.
Photon queues are a work proxy without a measured VRAM conversion.

Next steps are to validate the frozen policies in an actual charged asynchronous
executor, measure service-time changes under concurrency, and improve conditional
calibration. Cleanup reservations, larger worker-owned batches, and cross-event
photon aggregation require their own measurements, memory limits, and latency
budgets. No aggregation cost model is inferred from this CSV.

See the [model reference](bayesian-dispatch.md) for equations and scheduler rules,
the [evidence guide](study/README.md) for rebuild commands and SHA256 checks, and
[Simload documentation](simload.md) for the original synthetic CUDA experiment.
