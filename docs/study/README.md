# Production ratio-study-v3 evidence

The primary [Teerex notebook](../../notebooks/simload.py) uses this portable
snapshot of `teerex-study/ratio-study-v3/all-production-events.csv`. It contains
22,000 retained events from 44 blocking, single-threaded processes: 11 conditions,
four replications/condition, and 500 retained events/process (IDs 5–504).
No additional warm-up removal, tail trimming, or outlier filtering is applied.

| Workload | Conditions | Events | Definition |
| --- | ---: | ---: | --- |
| dRICH | 4 | 8,000 | e−/μ−, 20 GeV/c, 10/100 primaries |
| pfRICH | 4 | 8,000 | e−/μ−, 20 GeV/c, 10/100 primaries |
| Balanced argon | 1 | 2,000 | 65,536 electrons × 1 keV kinetic energy |
| GPU-heavy argon, n10 | 1 | 2,000 | 10 electrons × 10 MeV kinetic energy |
| GPU-heavy argon, n100 | 1 | 2,000 | 100 electrons × 1 MeV kinetic energy |

The original CSV records one GPU UUID, but no device model or CPU affinity.
No raw run manifests, geometry contents, device profiles, optical controls, or
asynchronous executor measurements were supplied with this export. The snapshot
attests to the CSV's bytes and internal consistency, not those missing sources.

## Files

- `events.csv.gz`: exact source CSV bytes after decompression, with every column,
  original run-local clock, seed, run identity, event identity, and timing tail.
- `manifest.json`: source SHA256 and byte count; SHA256/size for every artifact;
  fitting/scheduling code hashes; random seed, predictive draws, and policy grid.
- `duration-models.json`: five separate four-phase Bayesian fits, training-only
  transforms, priors, posterior parameters, OLS coefficients, and condition medians.
- `training-diagnostics.json`: training residual correlations and drift diagnostics.
- `heldout-metrics.csv`: validation/test errors for Bayesian medians, OLS on log
  time, and training-condition medians, plus Bayesian predictive coverage.
- `test-predictions.csv.gz`: all 5,500 test events × four phases, including
  observed durations, three point predictions, and 50/80/95% predictive bounds.
- `heldout-condition-metrics.csv`: condition/work-class errors and coverage.
- `frozen-policies.json`: validation-selected policy for each workload and the
  exact `(run_id, event_id)` identities/order of validation and test streams.
- `policy-summary.csv`: the 85-configuration validation search and frozen-policy
  test comparisons, including CPU demand ×1.5/×2 and joint-block sensitivity.
- `resource-summary.csv`: fixed observed-demand test replays for the frozen
  policy and blocking control, with GPU service and gate occupancy separated.
- `blocking-trace-validation.csv`: phase-service, submission-overhead, gap, and
  boundary-reconstruction accounting for each original process.

## Reproduction and selection

From the repository root:

```bash
PYTHONPATH=. uv run python scripts/refresh_ratio_study.py \
  ../teerex-study/ratio-study-v3/all-production-events.csv
PYTHONPATH=. uv run python scripts/refresh_ratio_study.py --verify
uv run marimo edit notebooks/simload.py
```

The source archive is read only. The refresh overwrites derived evidence here.
The notebook and hosted WebAssembly version load the saved artifacts without
refitting; neither needs CUDA, PyTorch, Ray, or access to the sibling archive.

Replications 0/1 train; replication 2 selects; replication 3 evaluates. Fits are
separate for dRICH, pfRICH, and each of the three argon conditions. Feature
availability, the normal/inverse-gamma prior, and scheduler rules remain those
of `teerex.study`; generated counts never enter CPU-generation prediction.

Each workload uses 256 events per scheduling stream, evenly spaced within each
condition, then deterministically shuffled with seed 20261031. RICH streams
contain 64 events/condition; each argon stream has 256 events from its one
condition. This equal-condition weighting is unrelated to the balanced argon
resource ratio. All held-out events participate in prediction checks.

The default search uses 64 shared predictive draws and 85 configurations:
blocking plus synchronous/async combinations of 1/2/4/8/16/20 CPU workers and
1/2/4/8/16/32/64 in-flight events. Selection chooses least workers, then least
in flight, within 5% of the best validation median makespan. Only the frozen
choice and blocking control are evaluated on test streams. Test values cannot
choose a configuration. Joint blocks are a dependence sensitivity, and CPU
multipliers are uncalibrated contention scenarios.

All scheduling results are offline finite-stream estimates. Measured GPU host
service includes transfers/synchronization and cannot establish device-kernel
duty or SM utilization. Photon queues have no validated VRAM conversion, and
serial phase timings need validation under concurrency.
