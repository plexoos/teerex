# Bayesian adaptation for CPU/GPU event dispatch

Build a new, executable marimo notebook starting with the decision and mathematics,
training on the current `docs/study/events.csv.gz`, and demonstrating online CPU
allocation on an unfamiliar machine. The user selected completion time plus CPU
allocation cost. Existing studies and raw data stay intact.

## Approaches

Whole-job routing across machines would need observations from multiple machines.
A fixed offline resource search does not demonstrate online adaptation. Implement
rolling batches of CPU generation → GPU host service → gate-held cleanup → remaining
CPU work. Choose workers and work-in-flight, drain the batch, observe, update, repeat.
This is a first adaptive controller, not an external-arrival or multi-GPU scheduler.

## Model and decisions

Fit joint four-phase log-duration distributions by known source condition, using
replications 0/1 and run-level baseline uncertainty. Hold out replication 2 for
validation and selection of a fixed baseline; use replication 3 for final replay.
Use an empirical-Bayes Gaussian mixture over unknown CPU/GPU log speed corrections,
condition baseline uncertainty, and a small explicit contention grid. Update only
from completed batches. Preserve joint phase variation and shared parameter draws.
Unknown condition labels require calibration; no extrapolation claim.

For action (c,q), minimize expected batch makespan + lambda * allocated core-seconds,
where allocated core-seconds = c * makespan. One exclusive GPU gate stays held through
locked cleanup. Compare fixed, nominal, posterior mean-demand, posterior expected-cost,
and hindsight policies on identical event blocks. Report Monte Carlo error and
planning overhead, along with completion time, CPU allocation, and objective.

Measured held-out replay, synthetic machine transfer, and misspecification stress
are distinct experiments. Synthetic variability tests the controller; it does not
supply evidence of physical hardware response. Discuss broader source physics,
randomized concurrency, hardware metadata, and multiple machines as next measurements.

## Validation

Check no training/test overlap, positive finite phases, analytic Bayesian updates,
likelihood identifiability, forward-only observation, reproducible paired workloads,
and fast simulation agreement with the existing replay. Run marimo check and export
with execution. Report existing snapshot inconsistencies without changing artifacts.
