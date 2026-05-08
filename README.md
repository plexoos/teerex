# Usage

```
simload --config config/fixed_balanced_50.json --out-dir simload_runs
```

or override the full CSV path explicitly.

```
simload --config config/fixed_balanced_50.json --out simload_runs/custom_name.csv
```

# Teerex Simload Architecture

This repository contains a synthetic Ray-based simulation that emulates a Geant4-style mixed CPU/GPU
event workflow.

The simulation is organized around a single sequential CPU event manager with optional asynchronous
GPU offload:

1. `sample_event`: synthetic event generation, including event-to-event size variation.
2. `_run_cpu_stage`: synthetic CPU event work executed sequentially by the main process.
3. `gpu_stage`: synthetic GPU event work submitted to Ray after the CPU stage for that event.
4. `_merge_event_results`: lightweight merge/bookkeeping for final per-event outputs.

This follows the normal Geant4 event loop assumption: the CPU manager processes one event's CPU part
at a time. There is no external event-arrival clock, batch mode, or synthetic queueing source.

GPU offload is controlled by `gpu_async`:

- `false`: after event N's CPU part finishes, the manager waits for event N's GPU work and merge
  before starting event N+1.
- `true`: after event N's CPU part finishes and GPU work is submitted, the manager can start event
  N+1's CPU part while event N's GPU work runs. CPU processing remains strictly sequential.

Each event is represented by the `EventSpec` dataclass, which includes:

- Size information: `event_id`, `event_size_s`
- Work split: `cpu_fraction`, `gpu_fraction`, `cpu_work_s`, `gpu_work_s`
- Synthetic compute sizes: `cpu_matrix_size`, `gpu_matrix_size`
- Synthetic GPU footprint: `gpu_memory_mb`, `h2d_mb`, `d2h_mb`
- Resource requests: `num_cpus`, `num_gpus`

## Event Generation

`generate_events()` creates a stream of `EventSpec` objects using `sample_event()`.

- `sample_event()` draws `event_size_s` from a configurable size distribution.
- CPU/GPU split is sampled from a Beta distribution (`cpu_fraction_beta_alpha`,
  `cpu_fraction_beta_beta`).
- Matrix sizes and memory/transfer sizes are scaled from sampled work fractions.

Event size distributions:

- `fixed`: all events use `mean_event_size_s`.
- `uniform`: flat variation in `[mean_event_size_s - event_size_spread_s,
  mean_event_size_s + event_size_spread_s]`.
- `gaussian`: normal variation with mean `mean_event_size_s` and standard deviation
  `event_size_spread_s`.

The sampled size is clipped to `min_event_size_s`.

## CPU Manager Workflow

For each event in `run_simulation()`:

1. Run CPU event work locally on the single CPU manager thread.
2. Submit `gpu_stage` with `num_gpus=num_gpus_per_event` and a small CPU allocation.
3. If `gpu_async=false`, wait for that GPU task and merge before starting the next event.
4. If `gpu_async=true`, keep the GPU task pending and start the next event's CPU part. After all CPU
   events have been issued, drain and merge pending GPU results.

The final merged record includes:

- `runtime_s` for the full event interval from CPU start to merge completion
- `cpu_runtime_s` from CPU stage
- `gpu_runtime_s` from GPU stage
- `merge_runtime_s` from the merge/bookkeeping step
- `cpu_start_time`, `cpu_end_time`, `gpu_start_time`, `gpu_end_time`, `merge_start_time`,
  `merge_end_time`

## Stage Behavior

### CPU stage

- Uses NumPy matrix multiplications in a time-bounded loop (`cpu_work_s`).
- Runs in the main process so CPU event intervals do not overlap.
- Returns timing (`start_time`, `end_time`, `runtime_s`) and synthetic counters (`iterations`, `checksum`).

### GPU stage

- Uses PyTorch CUDA tensors and synchronizations.
- Simulates host-to-device (`h2d_mb`) and device-to-host (`d2h_mb`) transfers.
- Simulates event-level GPU memory residency (`gpu_memory_mb`).
- Runs repeated matrix multiplications until `gpu_work_s` target elapsed.
- Returns timing and synthetic counters.

### Reduce stage

- Adds a small configurable delay (`reduce_work_s`) to represent final event bookkeeping.
- Returns event-level merged timing outputs.
- With `gpu_async=false`, this merge happens before the next event starts.
- With `gpu_async=true`, this merge may happen later, after the CPU manager has advanced to newer
  events.

## Output Assembly

After all event records have been merged:

1. Event specs and submission metadata are built into `events_df`.
2. Merged event outputs are built into `results_df`.
3. Both are merged on `event_id` into the final DataFrame.
4. Derived timing columns are added:
	- `start_offset_s = start_time - t0`
	- `end_offset_s = end_time - t0`

The full DataFrame is written to CSV at the resolved output path.

## Runtime Configuration

`main()` supports:

- `--config`: JSON file to override defaults
- `--num-cpus`, `--num-gpus`: Ray cluster resource limits
- `--out-dir`: directory for the generated CSV filename, such as `simload_runs`
- `--out`: full output CSV path, which overrides `--out-dir`

Important runtime defaults include event counts, event size distribution, CPU/GPU fraction
distribution, GPU async behavior, and per-task resource requests.

## Why This Architecture

This design separates concerns cleanly:

- Sampling and workload generation are deterministic per event id + seed.
- Stage implementations isolate CPU/GPU behavior while preserving a sequential CPU event loop.
- Ray object references represent offloaded GPU work and allow async overlap when configured.
- Final reduction provides a single record per event for downstream analysis.

The result is a practical synthetic benchmark for testing mixed CPU/GPU scheduling and end-to-end
event runtime under different workload profiles.
