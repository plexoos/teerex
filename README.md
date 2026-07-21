# Usage

```
simload --config config/fixed_balanced_50.json --out-dir simload_runs
```

or override the full CSV path explicitly.

```
simload --config config/fixed_balanced_50.json --out simload_runs/custom_name.csv
```

You can override the GPU synchronization mode from the CLI:

```
simload --config config/fixed_balanced_50.json --mode event_barrier
```

The previous Ray implementation is still available as:

```
simload-ray --config config/fixed_balanced_50.json --out-dir simload_runs
```

# Teerex Simload Architecture

This repository contains a synthetic local simulation that emulates a Geant4-style mixed CPU/GPU
event workflow using a single CPU event manager and a dedicated local Torch CUDA worker.

The simulation is organized around a single sequential CPU event manager with optional asynchronous
GPU offload:

1. `sample_event`: synthetic event generation, including event-to-event size variation.
2. `_run_cpu_payload`: synthetic CPU work executed sequentially by the main process.
3. `LocalGpuExecutor`: one local worker thread that runs Torch CUDA payloads in submission order.
4. `_merge_event_results`: lightweight merge/bookkeeping for final per-event outputs.

This follows the normal Geant4 event loop assumption: the CPU manager processes one event's CPU part
at a time. There is no external event-arrival clock, batch mode, or synthetic queueing source.

GPU offload is controlled by `gpu_mode`:

- `blocking`: dispatch the GPU payload during the CPU event, wait immediately, then continue the
  post-dispatch CPU payload and merge.
- `event_barrier`: dispatch the GPU payload during the CPU event, run the post-dispatch CPU payload
  while the GPU works, then wait/merge before starting the next event.
- `async`: dispatch the GPU payload during the CPU event, run the post-dispatch CPU payload, and
  allow later CPU events to start before earlier GPU results are merged.

Legacy `gpu_async: false` maps to `blocking`, and `gpu_async: true` maps to `async`.

Each event is represented by the `EventSpec` dataclass, which includes:

- Size information: `event_id`, `event_size_s`
- Work split: `cpu_fraction`, `gpu_fraction`, `cpu_work_s`, `gpu_work_s`
- GPU dispatch point: `dispatch_fraction`, `cpu_pre_work_s`, `cpu_post_work_s`
- Synthetic compute sizes: `cpu_matrix_size`, `gpu_matrix_size`
- Synthetic GPU footprint: `gpu_memory_mb`, `h2d_mb`, `d2h_mb`
- Resource requests: `num_cpus`, `num_gpus`

## Event Generation

`generate_events()` creates a stream of `EventSpec` objects using `sample_event()`.

- `sample_event()` draws `event_size_s` from a configurable size distribution.
- CPU/GPU split is sampled from a Beta distribution (`cpu_fraction_beta_alpha`,
  `cpu_fraction_beta_beta`).
- The CPU work is split around `gpu_dispatch_fraction_mean`, with bounded random jitter from
  `gpu_dispatch_fraction_jitter`, `gpu_dispatch_fraction_min`, and `gpu_dispatch_fraction_max`.
- Matrix sizes and memory/transfer sizes are scaled from sampled work fractions.
- Random sampling is deterministic per event id. `seed` defaults to `12345`, and
  `event_size_seed` can be set separately when multiple configs should use the
  same event sizes while changing scheduling or work-split settings.

Event size distributions:

- `fixed`: all events use `mean_event_size_s`.
- `uniform`: flat variation in `[mean_event_size_s - event_size_spread_s,
  mean_event_size_s + event_size_spread_s]`.
- `gaussian`: normal variation with mean `mean_event_size_s` and standard deviation
  `event_size_spread_s`.

The sampled size is clipped to `min_event_size_s`.

## CPU Manager Workflow

For each event in `run_simulation()`:

1. Run the pre-dispatch CPU payload locally on the single CPU manager thread.
2. Submit the GPU payload to the local GPU worker.
3. Depending on `gpu_mode`, wait immediately, wait at the event boundary, or keep the GPU result
   pending while later events begin.
4. Merge the CPU/GPU result once the selected mode permits it.

The final merged record includes:

- `runtime_s` for the full event interval from CPU start to merge completion
- `cpu_runtime_s` from the pre-dispatch plus post-dispatch CPU payloads
- `gpu_runtime_s` from GPU stage
- `gpu_queue_delay_s` from local submission to worker start
- `gpu_wait_runtime_s` from CPU wait points
- `merge_runtime_s` from the merge/bookkeeping step
- phase timings such as `cpu_pre_start_time`, `gpu_submit_time`, `gpu_start_time`,
  `cpu_post_end_time`, `merge_start_time`, and `merge_end_time`

## Stage Behavior

### CPU stage

- Uses NumPy matrix multiplications in time-bounded pre/post loops.
- Runs in the main process so CPU event intervals do not overlap.
- Returns timing (`start_time`, `end_time`, `runtime_s`) and synthetic counters (`iterations`, `checksum`).

### GPU stage

- Uses PyTorch CUDA tensors and synchronizations from a dedicated local worker thread.
- Simulates host-to-device (`h2d_mb`) and device-to-host (`d2h_mb`) transfers.
- Simulates event-level GPU memory residency (`gpu_memory_mb`).
- Runs repeated matrix multiplications until `gpu_work_s` target elapsed.
- Returns timing and synthetic counters.

### Reduce stage

- Adds a small configurable delay (`reduce_work_s`) to represent final event bookkeeping.
- Returns event-level merged timing outputs.
- With `gpu_mode=blocking` or `gpu_mode=event_barrier`, this merge happens before the next event
  starts.
- With `gpu_mode=async`, this merge may happen later, after the CPU manager has advanced to newer
  events.

## Output Assembly

After all event records have been merged:

1. Event specs and submission metadata are built into `events_df`.
2. Merged event outputs are built into `results_df`.
3. Both are merged on `event_id` into the final DataFrame.
4. Derived timing columns are added:
	- `start_offset_s = start_time - t0`
	- `end_offset_s = end_time - t0`
	- phase offsets such as `gpu_submit_time_offset_s`, `gpu_start_time_offset_s`,
	  `cpu_pre_start_time_offset_s`, and `cpu_post_end_time_offset_s`

The full DataFrame is written to CSV at the resolved output path.

## Runtime Configuration

`main()` supports:

- `--config`: JSON file to override defaults
- `--mode`: one of `blocking`, `event_barrier`, or `async`
- `--num-cpus`, `--num-gpus`: accepted for CLI compatibility; local mode does not create a Ray
  resource pool
- `--out-dir`: directory for the generated CSV filename, such as `simload_runs`
- `--out`: full output CSV path, which overrides `--out-dir`

Important runtime defaults include event counts, event size distribution, CPU/GPU fraction
distribution, GPU dispatch timing, GPU mode, and payload sizes.

## Why This Architecture

This design separates concerns cleanly:

- Sampling and workload generation are deterministic per event id + seed.
- Stage implementations isolate CPU/GPU behavior while preserving a sequential Geant4-like CPU
  event loop.
- The GPU worker removes Ray scheduler/worker-start latency from the timing model while still
  running real Torch CUDA payloads.
- Final reduction provides a single record per event for downstream analysis.

The result is a practical synthetic benchmark for testing mixed CPU/GPU scheduling and end-to-end
event runtime under different workload profiles.
