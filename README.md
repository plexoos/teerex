# Teerex Simload

Teerex Simload is a synthetic CPU/GPU scheduling study for event-based scientific applications. It
models a Geant4-like workflow in which one sequential CPU event manager performs part of an event,
offloads work to a GPU, and eventually merges the two results. The central question is how the
synchronization policy changes CPU/GPU overlap, GPU queueing, CPU waiting, event completion latency,
and the simulated workload makespan.

The benchmark runs real, time-bounded NumPy and PyTorch workloads. It is useful for controlled
scheduling experiments, but it is not a physics simulation or a predictive model of a particular
Geant4 application.

## Quick start

### Install and check the GPU

Simload requires Python 3.11 or later, an NVIDIA GPU, and a CUDA-enabled PyTorch installation. There
is no CPU-only fallback. From a checkout of this repository, the reproducible setup uses the
included `uv.lock` file:

```bash
uv sync
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name())"
```

### Run one experiment

```bash
uv run simload \
    --config config/mode_event_barrier.json \
    --out simload_runs/quickstart_event_barrier.csv
```

This command has one input and two forms of output:

- [`config/mode_event_barrier.json`](config/mode_event_barrier.json) is a partial JSON
  configuration. It selects the `event_barrier` synchronization policy; every omitted setting comes
  from the defaults in [`teerex/simload.py`](teerex/simload.py).
- The terminal shows the selected CUDA device, per-event progress, and a selected per-event results
  table.
- `simload_runs/quickstart_event_barrier.csv` contains one row per event and the complete sampled
  workload, measured phase timings, queue/wait durations, and timeline offsets.

With this configuration, Simload generates 20 events with a nominal mean work budget of 3 seconds
per event. The event manager runs the CPU part sequentially, while the event's post-dispatch CPU
work is allowed to overlap its GPU work. Because this is a real compute workload, expect the example
to take tens of seconds or longer, depending on the CPU, GPU, and numerical libraries.

> [!IMPORTANT]
> Simload overwrites an existing output CSV without prompting. Use a distinct `--out` path for
> every policy, workload, hardware configuration, or repeat that you want to retain.

### Inspect the result

The CSV separates the workload that was requested from the time that was observed. The following
snippet prints the most useful event-level columns and a small run summary:

```bash
uv run python - <<'PY'
import pandas as pd

df = pd.read_csv("simload_runs/quickstart_event_barrier.csv")

columns = [
    "event_id",
    "event_size_s",
    "runtime_s",
    "cpu_runtime_s",
    "gpu_runtime_s",
    "gpu_queue_delay_s",
    "gpu_wait_runtime_s",
]
print(df[columns].head().to_string(index=False))

workload_makespan_s = float(df["end_offset_s"].max())
print(f"\nevents:                 {len(df)}")
print(f"workload makespan:       {workload_makespan_s:.3f} s")
print(f"throughput:              {len(df) / workload_makespan_s:.3f} events/s")
print(f"mean event latency:      {df['runtime_s'].mean():.3f} s")
print(f"mean GPU queue:          {df['gpu_queue_delay_s'].mean():.3f} s")
print(f"mean explicit GPU wait:  {df['gpu_wait_runtime_s'].mean():.3f} s")
PY
```

Interpret these values as follows:

- `event_size_s` is a nominal work budget, not a measured runtime. It is divided into requested CPU
  and GPU work.
- `runtime_s` is event residence time: CPU-pre start through final merge. It is not the sum of CPU
  and GPU runtimes, because the stages may overlap.
- `gpu_queue_delay_s` reveals backlog at the single GPU worker. `gpu_wait_runtime_s` measures time
  spent at an explicit synchronization point.
- `end_offset_s.max()` is the measured workload makespan from the simulation timing origin through
  the final merge. It excludes event sampling before that origin and DataFrame, terminal, and CSV
  processing afterward.

For a visual comparison, open [the analysis notebook](notebooks/simload.ipynb):

```bash
uv run jupyter lab notebooks/simload.ipynb
```

For the quick-start output, change the notebook's `csv_files` cell and final timeline call to:

```python
csv_files = [Path("../simload_runs/quickstart_event_barrier.csv")]

fig, ax, timeline = plot_run_timeline(df, "quickstart_event_barrier")
```

The notebook combines runs and draws CPU/GPU busy intervals, which makes overlap and idle gaps much
easier to see.

## Compare the synchronization policies

The three bundled mode configurations use the same default workload and seed, so they are a useful
first comparison:

```bash
uv run simload --config config/mode_blocking.json \
    --out simload_runs/comparison/blocking.csv
uv run simload --config config/mode_event_barrier.json \
    --out simload_runs/comparison/event_barrier.csv
uv run simload --config config/mode_async.json \
    --out simload_runs/comparison/async.csv
```

The output directory is created automatically. These paths keep the bundled example CSVs untouched.

| Mode | Event ordering | Overlap | What to look for |
| --- | --- | --- | --- |
| `blocking` | CPU pre → submit GPU → wait → CPU post → merge | None by design | The wait exposes essentially the full GPU stage; queue delay should remain small. |
| `event_barrier` | CPU pre → submit GPU → CPU post → wait → merge | Within the current event | CPU-post work hides part of the GPU runtime; the wait records only the remaining GPU time. |
| `async` | CPU pre → submit GPU → CPU post → later CPU events → collect/merge | Within and across events | CPU progress can continue, but GPU queue delay can grow when submissions outpace the worker. |

All GPU tasks still execute in submission order on one worker and one CUDA stream. `async` pipelines
events; it does not run multiple GPU stages concurrently.

By default, asynchronous results are collected after all CPU events have been submitted. This means
that `runtime_s` for early asynchronous events includes deferred collection time. Use the measured
workload makespan, throughput, queue delay, and phase timeline alongside the event-latency
distribution.

## What the experiment models

The component topology is fixed; the synchronization mode determines the ordering of CPU post, GPU
waits, later CPU events, and merge:

```text
CPU event manager: sample -> CPU pre -> submit -> policy-dependent CPU post / wait / merge
                                         |
FIFO CUDA worker:                        +-> GPU stage -> result
```

The synthetic workload is constructed in four steps:

1. Sample `event_size_s` from a fixed, uniform, or Gaussian distribution.
2. Sample `cpu_fraction` from a Beta distribution and set
   `gpu_fraction = 1 - cpu_fraction`.
3. Split the requested CPU work around a sampled dispatch point into `cpu_pre_work_s` and
   `cpu_post_work_s`.
4. Scale the CPU/GPU matrix dimensions, nominal GPU allocation, and transfer volumes from the
   sampled work.

The CPU manager executes time-bounded NumPy matrix multiplications for the pre- and post-dispatch
phases. A dedicated local thread executes the GPU stage with PyTorch, including synthetic
host-to-device transfer, device allocation, matrix multiplication, device-to-host transfer, and
synchronization. A small final delay represents event reduction/bookkeeping.

There is no external event-arrival clock, batching model, or synthetic resource pool. CPU events are
started sequentially by one Python event-manager thread, although the NumPy BLAS implementation may
use additional native CPU threads internally.

## Reading the CSV

Fresh CSVs from the current local implementation contain one row and 62 columns per event. The
columns fall into four useful groups:

| Group | Representative columns | Meaning |
| --- | --- | --- |
| Sampled workload | `event_size_s`, `cpu_fraction`, `gpu_fraction`, `dispatch_fraction` | Nominal event composition chosen before execution. |
| Requested payload | `cpu_pre_work_s`, `gpu_work_s`, `cpu_post_work_s`, matrix sizes, `gpu_memory_mb`, `h2d_mb`, `d2h_mb` | Work targets and synthetic resource sizes. These are not observed utilization or bandwidth. |
| Measured durations | `runtime_s`, `cpu_*_runtime_s`, `gpu_runtime_s`, `gpu_queue_delay_s`, `gpu_wait_runtime_s`, `merge_runtime_s` | Wall-clock phase and event durations measured with `time.perf_counter()`. |
| Timeline | raw `*_time` columns and corresponding `*_time_offset_s` columns | Stage boundaries for reconstructing overlap within one run. |

The checked-in `simload_runs/mode_async.csv` and `mode_blocking.csv` files were produced with the
earlier, smaller schema. Regenerate experiments to obtain the columns documented here; the notebook
retains a fallback so it can also visualize those older artifacts.

### Key metrics

| Column | Interpretation |
| --- | --- |
| `event_size_s` | Requested total work budget. By construction, `cpu_work_s + gpu_work_s = event_size_s`. |
| `cpu_runtime_s` | Measured CPU busy time: `cpu_pre_runtime_s + cpu_post_runtime_s`. |
| `gpu_runtime_s` | Worker start to GPU completion, including allocation, transfers, matrix work, and synchronization. |
| `gpu_queue_delay_s` | Time from local submission to worker start. It measures FIFO backlog and thread scheduling, not CUDA kernel-queue latency. |
| `gpu_wait_runtime_s` | Time spent in the explicit wait performed by the event manager or result collector. |
| `merge_runtime_s` | Final synthetic reduction delay and its small Python overhead. |
| `runtime_s` | CPU-pre start to merge completion for the event. In `async`, this can include substantial deferred completion time. |
| `start_offset_s`, `end_offset_s` | Event start and completion relative to the simulation timing origin. |

The raw timestamp columns—such as `gpu_submit_time`, `gpu_start_time`, and `merge_end_time`—use an
arbitrary monotonic clock origin. They are not dates and should not be compared between processes.
Use their `*_offset_s` counterparts for plots. `start_offset_s` and `end_offset_s` are convenience
aliases of `start_time_offset_s` and `end_time_offset_s`.

A few distinctions matter when interpreting a run:

- In `blocking` mode, `cpu_end_time - cpu_start_time` spans the intervening GPU wait. Use
  `cpu_runtime_s` or the separate CPU phase intervals for actual CPU busy time.
- In `event_barrier` mode, CPU post and GPU runtime can overlap, so adding their durations
  overestimates elapsed time.
- In `async` mode, a near-zero `gpu_wait_runtime_s` can simply mean the GPU result was already ready
  when it was collected. It does not imply that the event completed immediately.
- `gpu_memory_mb` is the requested synthetic residency allocation, not measured peak device memory.
  Matrices and transfer buffers add to the actual footprint.
- CPU/GPU targets can overshoot because a matrix multiplication cannot be stopped partway through.
  GPU setup and the final device-to-host transfer also contribute to measured runtime.

The CSV records the sampled event specification but not the entire source configuration. Keep each
JSON config beside its uniquely named output when experiment provenance matters.

## Analysis notebook

[notebooks/simload.ipynb](notebooks/simload.ipynb) provides a compact comparison workflow. Its
checked-in inputs, `mode_async.csv` and `mode_blocking.csv`, use the earlier schema. In the notebook,
point the input cell at newly generated comparison files:

```python
csv_files = [
    Path("../simload_runs/comparison/blocking.csv"),
    Path("../simload_runs/comparison/event_barrier.csv"),
    Path("../simload_runs/comparison/async.csv"),
]
```

Then change the final timeline cell to use the matching file stems:

```python
fig, axes, timelines = plot_run_timeline(
    df,
    ["blocking", "event_barrier", "async"],
)
```

The notebook:

- loads and combines the CSVs, using each filename stem as the run label;
- writes the combined table to `simload_analysis/combined.csv`;
- displays separate CPU-pre and CPU-post intervals for current local outputs; and
- falls back to a whole-CPU interval when reading outputs from the earlier schema.

To save the displayed timeline, pass `outdir=outdir` to `plot_run_timeline` in the final cell.

## Configuration

Configuration files are JSON objects that shallowly override `default_params()` in
[`teerex/simload.py`](teerex/simload.py). They do not need to repeat every setting. The effective
precedence is:

```text
built-in defaults  <  JSON configuration  <  --mode CLI override
```

New configurations should prefer the explicit `gpu_mode` key. The older `gpu_async` boolean remains
supported: `false` maps to `blocking`, and `true` maps to `async`. If both are present, `gpu_mode`
wins.

For example, save the following as `config/my_experiment.json`:

```json
{
  "seed": 12345,
  "n_events": 8,
  "event_size_distribution": "gaussian",
  "mean_event_size_s": 1.0,
  "event_size_spread_s": 0.2,
  "cpu_fraction_beta_alpha": 4.0,
  "cpu_fraction_beta_beta": 4.0,
  "gpu_dispatch_fraction_mean": 0.5,
  "gpu_memory_base_mb": 128,
  "gpu_memory_jitter_mb": 128,
  "h2d_base_mb": 32,
  "d2h_base_mb": 8,
  "gpu_mode": "event_barrier"
}
```

Run it with an explicit output name:

```bash
uv run simload --config config/my_experiment.json --out simload_runs/my_experiment.csv
```

### Parameter reference

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `seed` | `12345` | Base per-event sampling seed. |
| `event_size_seed` | unset | Optional independent seed for event sizes, useful when varying the base seed separately. |
| `n_events` | `20` | Number of events. Use a positive integer. |
| `event_size_distribution` | `"gaussian"` | `fixed`, `uniform`, or `gaussian`. |
| `mean_event_size_s` | `3.0` | Fixed value or distribution mean, in nominal work seconds. |
| `event_size_spread_s` | `0.6` | Uniform half-width or Gaussian standard deviation; ignored for `fixed`. |
| `min_event_size_s` | `0.05` | Lower clamp applied after sampling. |
| `cpu_fraction_beta_alpha` | `4.0` | Alpha parameter of the CPU-share Beta distribution. |
| `cpu_fraction_beta_beta` | `4.0` | Beta parameter of the CPU-share Beta distribution. |
| `gpu_dispatch_fraction_mean` | `0.5` | Center of the dispatch fraction before uniform jitter and clamping. |
| `gpu_dispatch_fraction_jitter` | `0.1` | Uniform variation added around the dispatch center. |
| `gpu_dispatch_fraction_min` | `0.35` | Lower clamp for the sampled dispatch fraction. |
| `gpu_dispatch_fraction_max` | `0.65` | Upper clamp for the sampled dispatch fraction. |
| `cpu_matrix_base` | `512` | Base CPU matrix dimension before workload scaling. |
| `gpu_matrix_base` | `2048` | Base GPU matrix dimension before workload scaling. |
| `gpu_memory_base_mb` | `256` | Base synthetic device allocation before event scaling. |
| `gpu_memory_jitter_mb` | `512` | Maximum non-negative random addition to the base allocation before scaling. |
| `h2d_base_mb` | `64` | Base nominal host-to-device transfer volume. |
| `d2h_base_mb` | `16` | Base nominal device-to-host transfer volume. |
| `gpu_mode` | unset | Explicit `blocking`, `event_barrier`, or `async` policy. |
| `gpu_async` | `false` | Legacy policy flag used only when `gpu_mode` is absent. |
| `async_merge_ready` | `false` | If `true`, merge already-complete async events between CPU events instead of deferring all collection. Use a JSON boolean, not a string. |
| `reduce_work_s` | `0.01` | Synthetic event reduction delay. |
| `num_cpus_per_event` | `1.0` | Resource-request metadata in local output; not enforced. |
| `num_gpus_per_event` | `1.0` | Resource-request metadata in local output; not enforced. |

The expected CPU fraction of a Beta distribution is
`alpha / (alpha + beta)`. For example, `(8, 2)` is CPU-heavy, `(2, 8)` is GPU-heavy, and
`(4, 4)` is balanced on average.

### Bundled workloads

| File | Events | Implemented workload |
| --- | ---: | --- |
| `fixed_balanced_50.json` | 50 | Gaussian size, mean 2.5 s, standard deviation 0.7 s; balanced `(4, 4)` CPU share. |
| `fixed_cpu_heavy_50.json` | 50 | Gaussian size, mean 3.0 s, standard deviation 0.5 s; CPU-heavy `(8, 2)` share. |
| `fixed_gpu_heavy_50.json` | 50 | Gaussian size, mean 3.0 s, standard deviation 0.8 s; GPU-heavy `(2, 8)` share and larger GPU payloads. |
| `poisson_high_variance_100.json` | 100 | Gaussian size, mean 2.5 s, standard deviation 1.0 s; high-variance `(0.8, 0.8)` CPU share and memory allocation. |
| `mode_blocking.json` | 20 | Default workload with blocking synchronization. |
| `mode_event_barrier.json` | 20 | Default workload with within-event overlap. |
| `mode_async.json` | 20 | Default workload with cross-event pipelining. |

The `fixed_*` and `poisson_*` filenames are historical: their implemented size distribution is
Gaussian. Simload does not implement a Poisson distribution or an external Poisson arrival process.
To make every event the same nominal size, set `event_size_distribution` to `"fixed"`.

## Command-line reference

```text
simload [-c FILE | --config FILE] [-m MODE | --mode MODE]
        [-o FILE | --out FILE] [-d DIR | --out-dir DIR]
        [-n N | --num-cpus N] [-g N | --num-gpus N]
```

| Option | Behavior |
| --- | --- |
| `-c FILE`, `--config FILE` | Load a JSON object and apply it over the built-in defaults. |
| `-m MODE`, `--mode MODE` | Override the configuration with `blocking`, `event_barrier`, or `async`. |
| `-o FILE`, `--out FILE` | Write to this exact CSV path. Parent directories are created. |
| `-d DIR`, `--out-dir DIR` | Output directory when `--out` is absent; defaults to `simload_runs`. |
| `-n N`, `--num-cpus N` | Accepted for compatibility with the Ray program; has no scheduling effect locally. |
| `-g N`, `--num-gpus N` | Accepted for compatibility with the Ray program; local mode still uses the current CUDA device. |

Without `--out`, the filename is derived from the config stem:

- `--config config/example.json` writes `simload_runs/example.csv`;
- no `--config` writes `simload_runs/default_params.csv`.

The mode is not automatically appended to the filename. If you reuse one workload config with
several `--mode` values, provide distinct `--out` paths to prevent overwriting earlier runs.

## Python API

The same experiment can be run in Python. Start from `default_params()` so the API and CLI defaults
remain identical:

```python
from pathlib import Path

from teerex.simload import default_params, run_simulation

params = default_params()
params.update(
    {
        "n_events": 8,
        "mean_event_size_s": 1.0,
        "gpu_mode": "event_barrier",
    }
)

df = run_simulation(params)

Path("simload_runs").mkdir(exist_ok=True)
df.to_csv("simload_runs/python_api.csv", index=False)
```

`run_simulation()` returns the DataFrame but does not write it. It still requires CUDA and executes
the same local CPU/GPU workload as the command-line program.

## Reproducibility and scope

- Event specifications are deterministic for the same event id, complete sampling parameters, and
  seed or seeds. For policy-only comparisons, keep every sampling setting identical.
- Set the same `event_size_seed` and size-distribution parameters when event sizes must stay fixed
  while the base `seed` changes.
- NumPy and PyTorch payload data are not seeded, and measured timings and iteration counts depend on
  hardware, library versions, system load, and thermal state. Repeat comparisons on the same
  machine and retain each run separately.
- The first GPU event may include CUDA context, allocator, or kernel warm-up costs.
- A nominal work duration is a loop target, not an exact deadline. Whole matrix multiplications,
  setup, transfers, and synchronization can make measured runtimes longer.
- The model has one sequential event manager and one serial GPU worker. It does not report hardware
  utilization, model multi-GPU execution, validate memory capacity in advance, or predict a full
  Geant4 application's performance.

## Earlier Ray implementation

The earlier implementation remains available as `simload-ray` in
[`teerex/simload_ray.py`](teerex/simload_ray.py):

```bash
uv run simload-ray \
    --config config/fixed_balanced_50.json \
    --out simload_runs/ray_balanced.csv
```

The Ray runner initializes a 10 GB object store, so the environment must provide enough memory
(including shared memory in a container). It is useful when Ray scheduling effects are themselves
part of the experiment, but it is not a drop-in equivalent of the local model:

| `simload` | `simload-ray` |
| --- | --- |
| Splits CPU work before and after GPU dispatch. | Runs the whole CPU payload before GPU submission. |
| Supports `blocking`, `event_barrier`, and `async`. | Supports only the legacy blocking/async boolean. |
| Uses one local FIFO worker and the current CUDA device. | Uses Ray to schedule GPU tasks with configured CPU/GPU resource requests. |
| Records dispatch, CPU-phase, queue, wait, and detailed offset columns. | Writes a smaller timing schema; end-to-end timings may include Ray scheduling and worker overhead. |

The Ray CSV does not expose scheduler queue delay or explicit wait duration. Do not pass
`config/mode_event_barrier.json` to `simload-ray`; that implementation does not interpret `gpu_mode`.
Also use a separate output filename when comparing the two implementations.

## Repository layout

```text
teerex/simload.py       main local CPU/GPU experiment
teerex/simload_ray.py   earlier Ray-scheduled variant
teerex/analysis.py      CSV loading and plotting helpers
config/                 workload and synchronization examples
notebooks/simload.ipynb interactive comparison notebook
```
