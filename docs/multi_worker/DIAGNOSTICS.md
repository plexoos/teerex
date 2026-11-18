# Cleanup timing diagnostics and separate confirmation

The first-seed dRICH 10-primary serial control contains a large wall-time disturbance. A later repeat reproduces its event counts and retained GPU-hit bytes without the disturbance. This supports treating the original 24.705× serial-relative speedup as distorted by its baseline, while preserving that run in the primary dataset. The blocking cause is not isolated.

## Primary measurements

Across 56,000 retained events, 64 intervals in five serial runs have cleanup wall time above 50 ms and event-thread CPU time below 5% of that wall interval. This clock-based diagnostic does not filter events or identify the blocking operation.

| Primary run | Flagged events | Flagged cleanup wall, s | Retained completion window, s | Flagged wall / window |
|---|---:|---:|---:|---:|
| `argon_balanced_t1_r1` | 1 | 0.288 | 223.270 | 0.129% |
| `argon_gpu_n10_t1_r1` | 3 | 1.034 | 159.037 | 0.650% |
| `argon_gpu_n10_t1_r2` | 6 | 2.282 | 161.312 | 1.415% |
| `drich_n100_t1_r2` | 4 | 0.803 | 94.525 | 0.849% |
| `drich_n10_t1_r0` | 50 | 23.527 | 33.305 | 70.643% |

For `drich_n10_t1_r0`, 500 retained completions take 33.305 s (15.013 events/s). Total cleanup wall time is 23.872 s, including 23.527 s in the 50 flagged events. Most flagged events are 303–359, plus event 26. The longest intervals are roughly 0.6–0.8 s wall with about 1 ms or less of event-thread CPU; nearly all of that interval holds the GPU gate.

The measured cleanup boundary includes host hit copying, output, QEvt clearing, and event reset. In the frozen source, [`SEvt::endOfEvent`](https://github.com/BNLNPPS/simphony/blob/2c878526827c8d02aa48dbc8a83063c19e593272/sysrap/SEvt.cc#L1807) also calls `SaveRunMeta` on every event, so filesystem work is inside this boundary. The available clocks cannot assign the delay to an individual operation.

Five-second telemetry during the affected process shows dirty pages being written back while GPU samples are mostly idle. The samples below describe the process interval, including startup/shutdown, rather than exact individual cleanup intervals. Times below use America/New_York (UTC−05:00); the CSV retains UTC timestamps.

| America/New_York on 2026-11-13 | Dirty pages, GiB | Writeback, MiB | GPU 0 utilization sample, % |
|---|---:|---:|---:|
| 19:24:14 | 5.239 | 0.0 | 0 |
| 19:24:19 | 5.952 | 0.0 | 5 |
| 19:24:24 | 4.908 | 73.1 | 0 |
| 19:24:29 | 3.695 | 53.3 | 0 |
| 19:24:34 | 2.445 | 83.0 | 0 |
| 19:24:39 | 1.162 | 107.3 | 0 |
| 19:24:44 | 0.064 | 41.3 | 0 |

This is a temporal association with writeback, not proof of causation or proof that the writes came from another process. The collection itself also writes substantial output between measurements. No physics or runtime setting was changed to remove the observation.

## Separate repeat

After the primary matrix and exports completed, the affected seed was repeated at 1/2/8 workers with the same build, GPU, affinity, input hashes, seeds, 516 raw events, and 16-event warm-up. All three repeats reproduce the corresponding primary counts and retained hit hashes. The serial comparison includes raw counts; MT comparisons cover retained counts.

| Workers | Primary events/s | Repeat events/s | Repeat / repeat serial | Repeat flagged events | Repeat total cleanup wall, s |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.013 | 55.108 | 1.000 | 0 | 0.269 |
| 2 | 106.024 | 104.521 | 1.897 | 0 | 0.306 |
| 8 | 370.900 | 370.248 | 6.719 | 0 | 0.318 |

The repeat serial rate is 55.108 events/s, compared with 15.013 originally; eight-worker performance is nearly unchanged at 370.248 versus 370.900 events/s. The repeat 8/serial ratio is 6.719 and 8/2 is 3.542. No repeat event meets the diagnostic threshold. Confirmation telemetry samples show 0.009–0.173 GiB dirty pages and zero sampled writeback. These later conditions are observational evidence, not an isolated causal experiment.

The primary dataset is unchanged. The other three primary dRICH 10-primary seeds have 8/serial ratios of 6.410, 6.477, and 6.933. The affected seed remains visible in every primary table and plot. A clean repeat neither invalidates the original observation nor establishes its cause.

[Flagged event records](../../data/multi_worker/slow-cleanup.csv) · [Primary telemetry](../../data/multi_worker/telemetry.csv) · [Confirmation metrics](../../data/multi_worker/confirmation-summary.csv) · [Confirmation telemetry](../../data/multi_worker/confirmation-telemetry.csv) · [Confirmation verification](../../data/multi_worker/confirmation-verification.json)
