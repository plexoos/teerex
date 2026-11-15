# Measured multithreaded Geant4 evidence

This portable snapshot accompanies [mt_timelines.py](../../notebooks/mt_timelines.py).
It preserves the completed `charged-mt-study-v1` campaign's exported CSV bytes,
including every warm-up event and timing tail. No slow events are removed.
Read the [findings and scaling plots](FINDINGS.md) for the results, and the
[cleanup diagnostics](DIAGNOSTICS.md) for the anomalous serial control and repeat.

| Coverage | Definition |
|---|---|
| RICH workloads | pfRICH/dRICH, 10/100 electrons per event, 20 GeV/c |
| DUNE mock workloads | 65,536 × 1 keV; 10 × 10 MeV; 100 × 1 MeV kinetic energy |
| Execution | Serial control and 2/4/8 Geant4 workers; one ordered GPU gate |
| Seeds | Four CLHEP/GPU pairs: 50111/2100011, 60223/2200033, 70339/2300051, 80447/2400077 |
| Per run | 516 raw events, 16 warm-up events, 500 retained events |
| Total | 112 runs; 57,792 raw and 56,000 retained events |

All runs use one Release build of Simphony merge
`2c878526827c8d02aa48dbc8a83063c19e593272` (PR 462) and Teerex
`81de2cb8b85082954731075d6aacf56650d1c30e`. Geant4 used physical CPU cores
2–17 on a Xeon w7-3445 and one RTX 4090 selected by UUID. The host was shared.
The manifest records the exact commits, device, coverage, and file hashes.

## Files and reuse

- `events-raw.csv.gz`: all measured rows and columns, with original run-local
  offsets, event/worker identifiers, counts, seeds, and the `is_warmup` flag.
- `run-summary.csv`: 112 per-process summaries, including completion and batch
  rates, serial-relative and two-worker-relative speedups, interval occupancies,
  queue waits, process wall time, and cleanup-stall counts.
- `summary.csv`: medians and ranges across the four runs per workload/worker setting.
- `sensitivity.csv`: warm-up cutoffs 8/16/32 crossed with tail exclusions 0/8/16.
- `slow-cleanup.csv`: the 64 retained diagnostic intervals; these remain in all results.
- `confirmation-summary.csv`: a separate three-run dRICH repeat, excluded from
  the primary matrix and notebook's main statistics.
- `verification.json`: the original campaign's verification record.
- `figures/`: scaling plots relative to serial and two-worker MT controls.
- `telemetry.csv` and `confirmation-telemetry.csv`: primary and repeat GPU/host
  samples, including dirty pages and writeback counters.
- `confirmation-verification.json`: the separate repeat's verification record.
- `warmup-differences.json`: raw initial-event count differences across MT workers.
- `manifest.json`: SHA256/size for the copied evidence files and source provenance.

From the repository root:

```bash
PYTHONPATH=. uv run --with plotly marimo edit notebooks/mt_timelines.py
PYTHONPATH=. uv run --with plotly marimo export html notebooks/mt_timelines.py \
  -o /tmp/mt_timelines.html
uv run pytest tests/test_mt_timeline_snapshot.py tests/test_timeline_analysis.py
```

The notebook reads `docs/mt-study` relative to its repository. Set
`TEEREX_MT_DATA_DIR=/path/to/bundle` to use another compatible export containing
`events-raw.csv.gz`, `run-summary.csv`, `summary.csv`, and `sensitivity.csv`.
The current notebook expects the same seven workload keys, worker grid, zero-based
seed indices 0–3, and 16-event warm-up convention. Preserve separate `run_id` and
`scenario` values for independent process clocks. Recompute all summary tables
when replacing raw measurements; do not combine fresh events with stale summaries.
The notebook displays the saved analysis rather than regenerating those tables.

Timeline panels show all 516 recorded events, from event 0 through the final
cleanup, including the 16 warm-up events. Each process starts its display clock
at its earliest recorded CPU-generation start. No events or phase durations are
cut. Drag horizontally to zoom into startup or later events, double-click to
restore the full run, and hover for exact event timing. Scaling summaries and
duration distributions continue to use the 500 retained events per run.

The HTML export is a snapshot; changing notebook controls requires running
Marimo. This commit does not add the notebook to the GitHub Pages build.

The snapshot needs no CUDA runtime or original hit arrays. Its tests verify
the bundled bytes, full event/worker coverage, retained MT count equivalence,
and completion-rate/diagnostic arithmetic. The original campaign also checked
runtime/input/output hashes and retained GPU-hit bytes across worker counts;
repeating those checks requires the full external archive. The JSON verification
record reports that earlier validation, not a new validation of absent hit arrays.

## Definitions and interpretation

Completion throughput is `500 / (event-515 GPU release - event-15 GPU release)`.
GPU service, GPU-gate occupancy, CPU coverage, and CPU/GPU overlap are interval
unions clipped to that same completion window, including overlapping neighboring
raw events. CPU generation and cleanup use wall clocks; thread CPU counters cover
those two phases and exclude host work inside the GPU call. Host-observed GPU
service includes staging, transfers, and synchronization. GPU-gate occupancy
also includes locked cleanup; neither measures hardware SM utilization.

At eight workers, median serial-relative speedups are 6.86× and 7.50× for
100-primary pfRICH and dRICH. Balanced DUNE gains 1.64× and saturates by two
workers; GPU-heavy DUNE gains about 1–4%. These are measured synchronous workers
with serialized GPU launches, not asynchronous-policy replay predictions.

The first-seed dRICH 10-primary serial control has 50 long cleanup intervals
with little thread CPU, inflating its eight-worker ratio to 24.71×. Across the
whole matrix, 64 intervals in five serial runs exceed 50 ms cleanup wall with
thread CPU below 5% of that wall interval. All are retained and flagged.
The separate repeat reproduces each corresponding primary history without
flagged stalls: serial/2/8-worker rates are 55.108/104.521/370.248 events/s,
and eight-worker/serial speedup is 6.72×. The blocking cause remains unisolated.

Serial and MT Geant4 use different RNG initialization paths; matching seed labels
do not imply identical serial/MT event histories. Retained MT histories do match
across 2/4/8 workers. Four seed pairs provide descriptive run ranges, not precise
confidence intervals. The smallest pfRICH windows are only 0.304–0.767 seconds.
The earlier serial study used the other GPU; this campaign's fresh serial controls
support its scaling comparisons.
