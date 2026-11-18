# Charged events on native GPU threads

The [Marimo notebook](../../notebooks/timelines_multi_gpu.py) follows optical payloads through a single `simg4ox` process on npps0. Four Geant4 workers prepare charged events and each submits its payload at the event boundary. One native thread per visible GPU takes the next available payload. The notebook reads the committed timing data and runs on a CPU-only machine.

```bash
uv run marimo edit notebooks/timelines_multi_gpu.py
```

The [browser version](https://plexoos.github.io/teerex/timelines_multi_gpu/)
runs interactively without a local Python installation. Its export bundles
the manifest, event timings, run summaries, condition summaries, and matched
seed-pair summaries; the notebook loads those files into the browser's filesystem.

The five editable inputs select the event CSV, run summary, condition summary,
paired summary, and manifest. Defaults are `data/multi_gpu/...`
relative to the project folder locally and `public/...` relative to the notebook
page in the browser. Press Enter or leave a field to load its selected file.
Use files from the same measurement bundle; browser paths must identify
web-accessible files.

## What was measured

Each of seven workloads ran with one and two RTX 4090s, using four matched Geant4/GPU seed pairs. The pfRICH and dRICH sources used 10 or 100 electrons at 20 GeV/c. The DUNE liquid-argon mock sources used 10 electrons at 10 MeV, 100 at 1 MeV, or 65,536 at 1 keV kinetic energy. In all 28 matched pairs, per-event counts and the merged hit arrays agreed across GPU configurations.

The stacked timelines show each workload with two GPUs and the first shared seed pair. Each timeline opens on the first ten event IDs; the range slider covers all 72 events on that run's clock, including the warm-up. Run throughput counts the 64 completions after the eighth event **finishes**. The configuration summary covers the 28 two-GPU runs: 2,016 raw events and 1,792 retained completions. Condition throughput and paired speedup are geometric means across the four seed pairs; the speedup comparison uses the matching one-GPU controls as its baseline. [Per-run measurements](../../data/multi_gpu/run-summary.csv) · [Condition summaries](../../data/multi_gpu/condition-summary.csv) · [Matched seed pairs](../../data/multi_gpu/paired-summary.csv)

Each timeline is followed by CPU and GPU duration histograms, with a separate curve for each seed pair and a statistics table beneath each panel. These use only two-GPU runs and exclude the first eight completions of each run. CPU duration sums preparation and GPU-thread cleanup, excluding handoff/queue waiting; GPU duration is host-observed service. Each panel uses 25 shared bins over the pooled 1st–99th percentile range, padded by 25% on each side and bounded at zero. As in `timelines_single_worker`, the CPU panels for balanced argon (65,536 × 1 keV) and GPU-heavy argon (100 × 1 MeV) use the 97th percentile at the upper end. The tables report shown/total event counts and untrimmed mean, population standard deviation, and coefficient of variation for positive, finite durations.

## Reading the results

- Blue shows Geant4 preparation. Grey shows the producer's recorded handoff interval, including waiting for queue space and submission overhead. After handoff, the producer can return to tracking while the submitted event waits for or runs on a GPU, so that later payload time is not shaded as producer waiting.
- Each timeline has four Geant4 producer lanes and two physical GPU lanes. Each GPU lane shows host-observed service in orange and CPU cleanup in purple, using the GPU-lane colors from `timelines_multi_worker`. Service includes host work and device waiting. "Gate held for cleanup" refers to the individual GPU's reserved interval before its service thread can take another event; that cleanup still contributes to the CPU histogram.
- Each GPU lane follows the physical UUID. The single-GPU control uses physical GPU 1, which CUDA numbers as local device 0 in that process. GPU service and cleanup are measured on the host clock.
- Across the DUNE mock workloads, two GPUs improved post-warm-up throughput by **1.93–1.97×**, and full-job speed by **1.76–1.78×**. A separate 12-event Nsight capture measured overlapping optical kernels for **1.576 s of a 1.813 s kernel window**. [Device measurements](../../data/multi_gpu/profile/kernel-overlap.json)
- Geant4 event preparation for the 100-electron sources averaged **7.23 ms in pfRICH** and **168 ms in dRICH** with two GPUs. Their mean photon counts were much closer, about 112,000 and 133,000. For dRICH the range of paired speedups includes a 1.33× result with unusually slow CPU preparation on one GPU. An independent repeat of that pair gave 1.02×. The notebook's four-seed summary uses the original runs.

The [manifest](../../data/multi_gpu/manifest.json) records the measured source revision and data provenance. The bundle is stored in [`data/multi_gpu/`](../../data/multi_gpu/). To rebuild it from the original archive, run:

```bash
python docs/multi_gpu/assemble.py /workspaces/teerex-study/charged-native-multi-gpu-v1
```
