Teerex studies how to batch and dispatch event workloads across CPUs and a GPU.
It combines a synthetic scheduling experiment, measured Simphony timelines,
Bayesian phase-duration models, and offline policy replay.

The current question is how much CPU concurrency and work in flight are needed
to feed the GPU while controlling queueing and completion latency. The study now
uses the ratio-study-v3 production sample: 22,000 retained events across dRICH,
pfRICH, balanced argon, and two GPU-heavy argon conditions. Each workload has a
separate fitted model and validation-selected policy. Predicted asynchronous
gains remain estimates to validate in an executor.

## Run all notebooks

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run marimo run notebooks/
```

This opens a gallery of all four notebooks in one server, using the shared
project environment, including Plotly. If that environment is already activated,
you can use `marimo run notebooks/` directly. Restart an existing server after
syncing dependencies.

To edit the main study notebook, run `uv run marimo edit notebooks/simload.py`.

Each timeline notebook has editable data-file inputs. Defaults use `data/...`
paths relative to the project folder locally and `public/...` paths relative
to the notebook page on GitHub Pages. Press Enter or leave an input to load the
selected file. For the multi-worker and multi-GPU notebooks, select event and
summary files from the same measurement bundle. Local absolute paths are also
accepted; browser inputs must point to web-accessible files.

Notebook datasets live in `data/single_worker/`, `data/multi_worker/`, and
`data/multi_gpu/`. Simload shares the single-worker dataset. Study writeups
remain in `docs/`.

The [main notebook](notebooks/simload.py) tells the study's story and provides
comparison plots, model checks, timelines, and interactive replay. Its bundled
[data snapshot](docs/single_worker/README.md) works without the external study archive
or a GPU. Running the real synthetic workload requires CUDA-enabled PyTorch
and an NVIDIA GPU.

To rebuild the notebook's evidence from the combined production CSV:

```bash
PYTHONPATH=. uv run python scripts/refresh_ratio_study.py \
  ../teerex-study/ratio-study-v3/all-production-events.csv
PYTHONPATH=. uv run python scripts/refresh_ratio_study.py --verify
```

## Documentation

The [Geant4 CPU/GPU timelines: single worker](notebooks/timelines_single_worker.py)
notebook shows seven electron workloads, with configuration summaries, event
timelines, and CPU/GPU duration histograms across four seed pairs.

```bash
uv run marimo edit notebooks/timelines_single_worker.py
```

The [Geant4 CPU/GPU timelines: multiple workers](notebooks/timelines_multi_worker.py) notebook compares
measured 1/2/4/8-worker runs across seven workloads and four seed pairs. Its
[bundled CSV snapshot](docs/multi_worker/README.md) preserves all 57,792 raw events,
56,000 retained events, run summaries, and timing caveats without requiring
the external measurement archive or a GPU.
The [MT findings and scaling plots](docs/multi_worker/FINDINGS.md) explain the
measured gains, GPU-gate saturation, and the anomalous serial control.
The [browser version](https://plexoos.github.io/teerex/timelines_multi_worker/) runs
interactively without a local Python installation.

```bash
uv run marimo edit notebooks/timelines_multi_worker.py
```

The [Geant4 CPU/GPU timelines: multiple GPUs](notebooks/timelines_multi_gpu.py) notebook follows
charged events from four Geant4 workers to one or two RTX 4090s in a single
`simg4ox` process. Explore the seven electron workloads and four matched seed
pairs in its [bundled timing data](docs/multi_gpu/README.md).

```bash
uv run marimo edit notebooks/timelines_multi_gpu.py
```

The [multi-GPU browser version](https://plexoos.github.io/teerex/timelines_multi_gpu/)
runs interactively with its bundled timing data. Publishing executes all four
notebooks and verifies the study evidence before exporting and deploying the site.

The [published gallery](https://plexoos.github.io/teerex/) uses marimo's static
GitHub Pages card template and lists the three timeline notebooks. Their URLs
match their filenames: `timelines_single_worker/`, `timelines_multi_worker/`, and
`timelines_multi_gpu/`. Older `ratio_timelines/` and `mt_timelines/` links redirect
to the corresponding notebooks. Simload remains available at `simload/` and is
tested and exported, but does not appear in the gallery.

- [Current study: measurements, results, and next questions](docs/study.md)
- [Bayesian model, replay semantics, and study CLI](docs/bayesian-dispatch.md)
- [Synthetic Simload: quick start, modes, configuration, CSV, and Ray variant](docs/simload.md)
- [Versioned evidence and provenance](docs/single_worker/README.md)

To analyze a complete measured-run archive:

```bash
uv run study --root /path/to/teerex-study/drich-study-v2 --draws 64 \
  --workers 1 2 4 8 16 20 --inflight-limits 1 2 4 8 16 32 64
```

`--root` points to the archive containing `runs/<run_id>/`. `--draws` controls the
posterior sampling budget; `--workers` and `--inflight-limits` define the simulated
CPU worker counts and limits on unfinished events to compare. The two resource
lists form a grid, so larger lists increase analysis time. Run
`uv run study --help` for defaults and examples, or see the
[argument reference](docs/bayesian-dispatch.md#study-cli-and-selection-procedure)
for input requirements and replay semantics.

`make help` lists development commands; `make test` runs the test suite.
