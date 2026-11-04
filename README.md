Teerex studies how to batch and dispatch event workloads across CPUs and a GPU.
It combines a synthetic scheduling experiment, measured Simphony timelines,
Bayesian phase-duration models, and offline policy replay.

The current question is how much CPU concurrency and work in flight are needed
to feed the GPU while controlling queueing and completion latency. The study now
uses the ratio-study-v3 production sample: 22,000 retained events across dRICH,
pfRICH, balanced argon, and two GPU-heavy argon conditions. Each workload has a
separate fitted model and validation-selected policy. Predicted asynchronous
gains remain estimates to validate in an executor.

## Start with the study notebook

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run marimo edit notebooks/simload.py
```

The [main notebook](notebooks/simload.py) tells the study's story and provides
comparison plots, model checks, timelines, and interactive replay. Its bundled
[data snapshot](docs/study/README.md) works without the external study archive
or a GPU. Running the real synthetic workload requires CUDA-enabled PyTorch
and an NVIDIA GPU.

To rebuild the notebook's evidence from the combined production CSV:

```bash
PYTHONPATH=. uv run python scripts/refresh_ratio_study.py \
  ../teerex-study/ratio-study-v3/all-production-events.csv
PYTHONPATH=. uv run python scripts/refresh_ratio_study.py --verify
```

## Documentation

- [Current study: measurements, results, and next questions](docs/study.md)
- [Bayesian model, replay semantics, and study CLI](docs/bayesian-dispatch.md)
- [Synthetic Simload: quick start, modes, configuration, CSV, and Ray variant](docs/simload.md)
- [Versioned evidence and provenance](docs/study/README.md)

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
