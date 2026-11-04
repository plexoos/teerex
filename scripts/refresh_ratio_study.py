"""Rebuild the portable notebook evidence from a ratio-study-v3 combined CSV.

Run from the repository root. Raw measurements are read only; models and policy
choices are refitted separately for each geometry/resource regime.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from teerex import study


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs" / "study"
COHORTS = ("drich", "pfrich", "balanced", "gpu_heavy_n10", "gpu_heavy_n100")
WORKERS = (1, 2, 4, 8, 16, 20)
INFLIGHT = (1, 2, 4, 8, 16, 32, 64)


def load_events(path):
    """Validate the combined, already warm-up-filtered production export."""
    events = pd.read_csv(path)
    required = {"run_id", "event_id", "scenario", "condition_id", "replication",
                "split", "is_warmup", "geometry", "geometry_name", "case",
                "particle", "momentum_gev_c", "multiplicity", "num_photons",
                "num_gensteps", "clhep_seed", "gpu_seed", "gpu_offset", *study.PHASES}
    if not required.issubset(events):
        raise ValueError(f"missing production columns: {sorted(required - set(events))}")
    if events.empty or events[["run_id", "event_id"]].duplicated().any():
        raise ValueError("production events must have unique (run_id, event_id)")
    for name in ("event_id", "replication", "multiplicity", "num_photons", "num_gensteps"):
        events[name] = study._integer_column(events[name], name)
    expected = events.replication.map({0: "train", 1: "train", 2: "validation", 3: "test"})
    if expected.isna().any() or not expected.eq(events.split).all():
        raise ValueError("split must match train(0/1), validation(2), test(3)")
    if events.is_warmup.isna().any() or not events.is_warmup.eq(False).all():
        raise ValueError("combined export must contain only retained production events")
    values = events[list(study.PHASES)].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("phase durations must be finite and nonnegative")
    events["cohort"] = np.where(events["case"].eq("rich"), events.geometry_name, events["case"])
    if set(events.cohort) != set(COHORTS):
        raise ValueError("expected dRICH, pfRICH, balanced and both GPU-heavy production cohorts")
    for run_id, rows in events.groupby("run_id", sort=False):
        for name in ("replication", "condition_id", "scenario", "split", "cohort",
                     "clhep_seed", "gpu_seed", "gpu_offset"):
            if rows[name].nunique(dropna=False) != 1:
                raise ValueError(f"run {run_id} has inconsistent {name}")
        study._validate_recorded_timing(rows, run_id)
    study._validate_replication_seeds(events)
    counts = events.groupby(["condition_id", "replication"]).size().unstack("replication")
    if set(counts.columns) != {0, 1, 2, 3} or counts.isna().any().any():
        raise ValueError("every production condition needs all four replications")
    events["zero_photon"] = events.num_photons.eq(0)
    return events


def _json(path, value):
    path.write_text(json.dumps(study._clean_json(value), indent=2, allow_nan=False) + "\n")


def verify(destination=DESTINATION):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    for name, item in manifest["files"].items():
        data = (destination / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"snapshot checksum mismatch: {name}")
    return manifest


def build_snapshot(source, destination=DESTINATION, draws=64):
    source, destination = Path(source), Path(destination)
    events = load_events(source)
    destination.mkdir(parents=True, exist_ok=True)
    # Exact original bytes survive decompression; keep every column and timing tail.
    source_bytes = source.read_bytes()
    (destination / "events.csv.gz").write_bytes(gzip.compress(source_bytes, mtime=0))
    metrics, predictions, condition_metrics, policies, resources = [], [], [], [], []
    frozen, models, diagnostics = {}, {}, {}
    for index, cohort in enumerate(COHORTS):
        print(f"{cohort}: fitting and selecting on validation ({draws} draws)", flush=True)
        train, validation, test = study.split_by_replication(events.loc[events.cohort.eq(cohort)])
        bundle = study.fit_duration_models(train)
        models[cohort] = bundle.to_dict()
        diagnostics[cohort] = study._diagnostics(bundle)
        val_predictions, val_metrics = study.evaluate_models(bundle, validation, "validation")
        # Freeze the policy before evaluating held-out durations or schedules.
        val_stream = study._streams(validation)["balanced_mix"]
        samples = bundle.sample(val_stream, draws, np.random.default_rng(study.RANDOM_SEED + index))
        _, grid, chosen = study.select_posterior_policy(
            val_stream, samples, worker_counts=WORKERS, inflight_limits=INFLIGHT)
        test_stream = study._streams(test)["balanced_mix"]
        frozen[cohort] = {
            "configuration": chosen, "selected_on": "validation", "evaluated_on": "test",
            "criterion": "least workers, then in-flight events, within 5% of best validation median makespan",
            "streams": {name: frame[["run_id", "event_id"]].to_numpy().tolist()
                        for name, frame in [("validation", val_stream), ("test", test_stream)]},
        }
        test_predictions, test_metrics = study.evaluate_models(bundle, test, "test")
        metrics.extend([val_metrics.assign(cohort=cohort), test_metrics.assign(cohort=cohort)])
        predictions.append(test_predictions.assign(cohort=cohort))
        condition_metrics.append(study.summarize_condition_predictions(
            pd.concat([val_predictions, test_predictions], ignore_index=True)).assign(cohort=cohort))
        configurations = [study._config("blocking", 1, 1)]
        if chosen["configuration"] != configurations[0]["configuration"]:
            configurations.append(chosen)
        test_samples = bundle.sample(test_stream, draws, np.random.default_rng(study.RANDOM_SEED + 100 + index))
        comparisons = [study._simulate(test_stream, test_samples, configurations,
                        split="test", stream="balanced_mix", mode="independent", slowdown=factor)
                       for factor in (1.0, 1.5, 2.0)]
        joint = bundle.sample(test_stream, draws, np.random.default_rng(study.RANDOM_SEED + 200 + index),
                              mode="joint_block")
        comparisons.append(study._simulate(test_stream, joint, configurations, split="test",
                                            stream="balanced_mix", mode="joint_block"))
        comparisons.append(study._simulate(test_stream, test_stream[list(study.PHASES)].to_numpy(float),
                                            configurations, split="test", stream="balanced_mix"))
        test_results = study._policy_summary(pd.concat(comparisons, ignore_index=True))
        policies.extend([grid.assign(cohort=cohort), test_results.assign(cohort=cohort)])
        for config in configurations:
            timeline = study.replay(study._events(test_stream, test_stream[list(study.PHASES)].to_numpy(float)),
                                   **{key: config[key] for key in ("policy", "workers", "max_inflight")})
            resources.append({"cohort": cohort, **config, **study._timeline_metrics(timeline)})
        print(f"{cohort}: frozen {chosen['configuration']}", flush=True)
    pd.concat(metrics, ignore_index=True).to_csv(destination / "heldout-metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(destination / "test-predictions.csv.gz", index=False,
                                                    compression={"method": "gzip", "mtime": 0})
    pd.concat(condition_metrics, ignore_index=True).to_csv(destination / "heldout-condition-metrics.csv", index=False)
    pd.concat(policies, ignore_index=True).to_csv(destination / "policy-summary.csv", index=False)
    pd.DataFrame(resources).to_csv(destination / "resource-summary.csv", index=False)
    study.validate_blocking_trace(events).to_csv(destination / "blocking-trace-validation.csv", index=False)
    _json(destination / "frozen-policies.json", frozen)
    _json(destination / "duration-models.json", models)
    _json(destination / "training-diagnostics.json", diagnostics)
    names = ["events.csv.gz", "heldout-metrics.csv", "test-predictions.csv.gz",
             "heldout-condition-metrics.csv", "policy-summary.csv", "resource-summary.csv",
             "blocking-trace-validation.csv", "frozen-policies.json", "duration-models.json",
             "training-diagnostics.json"]
    manifest = {
        "study": "ratio-study-v3", "source": {"path": str(source.resolve()),
            "sha256": hashlib.sha256(source_bytes).hexdigest(), "bytes": len(source_bytes)},
        "events": len(events), "runs": events.run_id.nunique(),
        "conditions": events.condition_id.nunique(), "draws": draws,
        "random_seed": study.RANDOM_SEED, "worker_counts": WORKERS, "inflight_limits": INFLIGHT,
        "files": {name: {"sha256": hashlib.sha256((destination / name).read_bytes()).hexdigest(),
                          "bytes": (destination / name).stat().st_size} for name in names},
        "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [Path(__file__), ROOT / "teerex" / "study.py",
                                     ROOT / "teerex" / "bayes.py", ROOT / "teerex" / "replay.py"]},
    }
    _json(destination / "manifest.json", manifest)
    verify(destination)
    print(f"Verified {len(events):,} events and {len(names)} evidence files in {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="all-production-events.csv")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.destination)
        print("Snapshot checksums verified")
    elif args.source is None:
        parser.error("provide a source CSV or --verify")
    else:
        build_snapshot(args.source, args.destination, args.draws)
