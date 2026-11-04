"""Measured-run analysis and explicitly counterfactual dispatch experiments.

Run with ``python -m teerex.study --root /workspaces/teerex-study --draws 64``.
Raw runs are immutable inputs. Every invocation creates a new artifact directory.
Only replications 0/1 train the transforms/models; 2 selects the finite policy;
3 contributes only integrity checks until the selected policy has been frozen;
its outcomes are not used for fitting or selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .bayes import BayesianLinearRegression
from .replay import replay


PHASES = ("cpu_pre_runtime_s", "gpu_runtime_s", "cpu_post_locked_runtime_s",
          "cpu_post_unlocked_runtime_s")
TIMER_FLOOR_S = 1e-9
RANDOM_SEED = 20261031
DEFAULT_WORKERS = (1, 2, 4, 8)
DEFAULT_INFLIGHT = (1, 2, 4, 8, 16, 32)


def _integer_column(series, name):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError(f"{name} must contain finite nonnegative integers")
    return values.astype(np.int64)


def _checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_recorded_inputs(directory, metadata):
    # Raw run-local inputs remain auditable even when historical executables
    # were rebuilt. External binary/geometry archives belong to study preflight.
    paths = {"csv_sha256": directory / "events.csv", "macro_sha256": directory / "run.mac"}
    if "config_sha256" in metadata:
        configs = list((directory / "config").glob("*.json"))
        if len(configs) != 1:
            raise ValueError(f"run {directory.name}: cannot identify recorded configuration")
        paths["config_sha256"] = configs[0]
    for key, path in paths.items():
        if key in metadata and (not path.is_file() or _checksum(path) != metadata[key]):
            raise ValueError(f"run {directory.name}: recorded {key} checksum mismatch")


def _validate_replication_seeds(frame):
    for columns, label in [(["clhep_seed"], "CLHEP seed"),
                           (["gpu_seed", "gpu_offset"], "GPU stream")]:
        values = frame[columns]
        if values.isna().all().all():
            continue  # Legacy ingestion can omit streams; recorded streams must agree.
        if values.isna().any().any():
            raise ValueError(f"incomplete recorded {label} metadata")
        for column in columns:
            _integer_column(frame[column], column)
        owners = {}
        for replication, group in frame.groupby("replication"):
            for stream in set(map(tuple, group[columns].to_numpy())):
                if stream in owners and owners[stream] != replication:
                    raise ValueError(f"{label} reused across independent replications")
                owners[stream] = replication


def _validate_recorded_settings(frame, metadata, manifest, run_id):
    if "random_seed" in manifest and "clhep_seed" in metadata:
        if manifest["random_seed"] != metadata["clhep_seed"]:
            raise ValueError(f"run {run_id}: recorded random seed disagrees with metadata")
    for recorded, configured in (("primary_particle", "particle"),
                                 ("primary_momentum_gev_c", "momentum_gev_c"),
                                 ("primary_multiplicity", "multiplicity")):
        expected = metadata[configured]
        sources = []
        if recorded in frame:
            sources.append(frame[recorded].to_numpy())
        if recorded in manifest:
            sources.append(np.asarray([manifest[recorded]]))
        for values in sources:
            matches = (np.asarray(values, dtype=str) == str(expected) if configured == "particle"
                       else np.isclose(np.asarray(values, dtype=float), float(expected), rtol=1e-12, atol=1e-12))
            if not np.all(matches):
                raise ValueError(f"run {run_id}: recorded {recorded} disagrees with run metadata")


def _validate_recorded_timing(frame, run_id):
    def identity(actual, expected, description):
        if not np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float),
                           rtol=1e-7, atol=1e-8):
            raise ValueError(f"run {run_id}: inconsistent {description}")

    for start, end, duration in (
        ("cpu_pre_start_time_offset_s", "cpu_pre_end_time_offset_s", PHASES[0]),
        ("gpu_start_time_offset_s", "gpu_end_time_offset_s", PHASES[1]),
        ("gpu_end_time_offset_s", "gpu_release_time_offset_s", PHASES[2]),
        ("gpu_release_time_offset_s", "cpu_post_end_time_offset_s", PHASES[3]),
        ("gpu_submit_time_offset_s", "gpu_start_time_offset_s", "gpu_queue_delay_s"),
        ("gpu_start_time_offset_s", "gpu_release_time_offset_s", "gpu_gate_runtime_s"),
    ):
        if {start, end, duration}.issubset(frame):
            if (frame[end] < frame[start] - 1e-9).any():
                raise ValueError(f"run {run_id}: backwards {start}/{end}")
            identity(frame[end] - frame[start], frame[duration], duration)
    for left, right in (("cpu_pre_end_time_offset_s", "gpu_submit_time_offset_s"),
                        ("gpu_end_time_offset_s", "cpu_post_start_time_offset_s"),
                        ("start_offset_s", "cpu_pre_start_time_offset_s"),
                        ("end_offset_s", "cpu_post_end_time_offset_s")):
        if {left, right}.issubset(frame):
            identity(frame[left], frame[right], left + "/" + right)
    if "cpu_post_runtime_s" in frame:
        identity(frame.cpu_post_runtime_s, frame[PHASES[2]] + frame[PHASES[3]], "post phase sum")
    if "gpu_gate_runtime_s" in frame:
        identity(frame.gpu_gate_runtime_s, frame[PHASES[1]] + frame[PHASES[2]], "GPU gate sum")
    if {"start_offset_s", "end_offset_s", "gpu_queue_delay_s"}.issubset(frame):
        identity(frame.end_offset_s - frame.start_offset_s,
                 frame[list(PHASES)].sum(axis=1) + frame.gpu_queue_delay_s, "event phase sum")
        ordered = frame.sort_values("event_id")
        if (ordered.start_offset_s.to_numpy()[1:] < ordered.end_offset_s.to_numpy()[:-1] - 1e-9).any():
            raise ValueError(f"run {run_id}: overlapping blocking events")


def load_runs(root, arm="physics_production") -> pd.DataFrame:
    """Load complete recorded runs without joining independent time origins.

    Warm-up rows and zero-work rows remain in the result. Invalid complete runs
    are rejected rather than silently excluded; failed runs and other arms are
    excluded by their explicit metadata. ``scenario`` is unique per run while
    ``source_scenario`` preserves the producer label. Raw files are never edited.
    """
    root = Path(root)
    frames, run_index = [], []
    for metadata_path in sorted((root / "runs").glob("*/run.json")):
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("status") != "complete" or metadata.get("arm") != arm
                or not metadata.get("record_timing", True)):
            continue
        directory = metadata_path.parent
        run_id = str(metadata.get("run_id", directory.name))
        if run_id != directory.name:
            raise ValueError(f"run_id does not match directory: {directory}")
        paths = [directory / name for name in
                 ("events.csv", "events.manifest.json", "simg4ox.stdout.log")]
        if not all(path.is_file() for path in paths):
            raise ValueError(f"complete run {run_id} is missing its CSV, manifest, or stdout")
        manifest = json.loads(paths[1].read_text())
        _verify_recorded_inputs(directory, metadata)
        frame = pd.read_csv(paths[0])
        required = {"event_id", "num_photons", "num_gensteps", *PHASES}
        if not required.issubset(frame):
            raise ValueError(f"run {run_id} is missing columns {sorted(required - set(frame))}")
        if frame.empty or len(frame) != int(metadata["events"]):
            raise ValueError(f"run {run_id} has an incomplete event count")
        if "event_count" in manifest and int(manifest["event_count"]) != len(frame):
            raise ValueError(f"run {run_id} manifest event count disagrees with CSV")
        for column in ("event_id", "num_photons", "num_gensteps"):
            frame[column] = _integer_column(frame[column], column)
        if frame.event_id.duplicated().any():
            raise ValueError(f"duplicate event_id in run {run_id}")
        for column in frame:
            if column.endswith("_runtime_s") or column == "gpu_queue_delay_s":
                values = pd.to_numeric(frame[column], errors="raise").to_numpy(float)
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"run {run_id}: {column} must be finite and nonnegative")
            if column.endswith("offset_s") and not np.isfinite(pd.to_numeric(frame[column])).all():
                raise ValueError(f"run {run_id}: invalid {column}")
        for name in ("particle", "momentum_gev_c", "multiplicity", "replication"):
            if name not in metadata:
                raise ValueError(f"run {run_id} is missing {name} metadata")
            frame[name] = metadata[name]
        _validate_recorded_settings(frame, metadata, manifest, run_id)
        _validate_recorded_timing(frame, run_id)
        if int(metadata["replication"]) != metadata["replication"] or metadata["replication"] not in (0, 1, 2, 3):
            raise ValueError("replication must be 0, 1, 2, or 3")
        for name in ("clhep_seed", "gpu_seed", "gpu_offset", "gpu_uuid", "arm"):
            frame[name] = metadata.get(name)
        frame["run_id"] = run_id
        frame["geometry"] = manifest.get("geometry", metadata.get("geometry", "fixed_geometry"))
        frame["source_scenario"] = frame.get("scenario", pd.Series("simg4ox_blocking", index=frame.index))
        frame["scenario"] = run_id + "::" + frame.source_scenario.astype(str)
        frame["condition_id"] = (str(metadata["particle"]) + "|p="
                                 + format(float(metadata["momentum_gev_c"]), "g") + "|n="
                                 + str(int(metadata["multiplicity"])))
        warmup = int(metadata.get("warmup_events", 5))
        if warmup < 0 or warmup >= len(frame):
            raise ValueError(f"run {run_id} needs at least one event after warm-up")
        frame["event_ordinal"] = np.arange(len(frame))
        frame["is_warmup"] = frame.event_ordinal < warmup
        frame["zero_photon"] = frame.num_photons.eq(0)
        frames.append(frame)
        run_index.append({**metadata, "manifest": manifest,
                          "files": {path.name: {"path": str(path.resolve()), "sha256": _checksum(path)}
                                    for path in [metadata_path, *paths]}})
    if not frames:
        raise ValueError(f"no complete recorded {arm} runs under {root / 'runs'}")
    result = pd.concat(frames, ignore_index=True)
    if result[["run_id", "event_id"]].duplicated().any():
        raise ValueError("(run_id, event_id) must be unique")
    _validate_replication_seeds(result)
    result.attrs["run_index"] = run_index
    return result


def split_by_replication(frame):
    """Return train(0/1), validation(2), test(3), excluding marked warm-up."""
    if frame.groupby("run_id").replication.nunique().gt(1).any():
        raise ValueError("a run cannot cross replication splits")
    if not set(frame.replication.unique()).issubset({0, 1, 2, 3}):
        raise ValueError("unexpected replication; expected 0/1/2/3")
    retained = frame.loc[~frame.is_warmup].copy()
    # Keep audit records on the loaded dataset, not every pandas view/Series.
    retained.attrs = {}
    return tuple(retained.loc[retained.replication.isin(replications)].copy().reset_index(drop=True)
                 for replications in ([0, 1], [2], [3]))


class FeatureTransform:
    """Training-only source features; generated counts enter at submission.

    No feature is selected by inspecting arbitrary numeric columns. In
    particular hits, clocks and future durations cannot enter the design matrix.
    The intercept is supplied here, as required by BayesianLinearRegression.
    """
    def __init__(self, stage, numeric, categories, means, scales):
        self.stage = stage
        self.numeric = list(numeric)
        self.categories = {name: list(values) for name, values in categories.items()}
        self.means = np.asarray(means, dtype=float)
        self.scales = np.asarray(scales, dtype=float)
        self.feature_names = (["intercept"] + ["log1p_" + name for name in self.numeric]
                              + [name + "=" + value for name, values in self.categories.items() for value in values])

    @staticmethod
    def _numeric(frame, names):
        values = frame[list(names)].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("source/count predictors must be finite and nonnegative")
        return np.log1p(values)

    @classmethod
    def fit(cls, frame, stage="cpu_pre"):
        if stage not in ("cpu_pre", "gpu", "post") or frame.empty:
            raise ValueError("fit needs training rows and stage cpu_pre/gpu/post")
        numeric = ["momentum_gev_c", "multiplicity"]
        if stage != "cpu_pre":
            numeric += ["num_photons", "num_gensteps"]
        raw = cls._numeric(frame, numeric)
        means, scales = raw.mean(axis=0), raw.std(axis=0)
        scales[scales == 0] = 1
        categories = {"particle": sorted(frame.particle.astype(str).unique())}
        if "geometry" in frame:
            categories["geometry"] = sorted(frame.geometry.astype(str).unique())
        return cls(stage, numeric, categories, means, scales)

    def transform(self, frame):
        values = (self._numeric(frame, self.numeric) - self.means) / self.scales
        columns = [np.ones((len(frame), 1)), values]
        for name, categories in self.categories.items():
            labels = frame[name].astype(str)
            unknown = set(labels) - set(categories)
            if unknown:
                raise ValueError(f"unseen {name}: {sorted(unknown)}")
            columns.append(np.column_stack([labels.eq(value).to_numpy(float) for value in categories]))
        return np.column_stack(columns)

    def to_dict(self):
        return dict(stage=self.stage, numeric=self.numeric, categories=self.categories,
                    means=self.means.tolist(), scales=self.scales.tolist(), feature_names=self.feature_names)

    @classmethod
    def from_dict(cls, data):
        return cls(data["stage"], data["numeric"], data["categories"], data["means"], data["scales"])


def _class_labels(frame, phase):
    # CPU-pre cannot know whether its transport will generate any photons.
    return (np.full(len(frame), "all") if phase == PHASES[0]
            else np.where(frame.num_photons.to_numpy() == 0, "zero", "work"))


class DurationModels:
    """Four log-duration regressions with explicit submission-time work classes."""
    def __init__(self, train, prior_strength=10.0):
        self.train = train.copy().reset_index(drop=True)
        self.prior_strength = prior_strength
        self.transforms, self.models, self.ols, self.medians = {}, {}, {}, {}
        self.residuals = np.zeros((len(train), len(PHASES)))
        for phase_index, phase in enumerate(PHASES):
            transform = FeatureTransform.fit(train, "cpu_pre" if phase_index == 0 else "gpu")
            self.transforms[phase] = transform
            X = transform.transform(train)
            y = np.log(np.maximum(train[phase].to_numpy(float), TIMER_FLOOR_S))
            labels = _class_labels(train, phase)
            self.models[phase], self.ols[phase] = {}, {}
            for label in sorted(set(labels)):
                mask = labels == label
                model = BayesianLinearRegression(V0=np.eye(X.shape[1]) * prior_strength).fit(X[mask], y[mask])
                self.models[phase][label] = model
                self.ols[phase][label] = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
                self.residuals[mask, phase_index] = y[mask] - X[mask] @ model.m_n
            self.medians[phase] = train.groupby("condition_id")[phase].median().to_dict()
            self.medians[phase]["__global__"] = float(train[phase].median())

    def _design(self, frame, phase):
        X, labels = self.transforms[phase].transform(frame), _class_labels(frame, phase)
        missing = set(labels) - set(self.models[phase])
        if missing:
            raise ValueError(f"unseen no-work/work class for {phase}: {sorted(missing)}; additional training runs required")
        return X, labels

    def sample(self, frame, draws, rng, mode="independent"):
        """Sample shared parameters per stream; joint mode resamples train blocks.

        Joint mode retains phase/short-range residual dependence as a sensitivity
        check, not a fully specified correlated Bayesian posterior.
        """
        if mode not in ("independent", "joint_block"):
            raise ValueError("unknown residual sampling mode")
        output = np.empty((draws, len(frame), len(PHASES)))
        residual_rows = None
        if mode == "joint_block":
            residual_rows = np.empty((draws, len(frame)), dtype=int)
            target_class = frame.num_photons.eq(0).to_numpy()
            train_class = self.train.num_photons.eq(0).to_numpy()
            for draw in range(draws):
                for no_work in (False, True):
                    positions = np.flatnonzero(target_class == no_work)
                    if not len(positions):
                        continue
                    groups = [np.asarray(indices, dtype=int) for _, indices in
                              self.train.loc[train_class == no_work].groupby("run_id", sort=False).groups.items()]
                    if not groups:
                        raise ValueError("unseen no-work class for joint residual sampling")
                    sampled = []
                    while len(sampled) < len(positions):
                        group = groups[int(rng.integers(len(groups)))]
                        start = int(rng.integers(len(group)))
                        sampled.extend(group[start:min(start + 8, len(group))].tolist())
                    residual_rows[draw, positions] = sampled[:len(positions)]
        for phase_index, phase in enumerate(PHASES):
            X, labels = self._design(frame, phase)
            for label, model in self.models[phase].items():
                mask = labels == label
                if not mask.any():
                    continue
                if mode == "independent":
                    logs = model.sample_predictive(X[mask], draws, rng)
                else:
                    variance = model.b_n / rng.gamma(model.a_n, size=draws)
                    beta = (model.m_n + (rng.standard_normal((draws, X.shape[1]))
                            @ np.linalg.cholesky(model.V_n).T) * np.sqrt(variance)[:, None])
                    logs = beta @ X[mask].T + self.residuals[residual_rows[:, mask], phase_index]
                with np.errstate(over="raise", invalid="raise"):
                    output[:, mask, phase_index] = np.maximum(np.exp(logs), TIMER_FLOOR_S)
        return output

    def to_dict(self):
        return {"timer_floor_s": TIMER_FLOOR_S, "prior_strength": self.prior_strength,
                "feature_availability": "source only before CPU-pre; generated counts at GPU submission",
                "phases": {phase: {"transform": self.transforms[phase].to_dict(),
                                    "classes": {label: model.to_dict() for label, model in models.items()},
                                    "ols": {label: values.tolist() for label, values in self.ols[phase].items()},
                                    "condition_medians": self.medians[phase]}
                           for phase, models in self.models.items()}}


def fit_duration_models(train, prior_strength=10.0):
    if train.empty:
        raise ValueError("empty training split")
    return DurationModels(train, prior_strength)


def evaluate_models(bundle, frame, split, draws=64, seed=RANDOM_SEED):
    """Return per-event predictions and aggregate errors/analytic coverage."""
    predictions, metrics = [], []
    for phase_index, phase in enumerate(PHASES):
        X, labels = bundle._design(frame, phase)
        observed = frame[phase].to_numpy(float)
        observed_log = np.log(np.maximum(observed, TIMER_FLOOR_S))
        point = np.empty(len(frame))
        ols_log = np.empty(len(frame))
        intervals = {level: [np.empty(len(frame)), np.empty(len(frame))] for level in (.5, .8, .95)}
        for label, model in bundle.models[phase].items():
            mask = labels == label
            if not mask.any():
                continue
            mean, scale, df = model.predict(X[mask])
            # Exponentiation preserves quantiles: this is the exact predictive
            # median, without assuming a finite exponential Student-t mean.
            point[mask] = np.exp(mean)
            ols_log[mask] = X[mask] @ bundle.ols[phase][label]
            for level, (low, high) in intervals.items():
                quantile = student_t.ppf((1 + level) / 2, df)
                low[mask], high[mask] = mean - quantile * scale, mean + quantile * scale
        median = frame.condition_id.map(bundle.medians[phase]).fillna(bundle.medians[phase]["__global__"]).to_numpy(float)
        points = {"bayesian": point, "ols_log": np.exp(ols_log), "condition_median": median}
        for method, predicted in points.items():
            error, log_error = predicted - observed, np.log(np.maximum(predicted, TIMER_FLOOR_S)) - observed_log
            row = dict(split=split, phase=phase, method=method, n=len(frame),
                       mae_s=float(np.mean(np.abs(error))), rmse_s=float(np.sqrt(np.mean(error**2))),
                       log_mae=float(np.mean(np.abs(log_error))), log_rmse=float(np.sqrt(np.mean(log_error**2))))
            if method == "bayesian":
                for level, (low, high) in intervals.items():
                    row[f"coverage_{int(level * 100)}"] = float(np.mean((observed_log >= low) & (observed_log <= high)))
                    row[f"median_interval_width_{int(level * 100)}_s"] = float(np.median(np.exp(high) - np.exp(low)))
            metrics.append(row)
        for index in range(len(frame)):
            row = dict(split=split, phase=phase, run_id=frame.run_id.iloc[index],
                       event_id=int(frame.event_id.iloc[index]), condition_id=frame.condition_id.iloc[index],
                       work_class=labels[index],
                       observed_s=observed[index], predicted_s=point[index],
                       ols_s=points["ols_log"][index], median_s=median[index])
            for level, (low, high) in intervals.items():
                row[f"low_{int(level * 100)}_s"] = float(np.exp(low[index]))
                row[f"high_{int(level * 100)}_s"] = float(np.exp(high[index]))
            predictions.append(row)
    return pd.DataFrame(predictions), pd.DataFrame(metrics)


def summarize_condition_predictions(predictions):
    """Keep heteroscedastic condition/work-class failures visible after pooling."""
    records = []
    keys = ["split", "phase", "condition_id", "work_class"]
    for context, rows in predictions.groupby(keys, sort=False):
        observed = np.maximum(rows.observed_s.to_numpy(float), TIMER_FLOOR_S)
        predicted = np.maximum(rows.predicted_s.to_numpy(float), TIMER_FLOOR_S)
        error = rows.predicted_s.to_numpy(float) - rows.observed_s.to_numpy(float)
        log_error = np.log(predicted) - np.log(observed)
        record = dict(zip(keys, context), n=len(rows), mae_s=float(np.mean(np.abs(error))),
                      rmse_s=float(np.sqrt(np.mean(error**2))),
                      log_rmse=float(np.sqrt(np.mean(log_error**2))))
        for level in (50, 80, 95):
            low, high = rows[f"low_{level}_s"].to_numpy(float), rows[f"high_{level}_s"].to_numpy(float)
            record[f"coverage_{level}"] = float(np.mean((observed >= low) & (observed <= high)))
        records.append(record)
    return pd.DataFrame(records)


def validate_blocking_trace(frame):
    """Compare service replay, submission overhead, and gaps within each run.

    The uncorrected service sum is not compared with a trace as though gaps
    were replay error. Clock origins and initialization before the first event
    are excluded from within-run spans. Warm-up exclusion is the caller's choice.
    """
    records = []
    for run_id, rows in frame.groupby("run_id", sort=False):
        rows = rows.sort_values("event_id")
        start_name = "start_offset_s" if "start_offset_s" in rows else "cpu_pre_start_time_offset_s"
        end_name = "end_offset_s" if "end_offset_s" in rows else "cpu_post_end_time_offset_s"
        starts, ends = rows[start_name].to_numpy(float), rows[end_name].to_numpy(float)
        if (ends < starts).any() or (starts[1:] < ends[:-1] - 1e-9).any():
            raise ValueError(f"run {run_id} is not a blocking, nonoverlapping trace")
        gaps = starts[1:] - ends[:-1]
        overhead = rows.get("gpu_queue_delay_s", pd.Series(0., index=rows.index)).to_numpy(float)
        demands = rows[list(PHASES)].to_numpy(float)
        timeline = replay(_events(rows, demands), policy="blocking")
        service = float(timeline.end_time_offset_s.max())
        corrected_ends = np.cumsum(demands.sum(axis=1) + overhead) + np.r_[0., np.cumsum(gaps)]
        measured = float(ends[-1] - starts[0])
        reconstructed = service + float(overhead.sum() + gaps.sum())
        records.append(dict(run_id=run_id, events=len(rows), measured_span_s=measured,
                            service_replay_makespan_s=service, phase_service_sum_s=float(demands.sum()),
                            submission_overhead_s=float(overhead.sum()), interevent_gap_s=float(gaps.sum()),
                            reconstructed_span_s=reconstructed, reconstruction_error_s=reconstructed - measured,
                            max_event_boundary_error_s=float(np.max(np.abs(corrected_ends - (ends - starts[0])))),
                            cpu_service_fraction=float(demands[:, [0, 2, 3]].sum() / measured) if measured else 0,
                            gpu_service_fraction=float(demands[:, 1].sum() / measured) if measured else 0,
                            gpu_gate_service_fraction=float(demands[:, [1, 2]].sum() / measured) if measured else 0))
    return pd.DataFrame(records)


def _events(frame, demands):
    return [dict(event_id=index, num_photons=int(photons),
                 **{phase: float(demands[index, phase_index]) for phase_index, phase in enumerate(PHASES)})
            for index, photons in enumerate(frame.num_photons.to_numpy())]


def _peak(starts, ends, weights):
    changes = []
    for start, end, weight in zip(starts, ends, weights):
        if end > start:
            changes.extend([(float(start), float(weight)), (float(end), -float(weight))])
    current = peak = 0.
    for _, change in sorted(changes, key=lambda item: (item[0], item[1])):
        current += change
        peak = max(peak, current)
    return peak


def _timeline_metrics(timeline):
    return dict(makespan_s=float(timeline.end_time_offset_s.max()),
                p95_latency_s=float(timeline.latency_s.quantile(.95)),
                p99_latency_s=float(timeline.latency_s.quantile(.99)),
                p95_service_latency_s=float(timeline.service_latency_s.quantile(.95)),
                admission_delay_sum_s=float(timeline.admission_time_offset_s.sum()),
                gpu_queue_delay_sum_s=float(timeline.gpu_queue_delay_s.sum()),
                cpu_service_s=float(timeline.cpu_runtime_s.sum()),
                gpu_service_s=float(timeline.gpu_runtime_s.sum()),
                gpu_gate_occupied_s=float(timeline.gpu_gate_runtime_s.sum()),
                peak_queued_photons=_peak(timeline.gpu_submit_time_offset_s, timeline.gpu_start_time_offset_s,
                                         timeline.num_photons),
                peak_inflight=_peak(timeline.admission_time_offset_s, timeline.end_time_offset_s,
                                    np.ones(len(timeline))))


def _config(policy, workers, max_inflight):
    return dict(policy=policy, workers=workers, max_inflight=max_inflight,
                configuration=f"{policy}_w{workers}_q{max_inflight}")


def _resource_grid(values, name):
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be a nonempty sequence of unique positive integers") from error
    if (not values or any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
                          or value < 1 for value in values) or len(set(values)) != len(values)):
        raise ValueError(f"{name} must be a nonempty sequence of unique positive integers")
    return tuple(sorted(int(value) for value in values))


def _candidates(worker_counts=DEFAULT_WORKERS, inflight_limits=DEFAULT_INFLIGHT):
    worker_counts = _resource_grid(worker_counts, "worker_counts")
    inflight_limits = _resource_grid(inflight_limits, "inflight_limits")
    return ([_config("blocking", 1, 1)]
            + [_config(policy, workers, cap) for policy in ("synchronous", "async")
               for workers in worker_counts for cap in inflight_limits])


def _streams(frame):
    groups = []
    streams = {}
    for condition, rows in frame.groupby("condition_id", sort=True):
        rows = rows.sort_values(["run_id", "event_id"]).reset_index(drop=True)
        homogeneous = rows.iloc[np.linspace(0, len(rows) - 1, min(128, len(rows)), dtype=int)].copy()
        streams["condition:" + str(condition)] = homogeneous
        groups.append(rows)
    per_condition = min(min(len(group) for group in groups), max(1, 256 // len(groups)))
    balanced = pd.concat([group.iloc[np.linspace(0, len(group) - 1, per_condition, dtype=int)] for group in groups],
                         ignore_index=True)
    streams["balanced_mix"] = balanced.iloc[np.random.default_rng(RANDOM_SEED).permutation(len(balanced))].reset_index(drop=True)
    return streams


def _simulate(frame, demands, configurations, *, split, stream, mode="observed", slowdown=1.):
    if demands.ndim == 2:
        demands = demands[None, :, :]
    records = []
    for draw, values in enumerate(demands):
        for configuration in configurations:
            local = values.copy()
            # Contention sensitivity affects parallel candidates, not the serial control.
            effective_slowdown = slowdown if configuration["workers"] > 1 else 1.
            local[:, [0, 2, 3]] *= effective_slowdown
            timeline = replay(_events(frame, local), **{name: configuration[name] for name in ("workers", "max_inflight", "policy")})
            records.append({**configuration, "split": split, "stream": stream, "mode": mode,
                            "cpu_slowdown": slowdown, "draw": draw, "events": len(frame),
                            **_timeline_metrics(timeline)})
    return pd.DataFrame(records)


def _policy_summary(draws):
    records = []
    keys = ["split", "stream", "mode", "cpu_slowdown"]
    for context, context_rows in draws.groupby(keys, sort=False):
        baseline = context_rows.loc[context_rows.policy.eq("blocking")].set_index("draw").makespan_s
        minimum = context_rows.groupby("draw").makespan_s.min()
        for configuration, rows in context_rows.groupby("configuration", sort=False):
            rows = rows.sort_values("draw")
            reference = baseline.reindex(rows.draw).to_numpy()
            makespan = rows.makespan_s.to_numpy()
            low, median, high = np.quantile(makespan, [.025, .5, .975])
            records.append(dict(zip(keys, context), configuration=configuration,
                                policy=rows.policy.iloc[0], workers=int(rows.workers.iloc[0]),
                                max_inflight=int(rows.max_inflight.iloc[0]), events=int(rows.events.iloc[0]),
                                draws=len(rows), makespan_low_s=float(low), makespan_median_s=float(median),
                                makespan_high_s=float(high),
                                p95_latency_median_s=float(rows.p95_latency_s.median()),
                                p99_latency_median_s=float(rows.p99_latency_s.median()),
                                p95_service_latency_median_s=float(rows.p95_service_latency_s.median()),
                                peak_queued_photons_median=float(rows.peak_queued_photons.median()),
                                peak_inflight_max=float(rows.peak_inflight.max()),
                                simulated_speedup_median=float(np.median(reference / makespan)),
                                probability_of_improvement=float(np.mean(makespan < reference)),
                                probability_within_5pct_best_evaluated=float(np.mean(makespan <= 1.05 * minimum.reindex(rows.draw).to_numpy()))))
    return pd.DataFrame(records)


def select_posterior_policy(frame, samples, *, worker_counts=DEFAULT_WORKERS,
                            inflight_limits=DEFAULT_INFLIGHT):
    """Compare the complete policy grid on shared validation predictive draws."""
    configurations = _candidates(worker_counts, inflight_limits)
    draws = _simulate(frame, samples, configurations, split="validation",
                      stream="balanced_mix", mode="independent")
    summary = _policy_summary(draws)
    eligible = summary.loc[summary.makespan_median_s <= 1.05 * summary.makespan_median_s.min()]
    winner = eligible.sort_values(["workers", "max_inflight", "policy"]).iloc[0]
    frozen = _config(str(winner.policy), int(winner.workers), int(winner.max_inflight))
    return draws, summary, frozen


def _diagnostics(bundle):
    lag, drift = {}, {}
    for phase_index, phase in enumerate(PHASES):
        before, after, times, residuals = [], [], [], []
        for _, indices in bundle.train.groupby("run_id", sort=False).groups.items():
            values = bundle.residuals[np.asarray(indices), phase_index]
            if len(values) > 1:
                before.extend(values[:-1]); after.extend(values[1:])
            times.extend(np.linspace(0, 1, len(values)))
            residuals.extend(values)
        lag[phase] = float(np.corrcoef(before, after)[0, 1]) if np.std(before) > 0 and np.std(after) > 0 else None
        drift[phase] = float(np.corrcoef(times, residuals)[0, 1]) if np.std(residuals) > 0 else None
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.corrcoef(bundle.residuals.T)
    return dict(phase_order=list(PHASES), training_phase_correlation=correlation.tolist(),
                training_within_run_lag1=lag, training_within_run_time_correlation=drift,
                joint_residual_block_length=8)


def _prior_checks(bundle):
    checks = {}
    rng = np.random.default_rng(RANDOM_SEED)
    for phase in PHASES:
        X = bundle.transforms[phase].transform(bundle.train)
        vector = X.mean(axis=0)
        variance = 1. / rng.gamma(2., size=1024)
        beta = rng.standard_normal((1024, X.shape[1])) * np.sqrt(10 * variance)[:, None]
        log_samples = beta @ vector + rng.standard_normal(1024) * np.sqrt(variance)
        quantiles = np.quantile(log_samples, [.025, .5, .975])
        checks[phase] = dict(log_duration_quantiles_025_50_975=quantiles.tolist(),
                            duration_quantiles_025_50_975_s=np.exp(quantiles).tolist(),
                            note="Default m0=0, V0=10I, a0=2, b0=1; broad prior center is 1 second")
    return checks


def _clean_json(value):
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_clean_json(value), indent=2, allow_nan=False) + "\n")


def _table(frame, columns):
    frame = frame[list(columns)].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format(value, ".5g") if isinstance(value, (float, np.floating))
                                       else str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def _figures(directory, raw, predictions, metrics, observed, policy_summary, diagnostics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(figure, name):
        figure.tight_layout()
        figure.savefig(directory / name, dpi=160, bbox_inches="tight")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    retained = raw.loc[~raw.is_warmup]
    axes[0].boxplot([np.maximum(retained[phase].to_numpy(float), TIMER_FLOOR_S) for phase in PHASES],
                    tick_labels=["CPU pre", "GPU host", "Post locked", "Post remaining"], showfliers=False)
    axes[0].set(yscale="log", ylabel="Host-observed seconds", title="Measured phase distributions")
    for _, run in raw.groupby("run_id", sort=False):
        total = run[list(PHASES)].sum(axis=1)
        axes[1].plot(run.event_ordinal, total, alpha=.15, linewidth=.7)
    axes[1].set(yscale="log", xlabel="Event ordinal within process", ylabel="Summed service seconds",
                title="Warm-up and time drift (all raw rows)")
    axes[1].axvline(float(raw.loc[raw.is_warmup, "event_ordinal"].max()) + .5 if raw.is_warmup.any() else -.5,
                    color="black", linestyle="--", label="Warm-up cutoff")
    axes[1].legend()
    save(figure, "measured_phases_warmup.png")

    figure, axes = plt.subplots(2, 2, figsize=(9, 8))
    test = predictions.loc[predictions.split.eq("test")]
    for axis, phase in zip(axes.flat, PHASES):
        rows = test.loc[test.phase.eq(phase)]
        x, y = np.maximum(rows.observed_s, TIMER_FLOOR_S), np.maximum(rows.predicted_s, TIMER_FLOOR_S)
        axis.scatter(x, y, s=4, alpha=.35)
        bounds = [min(x.min(), y.min()), max(x.max(), y.max())]
        axis.plot(bounds, bounds, color="black", linestyle="--")
        axis.set(xscale="log", yscale="log", xlabel="Measured seconds", ylabel="Predictive median seconds", title=phase)
    save(figure, "heldout_predictions.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, style in [("validation", "o-"), ("test", "s--")]:
        rows = metrics.loc[metrics.method.eq("bayesian") & metrics.split.eq(split)]
        for _, row in rows.iterrows():
            axes[0].plot([.5, .8, .95], [row.coverage_50, row.coverage_80, row.coverage_95], style,
                         alpha=.65, label=split + ": " + row.phase)
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1)
    axes[0].set(xlabel="Nominal predictive coverage", ylabel="Held-out coverage", xlim=(.4, 1), ylim=(0, 1.03),
                title="Coverage deviations remain visible")
    axes[0].legend(fontsize=5)
    correlation = np.array(diagnostics["training_phase_correlation"], dtype=float)
    im = axes[1].imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_xticks(range(4), ["Pre", "GPU", "Locked", "Remaining"], rotation=30)
    axes[1].set_yticks(range(4), ["Pre", "GPU", "Locked", "Remaining"])
    axes[1].set_title("Training log-residual correlations")
    figure.colorbar(im, ax=axes[1])
    save(figure, "coverage_and_residuals.png")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    rows = observed.loc[observed.stream.eq("balanced_mix")]
    for policy in ("blocking", "synchronous", "async"):
        subset = rows.loc[rows.policy.eq(policy)]
        axes[0].scatter(subset.makespan_s, subset.p95_latency_s, s=15 + subset.workers * 4, alpha=.65, label=policy)
    axes[0].set(xlabel="Simulated makespan seconds", ylabel="Release-based p95 latency seconds",
                title="Validation: finite observed-duration policies")
    axes[0].legend()
    rows = policy_summary.loc[policy_summary.split.eq("validation") & policy_summary.stream.eq("balanced_mix")
                              & policy_summary["mode"].eq("independent")]
    # Keep the figure legible; selection and exported tables use the full grid.
    best = rows.nsmallest(8, "makespan_median_s")
    rows = pd.concat([best, rows.loc[rows.policy.eq("blocking")]]).drop_duplicates("configuration")
    positions = np.arange(len(rows))
    axes[1].errorbar(positions, rows.makespan_median_s,
                     yerr=[rows.makespan_median_s - rows.makespan_low_s,
                           rows.makespan_high_s - rows.makespan_median_s], fmt="o", capsize=3)
    axes[1].set_xticks(positions, rows.configuration, rotation=65, ha="right", fontsize=7)
    axes[1].set(ylabel="Simulated makespan seconds", title="Full posterior grid: best eight and blocking (95% intervals)")
    save(figure, "counterfactual_policy_comparison.png")

    figure, axis = plt.subplots(figsize=(9, 4))
    rows = policy_summary.loc[policy_summary.split.eq("test") & policy_summary.stream.eq("balanced_mix")]
    for policy, subset in rows.groupby("configuration"):
        labels = subset["mode"] + " / CPU x" + subset.cpu_slowdown.astype(str)
        axis.errorbar(labels, subset.simulated_speedup_median, marker="o", linestyle="none", label=policy)
    axis.axhline(1, color="black", linewidth=1)
    axis.set(ylabel="Simulated baseline/chosen makespan ratio", title="Frozen policy: contention and residual sensitivity")
    axis.tick_params(axis="x", labelrotation=25)
    axis.legend(fontsize=7)
    save(figure, "frozen_policy_sensitivity.png")


def run_study(root, draws=64, *, worker_counts=DEFAULT_WORKERS,
              inflight_limits=DEFAULT_INFLIGHT):
    """Fit, calibrate, freeze a validation policy, then write held-out artifacts.

    All configured finite policies use observed validation durations and shared
    predictive draws on the validation mix;
    only the frozen choice and blocking control receive full test predictions.
    Homogeneous streams cap at 128 events; the deterministic balanced mix at
    256. This is an offline inference experiment, not a hardware speedup test.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    worker_counts = _resource_grid(worker_counts, "worker_counts")
    inflight_limits = _resource_grid(inflight_limits, "inflight_limits")
    root = Path(root).resolve()
    raw = load_runs(root)
    train, validation, test = split_by_replication(raw)
    if any(part.empty for part in (train, validation, test)):
        raise ValueError("study needs complete train(0/1), validation(2), and test(3) replications")
    if set(train.replication) != {0, 1}:
        raise ValueError("both independent training replications 0 and 1 are required")
    conditions = set(train.condition_id)
    if set(validation.condition_id) != conditions or set(test.condition_id) != conditions:
        raise ValueError("every measured condition must occur in train, validation, and test")
    stamp = datetime.now(timezone.utc).strftime("analysis-%Y%m%dT%H%M%S-%fZ")
    report_dir = root / "reports" / stamp
    dataset_dir, model_dir, figure_dir = (root / folder / stamp for folder in ("datasets", "models", "figures"))
    for directory in (report_dir, dataset_dir, model_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=False)
    raw.to_csv(dataset_dir / "events.csv", index=False)
    _write_json(dataset_dir / "run-index.json", raw.attrs["run_index"])
    splits = dict(train_replications=[0, 1], validation_replications=[2], test_replications=[3],
                  train_runs=sorted(train.run_id.unique()), validation_runs=sorted(validation.run_id.unique()),
                  test_runs=sorted(test.run_id.unique()), train_events=len(train), validation_events=len(validation), test_events=len(test))
    _write_json(dataset_dir / "splits.json", splits)

    bundle = fit_duration_models(train)
    _write_json(model_dir / "duration-models.json", bundle.to_dict())
    diagnostics, prior_checks = _diagnostics(bundle), _prior_checks(bundle)
    _write_json(model_dir / "training-diagnostics.json", diagnostics)
    prediction_frames, metric_frames = [], []
    prediction, metrics = evaluate_models(bundle, validation, "validation", draws)
    prediction_frames.append(prediction); metric_frames.append(metrics)
    prior_sensitivity = []
    for strength in (1., 100.):
        alternative = fit_duration_models(train, prior_strength=strength)
        _, alternative_metrics = evaluate_models(alternative, validation, "validation", draws)
        alternative_metrics["prior_strength"] = strength
        prior_sensitivity.append(alternative_metrics)
    pd.concat(prior_sensitivity, ignore_index=True).to_csv(dataset_dir / "validation-prior-sensitivity.csv", index=False)

    validation_streams = _streams(validation)
    candidates = _candidates(worker_counts, inflight_limits)
    observed = pd.concat([_simulate(frame, frame[list(PHASES)].to_numpy(float), candidates,
                                     split="validation", stream=name)
                          for name, frame in validation_streams.items()], ignore_index=True)
    observed.to_csv(dataset_dir / "validation-observed-policy-grid.csv", index=False)
    mix = validation_streams["balanced_mix"]
    sampled = bundle.sample(mix, draws, np.random.default_rng(RANDOM_SEED + 1))
    validation_draws, validation_summary, frozen = select_posterior_policy(
        mix, sampled, worker_counts=worker_counts, inflight_limits=inflight_limits)
    selection = dict(selected_on="validation", final_evaluation_on="test", configuration=frozen,
                     criterion="least workers then inflight within 5% of best posterior median makespan on validation balanced mix",
                     cpu_event_batch_size=1,
                     candidate_configurations=len(candidates), posterior_configurations=candidates,
                     worker_counts=worker_counts, inflight_limits=inflight_limits,
                     mixture_shuffle_seed=RANDOM_SEED, homogeneous_event_cap=128, balanced_mix_event_cap=256)
    # Final-test outcomes do not participate in fitting or policy selection.
    # Integrity/condition checks above are allowed before this freeze.
    _write_json(model_dir / "frozen-policy.json", selection)

    prediction, metrics = evaluate_models(bundle, test, "test", draws, RANDOM_SEED + 2)
    prediction_frames.append(prediction); metric_frames.append(metrics)
    predictions, metrics = pd.concat(prediction_frames, ignore_index=True), pd.concat(metric_frames, ignore_index=True)
    condition_metrics = summarize_condition_predictions(predictions)
    condition_metrics.to_csv(dataset_dir / "heldout-condition-metrics.csv", index=False)
    predictions.to_csv(dataset_dir / "heldout-predictions.csv", index=False)
    metrics.to_csv(dataset_dir / "heldout-metrics.csv", index=False)
    test_streams = _streams(test)
    controls = [candidates[0]] + ([frozen] if frozen["policy"] != "blocking" else [])
    policy_draws = [validation_draws]
    observed_test = []
    for index, (name, frame) in enumerate(test_streams.items()):
        sampled = bundle.sample(frame, draws, np.random.default_rng(RANDOM_SEED + 100 + index))
        policy_draws.append(_simulate(frame, sampled, controls, split="test", stream=name, mode="independent"))
        observed_test.append(_simulate(frame, frame[list(PHASES)].to_numpy(float), controls, split="test", stream=name))
        if name == "balanced_mix":
            for slowdown in (1.5, 2.):
                policy_draws.append(_simulate(frame, sampled, controls, split="test", stream=name,
                                               mode="independent", slowdown=slowdown))
            joint = bundle.sample(frame, draws, np.random.default_rng(RANDOM_SEED + 300), mode="joint_block")
            policy_draws.append(_simulate(frame, joint, controls, split="test", stream=name, mode="joint_block"))
    all_draws = pd.concat(policy_draws, ignore_index=True)
    all_draws.to_csv(dataset_dir / "posterior-policy-draws.csv", index=False)
    policy_summary = _policy_summary(all_draws)
    policy_summary.to_csv(dataset_dir / "posterior-policy-summary.csv", index=False)
    pd.concat(observed_test, ignore_index=True).to_csv(dataset_dir / "test-observed-frozen-policy.csv", index=False)
    streams = {split: {name: [[row.run_id, int(row.event_id)] for row in frame.itertuples()]
                       for name, frame in stream_frames.items()}
               for split, stream_frames in [("validation", validation_streams), ("test", test_streams)]}
    _write_json(dataset_dir / "policy-streams.json", streams)
    trace = validate_blocking_trace(pd.concat([validation, test], ignore_index=True))
    trace.to_csv(dataset_dir / "blocking-trace-validation.csv", index=False)

    test_calibration = metrics.loc[metrics.split.eq("test") & metrics.method.eq("bayesian")]
    calibration_failures = test_calibration.loc[(test_calibration.coverage_95 < .9)
                                               | (test_calibration.coverage_95 > .99), ["phase", "coverage_95"]].to_dict("records")
    condition_failures = condition_metrics.loc[condition_metrics.split.eq("test")
                                               & ((condition_metrics.coverage_95 < .9)
                                                  | (condition_metrics.coverage_95 > .99))].to_dict("records")
    limitations = [
        "Policy gains are simulated counterfactual estimates; concurrent charged-particle execution was not measured or implemented.",
        "The policy grid uses unit-event CPU admission (event_batch_size=1). Actual Geant4 auto batching needs the explicit batch-aware synchronous replay and separate optical controls.",
        "Host-observed GPU service includes transfer/synchronization; service fractions are not hardware utilization.",
        "Serial CPU demands are assumed unchanged by concurrency except in the explicit x1/x1.5/x2 sensitivity; the serial control remains x1.",
        "Posterior regression treats training residuals as independent; shared parameter draws preserve parameter uncertainty but do not estimate run random effects.",
        "Joint eight-row within-run residual resampling is a sensitivity check, not a calibrated correlated Bayesian posterior; only two independent training replications limit uncertainty claims.",
        "Generated photons/gensteps condition GPU/post forecasts at submission; CPU-pre sees source settings only. Zero-work submission paths have separate models and unseen classes are rejected.",
        "Zero measured phase lengths are retained and floored to 1 ns for log fitting; this is a modeling approximation.",
        "The per-condition median and OLS baselines use training data only. Prior-strength alternatives are diagnostics on validation, with the predeclared default frozen for test.",
        "Finite-batch service comparisons omit measured driver gaps and submission overhead; their separate blocking-trace accounting does not establish concurrent overhead behavior.",
        "Posterior probabilities have finite Monte Carlo resolution of 1/draws; 64 draws are an initial bounded analysis, not high-precision tail estimation.",
        "Finite streams contain at most 128 events per condition and 256 in the deterministic balanced mixture. This is not a GPU-batching or memory-capacity experiment.",
        "Five warm-up events (or run-specific metadata) are marked, with all raw rows preserved. The warm-up plot must be inspected for remaining startup drift.",
        "Recording overhead controls and actual optical MT scaling, if collected, require separate analysis with their recorded geometry/source settings.",
    ]
    if condition_failures:
        limitations.insert(0, f"{len(condition_failures)} test condition/phase/work-class groups fall outside the descriptive 90%-99% coverage band; pooled coverage does not establish local calibration. See heldout-condition-metrics.csv.")
    if calibration_failures:
        limitations.insert(0, "Predictive calibration failed the descriptive 90%-99% acceptance band for nominal 95% coverage in at least one test phase. Policy intervals must not be read as validated uncertainty bounds.")
    if any(value is not None and abs(value) > .3 for value in diagnostics["training_within_run_lag1"].values()):
        limitations.insert(0, "Training residuals show material lag-one correlation (absolute correlation above 0.3); iid model parameter intervals may be too narrow.")
    if trace.max_event_boundary_error_s.max() > 1e-7:
        limitations.insert(0, "Blocking trace reconstruction has unexplained boundary errors above 100 ns; inspect blocking-trace-validation.csv before trusting replay comparisons.")
    models_summary = {phase: {"classes": {label: {"training_events": model.n_observations,
                                                  "a_n": model.a_n, "b_n": model.b_n}
                                             for label, model in phase_models.items()},
                             "features": bundle.transforms[phase].feature_names,
                             "zero_timer_rows": int(train[phase].eq(0).sum())}
                      for phase, phase_models in bundle.models.items()}
    final_mix = policy_summary.loc[policy_summary.split.eq("test") & policy_summary.stream.eq("balanced_mix")]
    summary = dict(version=1, experiment="offline counterfactual Bayesian dispatch study", created_utc=stamp,
                   dataset=dict(raw_events=len(raw), retained_events=len(train) + len(validation) + len(test),
                                runs=raw.run_id.nunique(), conditions=raw.condition_id.nunique(),
                                zero_photon_events=int((~raw.is_warmup & raw.zero_photon).sum())),
                   splits=splits, draws=draws, random_seed=RANDOM_SEED, timer_floor_s=TIMER_FLOOR_S,
                   models=models_summary, prior_predictive_checks=prior_checks, residual_diagnostics=diagnostics,
                   calibration_failures=calibration_failures, condition_calibration_failures=condition_failures,
                   heldout_metrics=metrics.to_dict("records"),
                   policy_selection=selection, final_balanced_mix=final_mix.to_dict("records"),
                   blocking_trace=dict(max_reconstruction_error_s=float(trace.reconstruction_error_s.abs().max()),
                                       max_event_boundary_error_s=float(trace.max_event_boundary_error_s.max()),
                                       total_interevent_gaps_s=float(trace.interevent_gap_s.sum()),
                                       total_submission_overhead_s=float(trace.submission_overhead_s.sum())),
                   artifacts=dict(dataset=str(dataset_dir), models=str(model_dir), figures=str(figure_dir)),
                   limitations=limitations)
    _figures(figure_dir, raw, predictions, metrics, observed, policy_summary, diagnostics)
    summary_path, report_path = report_dir / "summary.json", report_dir / "report.md"
    _write_json(summary_path, summary)
    rerun = ("PYTHONPATH=" + shlex.quote(str(Path(__file__).resolve().parent.parent))
             + " PYTHONPYCACHEPREFIX=" + shlex.quote(str(root / "cache/pycache"))
             + " TMPDIR=" + shlex.quote(str(root / "tmp")) + " MPLCONFIGDIR=" + shlex.quote(str(root / "cache/matplotlib"))
             + " " + shlex.quote(sys.executable) + " -m teerex.study --root " + shlex.quote(str(root)) + " --draws " + str(draws)
             + " --workers " + " ".join(map(str, worker_counts))
             + " --inflight-limits " + " ".join(map(str, inflight_limits)))
    report = ["# Measured durations and offline dispatch estimates", "",
              f"The study retains {summary['dataset']['retained_events']} measured events from {summary['dataset']['runs']} independent process runs. Replications 0/1 train the transforms and four conditional log-duration models; replication 2 selects the finite policy and replication 3 evaluates the frozen choice. Every original timeline remains unchanged.", "",
              f"The selected configuration is **{frozen['configuration']}**, chosen as the least concurrency within 5% of the best posterior median makespan across the complete {len(candidates)}-configuration grid on shared validation posterior draws. All configurations also received identical observed validation event streams. The policy is FIFO and does not rank events using future observed durations.", "",
              "These are simulated counterfactual comparisons, not observed hardware speedups. The confidence of the recommendation is constrained by the held-out calibration and sensitivity checks below.", "",
              "## Held-out duration checks", "",
              _table(metrics, ["split", "phase", "method", "mae_s", "rmse_s", "log_rmse", "coverage_95"]), "",
              ("Nominal 95% coverage falls outside the descriptive 90%-99% band for: " + str(calibration_failures)
               if calibration_failures else "All test phase 95% coverages fall inside the descriptive 90%-99% band; this is a diagnostic, not an independent policy validation."), "",
              "Duration-scale point predictions use the exact median exp(log-location); coverage uses analytic Student-t quantiles on the log scale and monotone exponentiation. Scheduling uses coherent posterior simulations. No finite-degree Student-t exponential expectation is used. Condition medians and ordinary log-regression are fitted only to training rows.", "",
              f"Per-condition/work-class diagnostics identify {len(condition_failures)} test groups outside that descriptive band; these n-specific coverages and errors are saved in heldout-condition-metrics.csv. Posterior intervals remain descriptive and conditional on the model, particularly with only two training replication clusters.", "",
              "## Blocking trace reconstruction", "",
              f"The largest reconstructed boundary discrepancy is {summary['blocking_trace']['max_event_boundary_error_s']:.6g} s. Summed phase replay is accounted separately from {summary['blocking_trace']['total_submission_overhead_s']:.6g} s of measured submission overhead and {summary['blocking_trace']['total_interevent_gaps_s']:.6g} s of inter-event gaps over validation/test runs. Each process retains its own clock origin; startup before its first retained event is outside this span.", "",
              "## Frozen policy on final test mixture", "",
              _table(final_mix, ["configuration", "mode", "cpu_slowdown", "makespan_low_s", "makespan_median_s", "makespan_high_s", "p95_latency_median_s", "p99_latency_median_s", "peak_queued_photons_median", "simulated_speedup_median", "probability_of_improvement"]), "",
              "All finite-batch events are released at time zero. Reported completion latency includes admission backlog; admission-to-completion latency is a separate CSV field. CPU pre and both dependent post phases share a finite CPU pool. Synchronous workers retain their event slot during GPU waits; async workers release it. The GPU dispatch gate remains occupied through lock-held post cleanup. Queued photons are a workload proxy, not a memory-capacity guarantee.", "",
              "## Prior, dependence and scope", "",
              "The saved models include m_n/V_n/a_n/b_n, the conditional normal/inverse-gamma prior and training-only feature means/scales. Prior predictive checks use default m0=0, V0=10I, a0=2, b0=1; validation-only V0=I and V0=100I alternatives are in validation-prior-sensitivity.csv. The broad log prior is centered at one second and should be assessed against measured timing scales. Phase correlations and lag-one/time-drift diagnostics use training residuals. Joint-block sensitivity resamples joint residual tuples in blocks of up to eight rows within a training run and work class, alongside coherent coefficient draws.", "",
              *["- " + limitation for limitation in limitations], "",
              "## Artifacts and reproduction", "",
              f"Derived tables and run-file SHA256 checksums: `{dataset_dir}`. Persisted split definitions, feature transforms and priors: `{model_dir}`. Standalone PNG figures: `{figure_dir}`. Machine-readable summary: `{summary_path}`.", "",
              "```bash", rerun, "```", "",
              "Each invocation creates new timestamped artifacts and leaves raw runs intact."]
    report_path.write_text("\n".join(report) + "\n")
    return dict(report=str(report_path), summary=str(summary_path), figures=str(figure_dir),
                datasets=str(dataset_dir), models=str(model_dir), selected_policy=frozen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS)
    parser.add_argument("--inflight-limits", type=int, nargs="+", default=DEFAULT_INFLIGHT)
    args = parser.parse_args(argv)
    result = run_study(args.root, args.draws, worker_counts=args.workers,
                       inflight_limits=args.inflight_limits)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
