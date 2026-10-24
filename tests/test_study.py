"""Analysis contracts: independent runs, available features, and trace accounting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from teerex import study


def write_run(
    root,
    run_id,
    replication=0,
    *,
    warmup=1,
    n=5,
    photons=100,
    status="complete",
    arm="physics_production",
    offset=10.0,
    multiplicity=1,
):
    directory = Path(root) / "runs" / run_id
    directory.mkdir(parents=True)
    metadata = dict(
        run_id=run_id,
        arm=arm,
        particle="mu-",
        momentum_gev_c=5.0,
        multiplicity=multiplicity,
        replication=replication,
        clhep_seed=100 + replication,
        gpu_seed=200 + replication,
        gpu_offset=0,
        events=n,
        warmup_events=warmup,
        record_timing=True,
        status=status,
        duration_wall_s=100,
        gpu_uuid="GPU-test",
    )
    (directory / "run.json").write_text(json.dumps(metadata))
    rows = []
    start = offset
    for event_id in range(n):
        # Independently hand-checkable: four phases sum to 4 s, queue 0.2 s.
        # A small varying multiplier also lets model smoke tests fit residuals.
        factor = 1 + 0.01 * event_id + 0.02 * replication
        pre, gpu, locked, unlocked = np.array([1, 2, 0.5, 0.5]) * factor
        submit = start + pre
        gpu_start = submit + 0.2
        gpu_end = gpu_start + gpu
        release = gpu_end + locked
        end = release + unlocked
        rows.append(
            dict(
                scenario="simg4ox_blocking",
                dispatch_mode="blocking",
                event_id=event_id,
                cpu_pre_runtime_s=pre,
                gpu_runtime_s=gpu,
                cpu_post_locked_runtime_s=locked,
                cpu_post_unlocked_runtime_s=unlocked,
                cpu_post_runtime_s=locked + unlocked,
                gpu_gate_runtime_s=gpu + locked,
                gpu_queue_delay_s=0.2,
                start_offset_s=start,
                end_offset_s=end,
                cpu_pre_start_time_offset_s=start,
                cpu_pre_end_time_offset_s=submit,
                gpu_submit_time_offset_s=submit,
                gpu_start_time_offset_s=gpu_start,
                gpu_end_time_offset_s=gpu_end,
                gpu_release_time_offset_s=release,
                cpu_post_end_time_offset_s=end,
                num_photons=photons,
                num_gensteps=int(photons > 0),
                num_gpu_hits=17,
                num_g4_hits=19,
            )
        )
        start = end + 0.3
    pd.DataFrame(rows).to_csv(directory / "events.csv", index=False)
    (directory / "events.manifest.json").write_text(json.dumps({"event_count": n}))
    (directory / "simg4ox.stdout.log").write_text("controlled fixture\n")
    return directory


def test_independent_origins_repeated_ids_and_raw_bytes_are_preserved(tmp_path):
    first = write_run(tmp_path, "first", offset=10)
    second = write_run(tmp_path, "second", replication=1, offset=1000)
    before = (first / "events.csv").read_bytes()
    frame = study.load_runs(tmp_path)
    assert len(frame) == 10
    assert len(frame[["run_id", "event_id"]].drop_duplicates()) == 10
    assert frame.scenario.nunique() == 2
    assert set(frame.source_scenario) == {"simg4ox_blocking"}
    assert frame.loc[frame.run_id.eq("second"), "start_offset_s"].min() == 1000
    assert frame.groupby("run_id").is_warmup.sum().to_dict() == {
        "first": 1,
        "second": 1,
    }
    assert before == (first / "events.csv").read_bytes()
    assert (second / "events.csv").is_file()


def test_recorded_raw_checksum_and_manifest_seed_are_enforced(tmp_path):
    directory = write_run(tmp_path, "checksum")
    path = directory / "run.json"
    metadata = json.loads(path.read_text())
    metadata["csv_sha256"] = study._checksum(directory / "events.csv")
    path.write_text(json.dumps(metadata))
    study.load_runs(tmp_path)
    with (directory / "events.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        study.load_runs(tmp_path)
    metadata["csv_sha256"] = study._checksum(directory / "events.csv")
    path.write_text(json.dumps(metadata))
    manifest = directory / "events.manifest.json"
    manifest.write_text(json.dumps({"event_count": 5, "random_seed": 999}))
    with pytest.raises(ValueError, match="seed"):
        study.load_runs(tmp_path)


@pytest.mark.parametrize("key, value", [("clhep_seed", 100), ("gpu_seed", 200)])
def test_replications_cannot_reuse_clhep_or_gpu_streams(tmp_path, key, value):
    write_run(tmp_path, "r0", replication=0)
    directory = write_run(tmp_path, "r1", replication=1)
    path = directory / "run.json"
    original = json.loads(path.read_text())
    metadata = {**original, key: value}
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="reus"):
        study.load_runs(tmp_path)


def test_failed_and_other_arm_runs_are_excluded(tmp_path):
    write_run(tmp_path, "good")
    write_run(tmp_path, "failed", status="failed")
    write_run(tmp_path, "pilot", arm="physics_pilot")
    assert set(study.load_runs(tmp_path).run_id) == {"good"}
    assert set(study.load_runs(tmp_path, arm="physics_pilot").run_id) == {"pilot"}


@pytest.mark.parametrize(
    "column, value",
    [("gpu_runtime_s", -1), ("cpu_pre_runtime_s", np.inf), ("event_id", 0)],
)
def test_invalid_durations_duplicate_ids_and_incomplete_count_are_rejected(
    tmp_path, column, value
):
    directory = write_run(tmp_path, "invalid")
    source = pd.read_csv(directory / "events.csv")
    bad = source.copy()
    bad.loc[1, column] = value
    bad.to_csv(directory / "events.csv", index=False)
    with pytest.raises(ValueError):
        study.load_runs(tmp_path)
    source.iloc[:-1].to_csv(directory / "events.csv", index=False)
    with pytest.raises(ValueError):
        study.load_runs(tmp_path)


@pytest.mark.parametrize(
    "column, value",
    [
        ("primary_particle", "e-"),
        ("primary_momentum_gev_c", 20),
        ("primary_multiplicity", 128),
        ("gpu_runtime_s", 200),
    ],
)
def test_metadata_disagreement_and_inconsistent_phase_boundaries_are_rejected(
    tmp_path, column, value
):
    directory = write_run(tmp_path, "mismatch")
    source = pd.read_csv(directory / "events.csv")
    bad = source.copy()
    bad[column] = value
    bad.to_csv(directory / "events.csv", index=False)
    with pytest.raises(ValueError):
        study.load_runs(tmp_path)
    source.to_csv(directory / "events.csv", index=False)
    (directory / "events.manifest.json").write_text(
        json.dumps({"event_count": 5, "primary_particle": "e-"})
    )
    with pytest.raises(ValueError):
        study.load_runs(tmp_path)


def test_splits_keep_whole_replications_and_exclude_only_marked_warmup(tmp_path):
    for replication in range(4):
        write_run(tmp_path, f"run{replication}", replication)
    frame = study.load_runs(tmp_path)
    train, val, test = study.split_by_replication(frame)
    assert set(train.replication) == {0, 1}
    assert set(val.replication) == {2}
    assert set(test.replication) == {3}
    assert [len(train), len(val), len(test)] == [8, 4, 4]
    assert not any((part.is_warmup.any() for part in (train, val, test)))
    assert set(train.run_id).isdisjoint(test.run_id)
    frame.loc[0, "replication"] = 3
    with pytest.raises(ValueError):
        study.split_by_replication(frame)


def test_zero_photon_and_zero_timer_rows_are_retained(tmp_path):
    directory = write_run(tmp_path, "zero", photons=0)
    source = pd.read_csv(directory / "events.csv")
    source.loc[4, "cpu_post_unlocked_runtime_s"] = 0
    source.loc[4, "end_offset_s"] = source.loc[4, "gpu_release_time_offset_s"]
    source.loc[4, "cpu_post_end_time_offset_s"] = source.loc[
        4, "gpu_release_time_offset_s"
    ]
    source.loc[4, "cpu_post_runtime_s"] = source.loc[4, "cpu_post_locked_runtime_s"]
    source.to_csv(directory / "events.csv", index=False)
    frame = study.load_runs(tmp_path)
    assert len(frame) == 5
    assert frame.zero_photon.all()
    assert frame.cpu_post_unlocked_runtime_s.iloc[4] == 0


@pytest.fixture
def training_frame():
    return pd.DataFrame(
        dict(
            particle=["mu-", "e-", "mu-", "e-"],
            momentum_gev_c=[1, 5, 20, 5],
            multiplicity=[1, 16, 128, 16],
            num_photons=[0, 10, 100, 10],
            num_gensteps=[0, 1, 2, 1],
            num_gpu_hits=[1, 2, 3, 4],
            gpu_runtime_s=[1, 2, 3, 4],
        )
    )


def test_pre_features_ignore_photons_hits_and_future_durations(training_frame):
    train = training_frame
    transform = study.FeatureTransform.fit(train, stage="cpu_pre")
    changed = train.copy()
    changed[["num_photons", "num_gensteps", "num_gpu_hits", "gpu_runtime_s"]] *= 1000
    np.testing.assert_array_equal(
        transform.transform(train), transform.transform(changed)
    )
    assert not any(
        ("photon" in name or "hit" in name for name in transform.feature_names)
    )
    gpu = study.FeatureTransform.fit(train, stage="gpu")
    assert not np.array_equal(gpu.transform(train), gpu.transform(changed))
    hit_only = train.copy()
    hit_only[["num_gpu_hits", "gpu_runtime_s"]] *= 1000
    np.testing.assert_array_equal(gpu.transform(train), gpu.transform(hit_only))


def test_normalization_is_training_only_and_persistence_reproduces_features(
    training_frame,
):
    train = training_frame
    transform = study.FeatureTransform.fit(train, stage="gpu")
    features = transform.transform(train)
    np.testing.assert_allclose(features[:, 1:5].mean(axis=0), 0, atol=1e-15)
    heldout = train.copy()
    heldout["momentum_gev_c"] = 10000
    assert transform.transform(heldout)[:, 1].mean() > 1
    restored = study.FeatureTransform.from_dict(transform.to_dict())
    np.testing.assert_array_equal(
        restored.transform(heldout), transform.transform(heldout)
    )
    heldout["particle"] = "pi-"
    with pytest.raises(ValueError):
        transform.transform(heldout)


def test_service_replay_and_gaps_are_accounted_separately_per_run():
    # Span: 8 service + 0.4 submission overhead + 0.3 gap = 8.7.
    records = []
    for run_id, origin in [("a", 10), ("b", 1000)]:
        for event_id, start in [(0, origin), (1, origin + 4.5)]:
            records.append(
                dict(
                    run_id=run_id,
                    event_id=event_id,
                    cpu_pre_runtime_s=1,
                    gpu_runtime_s=2,
                    cpu_post_locked_runtime_s=0.5,
                    cpu_post_unlocked_runtime_s=0.5,
                    start_offset_s=start,
                    end_offset_s=start + 4.2,
                    gpu_queue_delay_s=0.2,
                    num_photons=10,
                )
            )
    result = study.validate_blocking_trace(pd.DataFrame(records))
    assert len(result) == 2
    np.testing.assert_allclose(result.service_replay_makespan_s, [8, 8])
    np.testing.assert_allclose(result.measured_span_s, [8.7, 8.7])
    np.testing.assert_allclose(result.interevent_gap_s, [0.3, 0.3])
    np.testing.assert_allclose(result.submission_overhead_s, [0.4, 0.4])
    np.testing.assert_allclose(result.reconstruction_error_s, [0, 0], atol=1e-12)


def test_full_posterior_grid_survives_observed_prediction_rank_reversal():
    frame = pd.DataFrame({"num_photons": [1] * 30})
    observed_demands = np.tile([1.0, 100.0, 0.1, 0.1], (30, 1))
    observed = study._simulate(
        frame,
        observed_demands,
        study._candidates(),
        split="validation",
        stream="balanced_mix",
    )
    predictive = np.tile([100.0, 1.0, 0.1, 0.1], (1, 30, 1))
    draws, summary, chosen = study.select_posterior_policy(frame, predictive)
    assert draws.configuration.nunique() == len(study._candidates())
    assert chosen["configuration"] == "async_w8_q8"
    assert summary.makespan_median_s.min() < 407.0
    assert "async_w8_q32" in set(draws.configuration)


def test_split_keeps_run_provenance_at_dataset_level(tmp_path):
    for replication in range(4):
        write_run(tmp_path, f"run{replication}", replication)
    raw = study.load_runs(tmp_path)
    assert "run_index" in raw.attrs
    for split in study.split_by_replication(raw):
        assert "run_index" not in split.attrs
    assert len(raw.attrs["run_index"]) == 4


def test_condition_coverage_does_not_hide_a_failing_subgroup():
    rows = pd.DataFrame(
        dict(
            split=["test"] * 4,
            phase=["gpu_runtime_s"] * 4,
            condition_id=["small", "small", "large", "large"],
            work_class=["work"] * 4,
            observed_s=[1.0] * 4,
            predicted_s=[1.0] * 4,
            low_50_s=[0.5, 0.5, 0.5, 2.0],
            high_50_s=[2.0, 2.0, 2.0, 4.0],
            low_80_s=[0.5, 0.5, 0.5, 2.0],
            high_80_s=[2.0, 2.0, 2.0, 4.0],
            low_95_s=[0.5, 0.5, 0.5, 2.0],
            high_95_s=[2.0, 2.0, 2.0, 4.0],
        )
    )
    result = study.summarize_condition_predictions(rows).set_index("condition_id")
    assert result.loc["small", "coverage_95"] == 1.0
    assert result.loc["large", "coverage_95"] == 0.5
    assert result.loc["large", "n"] == 2
    assert result.loc["small", "rmse_s"] == 0.0


def test_heldout_point_errors_use_exact_median_and_unseen_work_class_is_rejected(
    tmp_path,
):
    root = tmp_path
    for replication in range(4):
        write_run(root, f"run{replication}", replication, n=9)
    train, val, _ = study.split_by_replication(study.load_runs(root))
    bundle = study.fit_duration_models(train)
    predictions, _ = study.evaluate_models(bundle, val, "validation", draws=2)
    phase = "cpu_pre_runtime_s"
    X = bundle.transforms[phase].transform(val)
    mean, _, _ = bundle.models[phase]["all"].predict(X)
    np.testing.assert_allclose(
        predictions.loc[predictions.phase.eq(phase), "predicted_s"],
        np.exp(mean),
        rtol=1e-14,
    )
    no_work = val.copy()
    no_work["num_photons"] = 0
    with pytest.raises(ValueError, match="unseen"):
        bundle.sample(no_work, 2, np.random.default_rng(42))


def test_expanded_cpu_budget_is_used_for_posterior_selection():
    workers = (1, 2, 4, 8, 16, 20)
    caps = (1, 2, 4, 8, 16, 32, 64)
    frame = pd.DataFrame({"num_photons": [100] * 80})
    draws = np.tile([100.0, 1.0, 0.1, 0.1], (1, 80, 1))
    simulations, summary, chosen = study.select_posterior_policy(
        frame, draws, worker_counts=workers, inflight_limits=caps
    )
    assert simulations.configuration.nunique() == 85
    assert len(summary) == 85
    assert chosen["workers"] == 20
    assert chosen["max_inflight"] >= 20
    assert "async_w20_q64" in set(simulations.configuration)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"worker_counts": ()},
        {"worker_counts": (1, 0)},
        {"worker_counts": (True, 2)},
        {"worker_counts": (1, 1)},
        {"inflight_limits": ()},
        {"inflight_limits": (1, -1)},
        {"inflight_limits": (1, 1.5)},
    ),
)
def test_invalid_resource_grids_are_rejected(kwargs):
    with pytest.raises(ValueError):
        study._candidates(**kwargs)


def test_end_to_end_report_freezes_validation_policy_and_retains_no_work_class(
    tmp_path,
):
    root = tmp_path
    for replication in range(4):
        write_run(root, f"work{replication}", replication, n=9)
        write_run(
            root, f"empty{replication}", replication, photons=0, n=9, multiplicity=16
        )
    result = study.run_study(
        root, draws=4, worker_counts=(1, 20), inflight_limits=(1, 32)
    )
    assert Path(result["report"]).is_file()
    summary = json.loads(Path(result["summary"]).read_text())
    assert summary["splits"]["train_replications"] == [0, 1]
    assert summary["policy_selection"]["selected_on"] == "validation"
    assert summary["policy_selection"]["final_evaluation_on"] == "test"
    assert summary["dataset"]["retained_events"] == 64
    assert summary["policy_selection"]["candidate_configurations"] == 9
    assert summary["policy_selection"]["worker_counts"] == [1, 20]
    report = Path(result["report"]).read_text()
    assert "9-configuration grid" in report
    assert "--workers 1 20 --inflight-limits 1 32" in report
    assert summary["models"]["gpu_runtime_s"]["classes"].keys() >= {"zero", "work"}
    assert summary["limitations"]
    assert len(list(Path(result["figures"]).glob("*.png"))) >= 4
    assert (root / "runs" / "work0" / "events.csv").is_file()
