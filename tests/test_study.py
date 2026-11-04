"""Analysis contracts: independent runs, available features, and trace accounting."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from teerex import study


STUDY_TMP = Path(os.environ.get("TMPDIR", "/workspaces/teerex-study/tmp"))


def write_run(root, run_id, replication=0, *, warmup=1, n=5, photons=100,
              status="complete", arm="physics_production", offset=10.0,
              multiplicity=1):
    directory = Path(root) / "runs" / run_id
    directory.mkdir(parents=True)
    metadata = dict(run_id=run_id, arm=arm, particle="mu-", momentum_gev_c=5.0,
                    multiplicity=multiplicity, replication=replication,
                    clhep_seed=100 + replication, gpu_seed=200 + replication,
                    gpu_offset=0, events=n, warmup_events=warmup,
                    record_timing=True, status=status, duration_wall_s=100,
                    gpu_uuid="GPU-test")
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
        rows.append(dict(scenario="simg4ox_blocking", dispatch_mode="blocking",
                         event_id=event_id, cpu_pre_runtime_s=pre,
                         gpu_runtime_s=gpu, cpu_post_locked_runtime_s=locked,
                         cpu_post_unlocked_runtime_s=unlocked,
                         cpu_post_runtime_s=locked + unlocked,
                         gpu_gate_runtime_s=gpu + locked, gpu_queue_delay_s=0.2,
                         start_offset_s=start, end_offset_s=end,
                         cpu_pre_start_time_offset_s=start,
                         cpu_pre_end_time_offset_s=submit,
                         gpu_submit_time_offset_s=submit,
                         gpu_start_time_offset_s=gpu_start,
                         gpu_end_time_offset_s=gpu_end,
                         gpu_release_time_offset_s=release,
                         cpu_post_end_time_offset_s=end,
                         num_photons=photons, num_gensteps=int(photons > 0),
                         num_gpu_hits=17, num_g4_hits=19))
        start = end + 0.3
    pd.DataFrame(rows).to_csv(directory / "events.csv", index=False)
    (directory / "events.manifest.json").write_text(json.dumps({"event_count": n}))
    (directory / "simg4ox.stdout.log").write_text("controlled fixture\n")
    return directory


class StudyIngestionTest(unittest.TestCase):
    def setUp(self):
        STUDY_TMP.mkdir(parents=True, exist_ok=True)
        self.temp = TemporaryDirectory(dir=STUDY_TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_independent_origins_repeated_ids_and_raw_bytes_are_preserved(self):
        first = write_run(self.root, "first", offset=10)
        second = write_run(self.root, "second", replication=1, offset=1000)
        before = (first / "events.csv").read_bytes()
        frame = study.load_runs(self.root)
        self.assertEqual(len(frame), 10)
        self.assertEqual(len(frame[["run_id", "event_id"]].drop_duplicates()), 10)
        self.assertEqual(frame.scenario.nunique(), 2)
        self.assertEqual(set(frame.source_scenario), {"simg4ox_blocking"})
        self.assertEqual(frame.loc[frame.run_id.eq("second"), "start_offset_s"].min(), 1000)
        self.assertEqual(frame.groupby("run_id").is_warmup.sum().to_dict(), {"first": 1, "second": 1})
        self.assertEqual(before, (first / "events.csv").read_bytes())
        self.assertTrue((second / "events.csv").is_file())

    def test_recorded_raw_checksum_and_manifest_seed_are_enforced(self):
        directory = write_run(self.root, "checksum")
        path = directory / "run.json"
        metadata = json.loads(path.read_text())
        metadata["csv_sha256"] = study._checksum(directory / "events.csv")
        path.write_text(json.dumps(metadata))
        study.load_runs(self.root)
        with (directory / "events.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            study.load_runs(self.root)
        metadata["csv_sha256"] = study._checksum(directory / "events.csv")
        path.write_text(json.dumps(metadata))
        manifest = directory / "events.manifest.json"
        manifest.write_text(json.dumps({"event_count": 5, "random_seed": 999}))
        with self.assertRaisesRegex(ValueError, "seed"):
            study.load_runs(self.root)

    def test_replications_cannot_reuse_clhep_or_gpu_streams(self):
        write_run(self.root, "r0", replication=0)
        directory = write_run(self.root, "r1", replication=1)
        path = directory / "run.json"
        original = json.loads(path.read_text())
        for key, value in [("clhep_seed", 100), ("gpu_seed", 200)]:
            metadata = {**original, key: value}
            path.write_text(json.dumps(metadata))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "reus"):
                study.load_runs(self.root)

    def test_failed_and_other_arm_runs_are_excluded(self):
        write_run(self.root, "good")
        write_run(self.root, "failed", status="failed")
        write_run(self.root, "pilot", arm="physics_pilot")
        self.assertEqual(set(study.load_runs(self.root).run_id), {"good"})
        self.assertEqual(set(study.load_runs(self.root, arm="physics_pilot").run_id), {"pilot"})

    def test_invalid_durations_duplicate_ids_and_incomplete_count_are_rejected(self):
        directory = write_run(self.root, "invalid")
        source = pd.read_csv(directory / "events.csv")
        for column, value in [("gpu_runtime_s", -1), ("cpu_pre_runtime_s", np.inf),
                              ("event_id", 0)]:
            bad = source.copy()
            bad.loc[1, column] = value
            bad.to_csv(directory / "events.csv", index=False)
            with self.subTest(column=column), self.assertRaises(ValueError):
                study.load_runs(self.root)
        source.iloc[:-1].to_csv(directory / "events.csv", index=False)
        with self.assertRaises(ValueError):
            study.load_runs(self.root)

    def test_metadata_disagreement_and_inconsistent_phase_boundaries_are_rejected(self):
        directory = write_run(self.root, "mismatch")
        source = pd.read_csv(directory / "events.csv")
        for column, value in [("primary_particle", "e-"),
                              ("primary_momentum_gev_c", 20),
                              ("primary_multiplicity", 128),
                              ("gpu_runtime_s", 200)]:
            bad = source.copy()
            bad[column] = value
            bad.to_csv(directory / "events.csv", index=False)
            with self.subTest(column=column), self.assertRaises(ValueError):
                study.load_runs(self.root)
        source.to_csv(directory / "events.csv", index=False)
        (directory / "events.manifest.json").write_text(json.dumps({"event_count": 5, "primary_particle": "e-"}))
        with self.assertRaises(ValueError):
            study.load_runs(self.root)

    def test_splits_keep_whole_replications_and_exclude_only_marked_warmup(self):
        for replication in range(4):
            write_run(self.root, f"run{replication}", replication)
        frame = study.load_runs(self.root)
        train, val, test = study.split_by_replication(frame)
        self.assertEqual(set(train.replication), {0, 1})
        self.assertEqual(set(val.replication), {2})
        self.assertEqual(set(test.replication), {3})
        self.assertEqual([len(train), len(val), len(test)], [8, 4, 4])
        self.assertFalse(any(part.is_warmup.any() for part in (train, val, test)))
        self.assertTrue(set(train.run_id).isdisjoint(test.run_id))
        frame.loc[0, "replication"] = 3
        with self.assertRaises(ValueError):
            study.split_by_replication(frame)

    def test_zero_photon_and_zero_timer_rows_are_retained(self):
        directory = write_run(self.root, "zero", photons=0)
        source = pd.read_csv(directory / "events.csv")
        source.loc[4, "cpu_post_unlocked_runtime_s"] = 0
        source.loc[4, "end_offset_s"] = source.loc[4, "gpu_release_time_offset_s"]
        source.loc[4, "cpu_post_end_time_offset_s"] = source.loc[4, "gpu_release_time_offset_s"]
        source.loc[4, "cpu_post_runtime_s"] = source.loc[4, "cpu_post_locked_runtime_s"]
        source.to_csv(directory / "events.csv", index=False)
        frame = study.load_runs(self.root)
        self.assertEqual(len(frame), 5)
        self.assertTrue(frame.zero_photon.all())
        self.assertEqual(frame.cpu_post_unlocked_runtime_s.iloc[4], 0)


class StudyFeatureTest(unittest.TestCase):
    def training(self):
        return pd.DataFrame(dict(particle=["mu-", "e-", "mu-", "e-"],
                                 momentum_gev_c=[1, 5, 20, 5], multiplicity=[1, 16, 128, 16],
                                 num_photons=[0, 10, 100, 10], num_gensteps=[0, 1, 2, 1],
                                 num_gpu_hits=[1, 2, 3, 4], gpu_runtime_s=[1, 2, 3, 4]))

    def test_pre_features_ignore_photons_hits_and_future_durations(self):
        train = self.training()
        transform = study.FeatureTransform.fit(train, stage="cpu_pre")
        changed = train.copy()
        changed[["num_photons", "num_gensteps", "num_gpu_hits", "gpu_runtime_s"]] *= 1000
        np.testing.assert_array_equal(transform.transform(train), transform.transform(changed))
        self.assertFalse(any("photon" in name or "hit" in name for name in transform.feature_names))
        gpu = study.FeatureTransform.fit(train, stage="gpu")
        self.assertFalse(np.array_equal(gpu.transform(train), gpu.transform(changed)))
        hit_only = train.copy()
        hit_only[["num_gpu_hits", "gpu_runtime_s"]] *= 1000
        np.testing.assert_array_equal(gpu.transform(train), gpu.transform(hit_only))

    def test_normalization_is_training_only_and_persistence_reproduces_features(self):
        train = self.training()
        transform = study.FeatureTransform.fit(train, stage="gpu")
        features = transform.transform(train)
        np.testing.assert_allclose(features[:, 1:5].mean(axis=0), 0, atol=1e-15)
        heldout = train.copy()
        heldout["momentum_gev_c"] = 10000
        self.assertGreater(transform.transform(heldout)[:, 1].mean(), 1)
        restored = study.FeatureTransform.from_dict(transform.to_dict())
        np.testing.assert_array_equal(restored.transform(heldout), transform.transform(heldout))
        heldout["particle"] = "pi-"
        with self.assertRaises(ValueError):
            transform.transform(heldout)


class StudyTraceTest(unittest.TestCase):
    def test_service_replay_and_gaps_are_accounted_separately_per_run(self):
        # Span: 8 service + 0.4 submission overhead + 0.3 gap = 8.7.
        records = []
        for run_id, origin in [("a", 10), ("b", 1000)]:
            for event_id, start in [(0, origin), (1, origin + 4.5)]:
                records.append(dict(run_id=run_id, event_id=event_id,
                                    cpu_pre_runtime_s=1, gpu_runtime_s=2,
                                    cpu_post_locked_runtime_s=.5, cpu_post_unlocked_runtime_s=.5,
                                    start_offset_s=start, end_offset_s=start + 4.2,
                                    gpu_queue_delay_s=.2, num_photons=10))
        result = study.validate_blocking_trace(pd.DataFrame(records))
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result.service_replay_makespan_s, [8, 8])
        np.testing.assert_allclose(result.measured_span_s, [8.7, 8.7])
        np.testing.assert_allclose(result.interevent_gap_s, [.3, .3])
        np.testing.assert_allclose(result.submission_overhead_s, [.4, .4])
        np.testing.assert_allclose(result.reconstruction_error_s, [0, 0], atol=1e-12)


class StudyModelTest(unittest.TestCase):
    def test_full_posterior_grid_survives_observed_prediction_rank_reversal(self):
        frame = pd.DataFrame({'num_photons': [1] * 30})
        observed_demands = np.tile([1., 100., .1, .1], (30, 1))
        observed = study._simulate(frame, observed_demands, study._candidates(),
                                   split='validation', stream='balanced_mix')
        predictive = np.tile([100., 1., .1, .1], (1, 30, 1))
        draws, summary, chosen = study.select_posterior_policy(frame, predictive)
        self.assertEqual(draws.configuration.nunique(), len(study._candidates()))
        self.assertEqual(chosen['configuration'], 'async_w8_q8')
        self.assertLess(summary.makespan_median_s.min(), 407.)
        self.assertIn('async_w8_q32', set(draws.configuration))

    def test_split_keeps_run_provenance_at_dataset_level(self):
        with TemporaryDirectory(dir=STUDY_TMP) as temporary:
            for replication in range(4):
                write_run(temporary, f'run{replication}', replication)
            raw = study.load_runs(temporary)
            self.assertIn('run_index', raw.attrs)
            for split in study.split_by_replication(raw):
                self.assertNotIn('run_index', split.attrs)
            self.assertEqual(len(raw.attrs['run_index']), 4)

    def test_condition_coverage_does_not_hide_a_failing_subgroup(self):
        rows = pd.DataFrame(dict(split=["test"] * 4, phase=["gpu_runtime_s"] * 4,
                                 condition_id=["small", "small", "large", "large"],
                                 work_class=["work"] * 4, observed_s=[1.] * 4,
                                 predicted_s=[1.] * 4,
                                 low_50_s=[.5, .5, .5, 2.], high_50_s=[2., 2., 2., 4.],
                                 low_80_s=[.5, .5, .5, 2.], high_80_s=[2., 2., 2., 4.],
                                 low_95_s=[.5, .5, .5, 2.], high_95_s=[2., 2., 2., 4.]))
        result = study.summarize_condition_predictions(rows).set_index("condition_id")
        self.assertEqual(result.loc["small", "coverage_95"], 1.)
        self.assertEqual(result.loc["large", "coverage_95"], .5)
        self.assertEqual(result.loc["large", "n"], 2)
        self.assertEqual(result.loc["small", "rmse_s"], 0.)

    def test_heldout_point_errors_use_exact_median_and_unseen_work_class_is_rejected(self):
        with TemporaryDirectory(dir=STUDY_TMP) as temporary:
            root = Path(temporary)
            for replication in range(4):
                write_run(root, f"run{replication}", replication, n=9)
            train, val, _ = study.split_by_replication(study.load_runs(root))
            bundle = study.fit_duration_models(train)
            predictions, _ = study.evaluate_models(bundle, val, "validation", draws=2)
            phase = "cpu_pre_runtime_s"
            X = bundle.transforms[phase].transform(val)
            mean, _, _ = bundle.models[phase]["all"].predict(X)
            np.testing.assert_allclose(predictions.loc[predictions.phase.eq(phase), "predicted_s"], np.exp(mean), rtol=1e-14)
            no_work = val.copy()
            no_work["num_photons"] = 0
            with self.assertRaisesRegex(ValueError, "unseen"):
                bundle.sample(no_work, 2, np.random.default_rng(42))


class StudyGridTest(unittest.TestCase):
    def test_expanded_cpu_budget_is_used_for_posterior_selection(self):
        workers = (1, 2, 4, 8, 16, 20)
        caps = (1, 2, 4, 8, 16, 32, 64)
        frame = pd.DataFrame({"num_photons": [100] * 80})
        draws = np.tile([100., 1., .1, .1], (1, 80, 1))
        simulations, summary, chosen = study.select_posterior_policy(
            frame, draws, worker_counts=workers, inflight_limits=caps)
        self.assertEqual(simulations.configuration.nunique(), 85)
        self.assertEqual(len(summary), 85)
        self.assertEqual(chosen["workers"], 20)
        self.assertGreaterEqual(chosen["max_inflight"], 20)
        self.assertIn("async_w20_q64", set(simulations.configuration))

    def test_invalid_resource_grids_are_rejected(self):
        for kwargs in ({"worker_counts": ()}, {"worker_counts": (1, 0)},
                       {"worker_counts": (True, 2)}, {"worker_counts": (1, 1)},
                       {"inflight_limits": ()}, {"inflight_limits": (1, -1)},
                       {"inflight_limits": (1, 1.5)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                study._candidates(**kwargs)


class StudyPipelineTest(unittest.TestCase):
    def test_end_to_end_report_freezes_validation_policy_and_retains_no_work_class(self):
        with TemporaryDirectory(dir=STUDY_TMP) as temporary:
            root = Path(temporary)
            for replication in range(4):
                write_run(root, f"work{replication}", replication, n=9)
                write_run(root, f"empty{replication}", replication, photons=0, n=9,
                          multiplicity=16)
            result = study.run_study(root, draws=4, worker_counts=(1, 20),
                                     inflight_limits=(1, 32))
            self.assertTrue(Path(result["report"]).is_file())
            summary = json.loads(Path(result["summary"]).read_text())
            self.assertEqual(summary["splits"]["train_replications"], [0, 1])
            self.assertEqual(summary["policy_selection"]["selected_on"], "validation")
            self.assertEqual(summary["policy_selection"]["final_evaluation_on"], "test")
            self.assertEqual(summary["dataset"]["retained_events"], 64)
            self.assertEqual(summary["policy_selection"]["candidate_configurations"], 9)
            self.assertEqual(summary["policy_selection"]["worker_counts"], [1, 20])
            report = Path(result["report"]).read_text()
            self.assertIn("9-configuration grid", report)
            self.assertIn("--workers 1 20 --inflight-limits 1 32", report)
            self.assertTrue(summary["models"]["gpu_runtime_s"]["classes"].keys() >= {"zero", "work"})
            self.assertTrue(summary["limitations"])
            self.assertGreaterEqual(len(list(Path(result["figures"]).glob("*.png"))), 4)
            self.assertTrue((root / "runs" / "work0" / "events.csv").is_file())


if __name__ == "__main__":
    unittest.main()
