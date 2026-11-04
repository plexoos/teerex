"""The notebook must use auditable v3 data and independent process splits."""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import refresh_ratio_study as snapshot


EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "study"


def test_production_csv_keeps_all_geometries_and_original_clocks(tmp_path):
    source = pd.read_csv(EVIDENCE / "events.csv.gz")
    path = tmp_path / "events.csv"
    source.to_csv(path, index=False)
    events = snapshot.load_events(path)
    assert len(events) == 22_000
    assert events.run_id.nunique() == 44
    assert events.condition_id.nunique() == 11
    assert events.groupby("cohort").size().to_dict() == {
        "balanced": 2000, "drich": 8000, "gpu_heavy_n10": 2000,
        "gpu_heavy_n100": 2000, "pfrich": 8000,
    }
    assert events.groupby("split").size().to_dict() == {
        "train": 11000, "validation": 5500, "test": 5500,
    }
    np.testing.assert_array_equal(events.start_offset_s, source.start_offset_s)
    assert events.groupby("run_id").size().eq(500).all()
    assert events.groupby("run_id").event_id.min().eq(5).all()


@pytest.mark.parametrize("corruption", ["duplicate", "split", "phase", "seed"])
def test_invalid_combined_csv_is_rejected(tmp_path, corruption):
    events = pd.read_csv(EVIDENCE / "events.csv.gz")
    if corruption == "duplicate":
        events = pd.concat([events, events.iloc[[0]]], ignore_index=True)
    elif corruption == "split":
        events.loc[0, "split"] = "test"
    elif corruption == "phase":
        events.loc[0, "gpu_runtime_s"] = -1
    else:
        run = events.loc[events.replication.eq(1), "run_id"].iloc[0]
        events.loc[events.run_id.eq(run), "clhep_seed"] = events.clhep_seed.iloc[0]
    path = tmp_path / "bad.csv"
    events.to_csv(path, index=False)
    with pytest.raises(ValueError):
        snapshot.load_events(path)


def test_all_evidence_matches_recorded_hashes():
    snapshot.verify(EVIDENCE)
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    raw = gzip.decompress((EVIDENCE / "events.csv.gz").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == manifest["source"]["sha256"]


def test_test_prediction_errors_match_metrics():
    predictions = pd.read_csv(EVIDENCE / "test-predictions.csv.gz")
    metrics = pd.read_csv(EVIDENCE / "heldout-metrics.csv")
    assert set(predictions.split) == {"test"}
    assert len(predictions) == 22_000
    for (cohort, phase), rows in predictions.groupby(["cohort", "phase"]):
        for method, column in [("bayesian", "predicted_s"), ("ols_log", "ols_s"),
                               ("condition_median", "median_s")]:
            metric = metrics.loc[metrics.split.eq("test") & metrics.cohort.eq(cohort)
                                 & metrics.phase.eq(phase) & metrics.method.eq(method)].iloc[0]
            assert np.abs(rows[column] - rows.observed_s).mean() == pytest.approx(metric.mae_s)
        metric = metrics.loc[metrics.split.eq("test") & metrics.cohort.eq(cohort)
                             & metrics.phase.eq(phase) & metrics.method.eq("bayesian")].iloc[0]
        assert rows.observed_s.between(rows.low_95_s, rows.high_95_s).mean() == pytest.approx(metric.coverage_95)


def test_frozen_policies_have_validation_selection_and_disjoint_test_streams():
    policies = json.loads((EVIDENCE / "frozen-policies.json").read_text())
    events = pd.read_csv(EVIDENCE / "events.csv.gz").set_index(["run_id", "event_id"])
    results = pd.read_csv(EVIDENCE / "policy-summary.csv")
    for cohort, policy in policies.items():
        selected = policy["configuration"]
        validation = results.loc[results.cohort.eq(cohort) & results.split.eq("validation")]
        eligible = validation.loc[validation.makespan_median_s <= 1.05 * validation.makespan_median_s.min()]
        expected = eligible.sort_values(["workers", "max_inflight", "policy"]).iloc[0]
        assert selected["configuration"] == expected.configuration
        assert len(validation) == 85
        for split, replication in [("validation", 2), ("test", 3)]:
            stream = events.loc[[tuple(key) for key in policy["streams"][split]]]
            assert len(stream) == 256
            assert set(stream.replication) == {replication}
            assert stream.groupby("condition_id").size().nunique() == 1
        test = results.loc[results.cohort.eq(cohort) & results.split.eq("test")]
        assert set(test.configuration) == {"blocking_w1_q1", selected["configuration"]}


def test_verifier_rejects_modified_evidence(tmp_path):
    (tmp_path / "one.csv").write_text("modified\n")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "files": {"one.csv": {"sha256": "0" * 64}}
    }))
    with pytest.raises(ValueError, match="checksum"):
        snapshot.verify(tmp_path)
