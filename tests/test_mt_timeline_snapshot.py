"""Protect the measured MT snapshot's bytes, clocks, and replication units."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "mt-study"


@pytest.fixture(scope="module")
def snapshot():
    return (
        pd.read_csv(EVIDENCE / "events-raw.csv.gz"),
        pd.read_csv(EVIDENCE / "run-summary.csv"),
    )


def test_snapshot_files_match_provenance():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        data = (EVIDENCE / name).read_bytes()
        assert len(data) == record["bytes"], name
        assert hashlib.sha256(data).hexdigest() == record["sha256"], name


def test_raw_events_preserve_all_runs_warmup_and_mt_histories(snapshot):
    events, runs = snapshot
    assert len(events) == 57_792
    assert len(runs) == events.run_id.nunique() == 112
    assert not events.duplicated(["run_id", "event_id"]).any()
    assert set(events.num_threads) == {1, 2, 4, 8}
    assert set(events.replication) == {0, 1, 2, 3}
    assert events.workload.nunique() == 7
    assert runs.groupby(["workload", "num_threads"]).size().eq(4).all()
    assert events.event_id.ge(16).sum() == runs.analysis_events.sum() == 56_000
    for _, raw in events.groupby("run_id"):
        assert sorted(raw.event_id) == list(range(516))
        assert raw.is_warmup.eq(raw.event_id.lt(16)).all()
        workers = int(raw.num_threads.iloc[0])
        assert set(raw.worker_id) == ({-1} if workers == 1 else set(range(workers)))
    counts = ["event_id", "num_gensteps", "num_photons", "num_gpu_hits", "num_g4_hits"]
    retained = events.loc[events.event_id.ge(16)]
    for _, group in retained.groupby(["workload", "replication"]):
        reference = group.loc[group.num_threads.eq(2)].sort_values("event_id")
        for workers in (4, 8):
            actual = group.loc[group.num_threads.eq(workers)].sort_values("event_id")
            np.testing.assert_array_equal(actual[counts], reference[counts])


def test_completion_rates_and_stall_flags_match_raw_events(snapshot):
    events, runs = snapshot
    for run in runs.itertuples():
        raw = events.loc[events.run_id.eq(run.run_id)].set_index("event_id")
        window = raw.loc[515, "gpu_release_time_offset_s"] - raw.loc[15, "gpu_release_time_offset_s"]
        assert run.events_per_second == pytest.approx(500 / window)
        retained = raw.loc[16:]
        flags = (retained.cpu_post_runtime_s > .05) & (retained.thread_cpu_post_s < .05 * retained.cpu_post_runtime_s)
        assert run.cleanup_stall_events == flags.sum()
        serial = runs.loc[runs.workload.eq(run.workload) & runs.replication.eq(run.replication) & runs.num_threads.eq(1)].iloc[0]
        assert run.speedup == pytest.approx(run.events_per_second / serial.events_per_second)
    assert runs.cleanup_stall_events.sum() == 64
    assert (runs.loc[runs.cleanup_stall_events.gt(0), "num_threads"] == 1).all()
