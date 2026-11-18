"""Check the charged-event notebook against the recorded native GPU-thread timings."""

import hashlib
import json
from pathlib import Path
from statistics import geometric_mean

import pandas as pd
import pytest

BUNDLE = Path(__file__).resolve().parents[1] / "data" / "multi_gpu"


@pytest.fixture(scope="module")
def events():
    return pd.read_csv(BUNDLE / "events.csv.gz")


@pytest.fixture(scope="module")
def notebook():
    from notebooks.timelines_multi_gpu import app

    _, definitions = app.run()
    return definitions


def test_bundle_matches_provenance():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == digest, name


def test_raw_timings_preserve_completion_windows_and_gpu_assignment(events):
    runs = pd.read_csv(BUNDLE / "run-summary.csv").set_index("run_id")
    assert len(events) == 4_032
    assert len(runs) == 56
    assert runs.index.is_unique
    assert set(events.run_id) == set(runs.index)
    assert set(events.dispatch_mode) == {"asynchronous"}
    assert runs.retained_events.sum() == 3_584

    for run_id, group in events.groupby("run_id"):
        run = runs.loc[run_id]
        assert sorted(group.event_id) == list(range(72))
        assert set(group.worker_id) == set(range(4))
        assert run.gpu_count == len(set(group.gpu_uuid))
        visible = run.gpu_mask.split(",")
        assert group.gpu_device_id.map(dict(enumerate(visible))).eq(group.gpu_uuid).all()

        completed = group.sort_values(["end_offset_s", "event_id"])
        warmup = int(run.warmup_events)
        assert warmup == 8
        start = completed.end_offset_s.iloc[warmup - 1]
        end = completed.end_offset_s.iloc[-1]
        assert run.window_start_s == pytest.approx(start)
        assert run.events_per_s == pytest.approx((len(group) - warmup) / (end - start))
        assert run.num_photons_total == completed.num_photons.iloc[warmup:].sum()


def test_speedups_pool_four_paired_seeds():
    runs = pd.read_csv(BUNDLE / "run-summary.csv")
    pairs = pd.read_csv(BUNDLE / "paired-summary.csv")
    conditions = pd.read_csv(BUNDLE / "condition-summary.csv")
    key = ["condition", "replication", "clhep_seed", "gpu_seed"]
    by_gpu = runs.pivot(index=key, columns="gpu_count", values="events_per_s")
    expected = (by_gpu[2] / by_gpu[1]).sort_index()
    actual = pairs.set_index(key).speedup.sort_index()
    assert len(actual) == 28
    assert actual.index.is_unique
    pd.testing.assert_series_equal(actual, expected, check_names=False)

    grouped = actual.groupby("condition")
    assert len(conditions) == 14
    assert grouped.size().eq(4).all()
    dual = conditions.loc[conditions.gpu_count.eq(2)].set_index("condition").sort_index()
    pd.testing.assert_series_equal(dual.paired_speedup, grouped.agg(geometric_mean), check_names=False)


def test_notebook_uses_native_gpu_and_producer_timestamps(events, notebook):
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    physical_device = {gpu["uuid"]: gpu["index"] for gpu in manifest["gpus"]}
    defs = notebook
    assert len(defs["configuration_summary"]) == 7
    assert defs["configuration_summary"]["GPUs"].eq(2).all()
    for devices in (1, 2):
        frame = events.loc[events.run_id.eq(f"argon_gpu_n100_r0_g{devices}")]
        fig = defs["make_timeline"](frame, 8)
        by_id = frame.set_index("event_id")
        origin = frame.start_offset_s.min()
        stages = [
            ("start_offset_s", "gpu_submit_time_offset_s"),
            ("gpu_submit_time_offset_s", "cpu_end_time_offset_s"),
            ("gpu_service_start_offset_s", "gpu_service_end_offset_s"),
            ("gpu_service_end_offset_s", "gpu_release_time_offset_s"),
        ]
        assert len(fig.data) == len(stages)
        for trace, (begin, finish) in zip(fig.data, stages):
            event_ids = [int(row[0]) for row in trace.customdata]
            assert sorted(event_ids) == list(range(72))
            ordered = by_id.loc[event_ids]
            assert list(trace.base) == pytest.approx((ordered[begin] - origin) * 1000)
            assert list(trace.x) == pytest.approx((ordered[finish] - ordered[begin]) * 1000)

        gpu_service = fig.data[2]
        for lane, hover in zip(gpu_service.y, gpu_service.customdata):
            uuid = by_id.at[int(hover[0]), "gpu_uuid"]
            assert str(lane) == f"GPU {physical_device[uuid]} · {uuid[4:12]}"
            assert hover[2] == uuid
        assert tuple(fig.data[3].y) == tuple(gpu_service.y)
        expected_lanes = {f"Geant4 {worker}" for worker in range(4)} | set(gpu_service.y)
        assert set(fig.layout.yaxis.categoryarray) == expected_lanes
        assert len(fig.layout.yaxis.categoryarray) == 4 + devices
        cutoff = frame.end_offset_s.nsmallest(8).iloc[-1]
        assert fig.layout.shapes[0].x0 == pytest.approx((cutoff - origin) * 1000)
        assert fig.layout.xaxis.rangeslider.visible
        initial_end = by_id.loc[range(10), "end_offset_s"].max()
        assert tuple(fig.layout.xaxis.range) == pytest.approx((0, (initial_end - origin) * 1000))
        assert fig.layout.xaxis.rangeslider.range[1] == pytest.approx((frame.end_offset_s.max() - origin) * 1000)


def test_histograms_use_two_gpu_completed_samples_and_untrimmed_statistics(events, notebook):
    runs = pd.read_csv(BUNDLE / "run-summary.csv")
    frames = notebook["configuration_frames"]
    figures = notebook["distribution_figures"]
    assert set(frames) == set(runs.condition)
    assert len(figures) == 7
    for condition, figure in zip(frames, figures, strict=True):
        frame = frames[condition]
        selected = runs.loc[runs.condition.eq(condition) & runs.gpu_count.eq(2)]
        assert set(frame.run_id) == set(selected.run_id)
        tables = [trace for trace in figure.data if trace.type == "table"]
        assert len(tables) == 2
        for index, run in enumerate(selected.sort_values("replication").itertuples()):
            retained = events.loc[events.run_id.eq(run.run_id)].sort_values(
                ["end_offset_s", "event_id"]
            ).iloc[run.warmup_events:]
            actual = frame.loc[frame.run_id.eq(run.run_id)]
            assert len(actual) == 64
            assert set(actual.event_id) == set(retained.event_id)
            assert actual.replication.eq(run.replication).all()
            cpu = (
                retained.gpu_submit_time_offset_s - retained.start_offset_s
                + retained.gpu_cleanup_end_offset_s - retained.gpu_service_end_offset_s
            ) * 1000
            gpu = retained.gpu_runtime_s * 1000
            for column, sample in enumerate((cpu, gpu)):
                values = tables[column].cells.values
                assert int(values[0][index]) == run.replication
                shown, total = map(int, values[1][index].split(" / "))
                assert total == len(sample)
                assert float(values[2][index]) == pytest.approx(sample.mean(), rel=5e-4)
                assert float(values[3][index]) == pytest.approx(sample.std(ddof=0), rel=5e-4)
                assert float(values[4][index]) == pytest.approx(
                    100 * sample.std(ddof=0) / sample.mean(), rel=5e-4,
                )
                axis = "x" if column == 0 else "x2"
                histogram = next(
                    trace for trace in figure.data
                    if trace.type == "scatter" and trace.xaxis == axis
                    and trace.legendgroup == str(run.replication)
                )
                in_range = sample.between(histogram.x[0], histogram.x[-1])
                assert shown == in_range.sum()
                assert sum(histogram.y[::2]) == shown
