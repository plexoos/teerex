import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from teerex.analysis import plot_run_timeline, read_one_csv, timeline_summary
from teerex.replay import replay


def event(event_id, pre=1, gpu=2, locked=1, unlocked=1, photons=10):
    return {
        "event_id": event_id,
        "cpu_pre_runtime_s": pre,
        "gpu_runtime_s": gpu,
        "cpu_post_locked_runtime_s": locked,
        "cpu_post_unlocked_runtime_s": unlocked,
        "num_photons": photons,
    }


def test_one_event_has_four_dependent_stages_and_visible_backlog():
    timeline = replay(
        [event(7, 2, 3, 4, 5), event(9, 1, 1, 1, 1)], workers=4, max_inflight=8
    )
    columns = [
        "admission_time_offset_s",
        "cpu_pre_start_time_offset_s",
        "cpu_pre_end_time_offset_s",
        "gpu_start_time_offset_s",
        "gpu_end_time_offset_s",
        "gpu_release_time_offset_s",
        "cpu_post_end_time_offset_s",
        "latency_s",
        "service_latency_s",
    ]
    np.testing.assert_allclose(
        timeline[columns],
        [[0, 0, 2, 2, 5, 9, 14, 14, 14], [14, 14, 15, 15, 16, 17, 18, 18, 4]],
    )
    assert timeline["release_time_offset_s"].tolist() == [0, 0]
    assert timeline["cpu_runtime_s"].tolist() == [11, 3]
    assert timeline["gpu_gate_runtime_s"].tolist() == [7, 2]
    # The envelope includes the GPU wait and must not count as CPU demand.
    assert timeline.iloc[0]["cpu_end_time_offset_s"] == 14
    assert timeline.iloc[0]["cpu_runtime_s"] == 11


def test_synchronous_holds_workers_through_gpu_and_cleanup():
    timeline = replay(
        [event(i) for i in range(3)], workers=2, max_inflight=3, policy="synchronous"
    )
    columns = [
        "admission_time_offset_s",
        "gpu_start_time_offset_s",
        "gpu_end_time_offset_s",
        "gpu_release_time_offset_s",
        "cpu_post_end_time_offset_s",
    ]
    np.testing.assert_allclose(
        timeline[columns], [[0, 1, 3, 4, 5], [0, 4, 6, 7, 8], [5, 7, 9, 10, 11]]
    )
    assert timeline["worker_id"].tolist() == [0, 1, 0]


def test_synchronous_uses_input_gpu_order_despite_pre_completion_order():
    events = [event(30, 5, 1, 1, 0), event(10, 1, 1, 1, 0)]
    timeline = replay(events, workers=2, max_inflight=2, policy="synchronous")
    assert timeline["gpu_submit_time_offset_s"].tolist() == [5, 1]
    assert timeline["gpu_start_time_offset_s"].tolist() == [5, 7]
    assert timeline["gpu_queue_delay_s"].tolist() == [0, 6]


def test_synchronous_batches_reproduce_head_of_line_blocking():
    events = [event(i, 10, 1, 0, 0) for i in range(4)]
    individual = replay(events, workers=2, max_inflight=2, policy="synchronous")
    batched = replay(
        events, workers=2, max_inflight=2, policy="synchronous", event_batch_size=2
    )
    assert individual["latency_s"].max() == 23
    assert batched["latency_s"].max() == 34
    columns = [
        "worker_id",
        "admission_time_offset_s",
        "cpu_pre_start_time_offset_s",
        "gpu_start_time_offset_s",
        "gpu_end_time_offset_s",
    ]
    np.testing.assert_allclose(
        batched[columns],
        [
            [0, 0, 0, 10, 11],
            [0, 11, 11, 21, 22],
            [1, 0, 0, 22, 23],
            [1, 23, 23, 33, 34],
        ],
    )
    assert batched["event_batch_size"].tolist() == [2, 2, 2, 2]
    assert individual.iloc[0]["scenario"] != batched.iloc[0]["scenario"]


@pytest.mark.parametrize("policy", ("blocking", "synchronous", "async"))
def test_batch_size_one_preserves_every_default_policy(policy):
    events = [event(4, 3, 1, 2, 1), event(2, 1, 3, 1, 0), event(9)]
    kwargs = dict(workers=2, max_inflight=3, policy=policy)
    pd.testing.assert_frame_equal(
        replay(events, **kwargs), replay(events, event_batch_size=1, **kwargs)
    )


def test_synchronous_batch_nondivisible_tail_keeps_its_worker():
    timeline = replay(
        [event(i, 10, 1, 0, 0) for i in range(5)],
        workers=2,
        max_inflight=2,
        policy="synchronous",
        event_batch_size=2,
    )
    assert timeline["worker_id"].tolist() == [0, 0, 1, 1, 0]
    assert timeline["admission_time_offset_s"].tolist() == [0, 11, 0, 23, 22]
    assert timeline["gpu_start_time_offset_s"].tolist() == [10, 21, 22, 33, 34]
    assert timeline["latency_s"].max() == 35


@pytest.mark.parametrize("workers, cap", [(1, 8), (4, 1)])
def test_synchronous_batches_with_one_worker_or_one_active_event(workers, cap):
    events = [event(i, 2, 1, 1, 1) for i in range(5)]
    timeline = replay(
        events,
        workers=workers,
        max_inflight=cap,
        policy="synchronous",
        event_batch_size=2,
    )
    assert timeline["admission_time_offset_s"].tolist() == [0, 5, 10, 15, 20]
    assert timeline["latency_s"].tolist() == [5, 10, 15, 20, 25]
    assert timeline["worker_id"].tolist() == [0, 0, 0, 0, 0]


@pytest.mark.parametrize("cap", (1, 2, 4))
def test_synchronous_batches_drain_zero_phases_and_empty_input(cap):
    timeline = replay(
        [event(i, 0, 0, 0, 0) for i in range(7)],
        workers=4,
        max_inflight=cap,
        policy="synchronous",
        event_batch_size=3,
    )
    assert timeline["latency_s"].tolist() == [0] * 7
    empty = replay([], policy="synchronous", event_batch_size=3)
    assert empty.empty
    assert "event_batch_size" in empty


def synchronous_batch_cases():
    rng = random.Random(834)
    for workers in (1, 2, 4):
        for cap in (1, 2, 4):
            for batch in (2, 3, 7):
                events = [
                    event(i, *(rng.randrange(4) for _ in range(4))) for i in range(11)
                ]
                yield pytest.param(
                    workers, cap, batch, events, id=f"w{workers}-q{cap}-b{batch}"
                )


@pytest.mark.parametrize(
    "workers, cap, batch, events", tuple(synchronous_batch_cases())
)
def test_synchronous_batch_random_schedules_obey_capacity_and_reservations(
    workers, cap, batch, events
):
    timeline = replay(
        events,
        workers=workers,
        max_inflight=cap,
        policy="synchronous",
        event_batch_size=batch,
    )
    repeated = replay(
        events,
        workers=workers,
        max_inflight=cap,
        policy="synchronous",
        event_batch_size=batch,
    )
    pd.testing.assert_frame_equal(timeline, repeated)
    spans = list(
        zip(timeline["admission_time_offset_s"], timeline["cpu_post_end_time_offset_s"])
    )
    assert peak(spans) <= min(workers, cap)
    for _, group in timeline.groupby("worker_id"):
        spans = list(
            zip(
                group["cpu_pre_start_time_offset_s"],
                group["cpu_post_end_time_offset_s"],
            )
        )
        assert peak(spans) <= 1
    for start in range(0, len(events), batch):
        assert timeline.iloc[start : start + batch]["worker_id"].nunique() == 1
    assert np.all(
        timeline["gpu_start_time_offset_s"].to_numpy()[1:]
        >= timeline["gpu_release_time_offset_s"].to_numpy()[:-1]
    )
    for _, row in timeline.iterrows():
        boundaries = row[
            [
                "admission_time_offset_s",
                "cpu_pre_start_time_offset_s",
                "cpu_pre_end_time_offset_s",
                "gpu_start_time_offset_s",
                "gpu_end_time_offset_s",
                "cpu_post_start_time_offset_s",
                "gpu_release_time_offset_s",
                "cpu_post_end_time_offset_s",
            ]
        ]
        assert np.all(np.diff(boundaries.to_numpy(dtype=float)) >= 0)
        for phase in ("pre", "post_locked", "post_unlocked"):
            assert (
                row[f"cpu_{phase}_end_time_offset_s"]
                - row[f"cpu_{phase}_start_time_offset_s"]
                == row[f"cpu_{phase}_runtime_s"]
            )


@pytest.mark.parametrize("batch", [0, -1, True, 1.5, None, "2"])
def test_invalid_event_batches_and_other_policies_are_rejected(batch):
    with pytest.raises(ValueError):
        replay([], policy="synchronous", event_batch_size=batch)


@pytest.mark.parametrize("policy", ["blocking", "async"])
def test_event_batches_are_rejected_for_other_policies(policy):
    with pytest.raises(ValueError):
        replay([], policy=policy, event_batch_size=2)


def test_async_submits_fifo_without_future_duration_ordering():
    timeline = replay(
        [event(30, 5, 1, 1, 0), event(10, 1, 1, 1, 0)],
        workers=2,
        max_inflight=2,
        policy="async",
    )
    assert timeline["gpu_start_time_offset_s"].tolist() == [5, 1]
    assert timeline["gpu_release_time_offset_s"].tolist() == [7, 3]


def test_async_releases_waiting_worker_and_prioritizes_ready_post():
    timeline = replay(
        [event(i) for i in range(3)], workers=1, max_inflight=3, policy="async"
    )
    columns = [
        "cpu_pre_start_time_offset_s",
        "cpu_pre_end_time_offset_s",
        "gpu_start_time_offset_s",
        "gpu_end_time_offset_s",
        "cpu_post_start_time_offset_s",
        "gpu_release_time_offset_s",
        "cpu_post_end_time_offset_s",
    ]
    np.testing.assert_allclose(
        timeline[columns],
        [[0, 1, 1, 3, 3, 4, 5], [1, 2, 4, 6, 6, 7, 8], [2, 3, 7, 9, 9, 10, 11]],
    )
    # With a fourth pre task waiting at t=3, ready post takes the worker.
    more = replay(
        [event(i) for i in range(4)], workers=1, max_inflight=4, policy="async"
    )
    assert more.iloc[3]["cpu_pre_start_time_offset_s"] == 5


def test_async_gate_includes_wait_for_cpu_and_locked_post():
    events = [event(0, 1, 1, 2, 0), event(1, 5, 1, 1, 0)]
    timeline = replay(events, workers=1, max_inflight=2, policy="async")
    assert timeline["gpu_end_time_offset_s"].tolist() == [2, 9]
    assert timeline["cpu_post_start_time_offset_s"].tolist() == [6, 9]
    assert timeline["gpu_release_time_offset_s"].tolist() == [8, 10]
    assert timeline["gpu_start_time_offset_s"].tolist() == [1, 8]
    assert timeline["gpu_gate_runtime_s"].tolist() == [7, 2]
    assert timeline["gpu_runtime_s"].tolist() == [1, 1]


def test_max_inflight_preserves_admission_delay_and_latency():
    timeline = replay(
        [event(i) for i in range(3)], workers=2, max_inflight=1, policy="async"
    )
    assert timeline["admission_time_offset_s"].tolist() == [0, 5, 10]
    assert timeline["latency_s"].tolist() == [5, 10, 15]
    assert timeline["service_latency_s"].tolist() == [5, 5, 5]


@pytest.mark.parametrize("policy", ("blocking", "synchronous", "async"))
def test_all_policies_respect_dependencies_wip_cpu_and_gpu_capacity(policy):
    # Independent interval sweeps catch worker overbooking and early gate release.
    events = [
        event(i, pre, gpu, locked, unlocked)
        for i, (pre, gpu, locked, unlocked) in enumerate(
            [
                (4, 2, 1, 5),
                (1, 3, 2, 1),
                (2, 1, 4, 2),
                (3, 4, 0, 1),
                (1, 1, 2, 0),
                (2, 2, 1, 3),
            ]
        )
    ]
    timeline = replay(events, workers=2, max_inflight=3, policy=policy)
    assert timeline["event_id"].tolist() == list(range(6))
    cpu_intervals = []
    worker_intervals = {}
    wip_intervals = []
    for _, row in timeline.iterrows():
        boundaries = row[
            [
                "admission_time_offset_s",
                "cpu_pre_start_time_offset_s",
                "cpu_pre_end_time_offset_s",
                "gpu_start_time_offset_s",
                "gpu_end_time_offset_s",
                "cpu_post_start_time_offset_s",
                "gpu_release_time_offset_s",
                "cpu_post_end_time_offset_s",
            ]
        ]
        assert np.all(np.diff(boundaries.to_numpy(dtype=float)) >= 0)
        for phase in ("pre", "post"):
            begin = row[f"cpu_{phase}_start_time_offset_s"]
            end = row[f"cpu_{phase}_end_time_offset_s"]
            cpu_intervals.append((begin, end))
            worker = row[f"cpu_{phase}_worker_id"]
            worker_intervals.setdefault(worker, []).append((begin, end))
        wip_intervals.append(
            (row["admission_time_offset_s"], row["cpu_post_end_time_offset_s"])
        )
    assert peak(cpu_intervals) <= 2
    for intervals in worker_intervals.values():
        assert peak(intervals) <= 1
    limit = 1 if policy == "blocking" else (2 if policy == "synchronous" else 3)
    assert peak(wip_intervals) <= limit
    gate = timeline.sort_values("gpu_start_time_offset_s")
    assert np.all(
        gate["gpu_start_time_offset_s"].to_numpy()[1:]
        >= gate["gpu_release_time_offset_s"].to_numpy()[:-1]
    )
    if policy == "synchronous":
        for _, group in timeline.groupby("worker_id"):
            occupied = list(
                zip(
                    group["cpu_pre_start_time_offset_s"],
                    group["cpu_post_end_time_offset_s"],
                )
            )
            assert peak(occupied) <= 1


def peak(intervals):
    changes = sorted(
        [(start, 1) for start, end in intervals if end > start]
        + [(end, -1) for start, end in intervals if end > start]
    )
    count = peak = 0
    for _, delta in changes:
        count += delta
        peak = max(peak, count)
    return peak


@pytest.mark.parametrize("policy", ("blocking", "synchronous", "async"))
def test_zero_durations_and_zero_photons_terminate(policy):
    events = [event(0, 0, 0, 0, 0, 0), event(1, 0, 0, 2, 0, 0), event(2, 0, 0, 0, 0, 0)]
    timeline = replay(events, workers=1, max_inflight=3, policy=policy)
    assert timeline["latency_s"].tolist() == [0, 2, 2]
    assert timeline["num_photons"].tolist() == [0, 0, 0]
    many = replay(
        [event(i, 0, 0, 0, 0, 0) for i in range(1000)],
        workers=8,
        max_inflight=32,
        policy="async",
    )
    assert many["latency_s"].sum() == 0


def test_empty_has_shared_timeline_schema():
    timeline = replay([])
    assert timeline.empty
    for column in (
        "scenario",
        "event_id",
        "cpu_start_time_offset_s",
        "cpu_end_time_offset_s",
        "gpu_start_time_offset_s",
        "gpu_end_time_offset_s",
        "gpu_release_time_offset_s",
        "latency_s",
    ):
        assert column in timeline


def test_deterministic_without_mutating_inputs():
    events = [event(3), event(1), event(8)]
    original = deepcopy(events)
    first = replay(events, workers=2, max_inflight=3, policy="async")
    second = replay(events, workers=2, max_inflight=3, policy="async")
    pd.testing.assert_frame_equal(first, second)
    assert events == original


def test_replay_is_compatible_with_csv_summary_and_plot(tmp_path):
    timeline = replay([event(0), event(1)], workers=1, max_inflight=2, policy="async")
    path = tmp_path / "timeline.csv"
    timeline.to_csv(path, index=False)
    loaded = read_one_csv(path)
    summary = timeline_summary(loaded).iloc[0]
    assert summary["makespan_s"] == 8
    assert summary["cpu_busy_s"] == 6
    assert summary["gpu_busy_s"] == 4
    figure, _, plotted = plot_run_timeline(loaded, loaded.iloc[0]["scenario"])
    assert len(plotted) == 2


def test_unrepresentable_durations_fail_as_validation_errors():
    with pytest.raises(ValueError):
        replay([event(0, pre=10**400)])
    with pytest.raises(ValueError):
        replay([event(0, pre=1e308, gpu=1e308)])


def invalid_events():
    bad_events = [
        None,
        [None],
        [{}],
        [event(1), event(1)],
        [event(-1)],
        [event(True)],
        [event(1, photons=-1)],
        [event(1, photons=1.5)],
    ]
    for field in (
        "cpu_pre_runtime_s",
        "gpu_runtime_s",
        "cpu_post_locked_runtime_s",
        "cpu_post_unlocked_runtime_s",
    ):
        for bad in (-1, float("nan"), float("inf"), "1", True, None):
            item = event(0)
            item[field] = bad
            bad_events.append([item])
    return bad_events


@pytest.mark.parametrize("items", invalid_events())
def test_invalid_inputs_raise_clear_value_errors(items):
    with pytest.raises(ValueError):
        replay(items)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"workers": 0},
        {"workers": True},
        {"workers": 1.5},
        {"max_inflight": 0},
        {"max_inflight": -1},
        {"max_inflight": True},
        {"max_inflight": 1.5},
        {"policy": "event_barrier"},
        {"policy": None},
    ),
)
def test_invalid_resource_and_policy_settings_raise_value_errors(kwargs):
    with pytest.raises(ValueError):
        replay([], **kwargs)
