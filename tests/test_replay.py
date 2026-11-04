from copy import deepcopy
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


class ReplayTest(unittest.TestCase):
    def test_one_event_has_four_dependent_stages_and_visible_backlog(self):
        timeline = replay([event(7, 2, 3, 4, 5), event(9, 1, 1, 1, 1)],
                          workers=4, max_inflight=8)
        columns = ["admission_time_offset_s", "cpu_pre_start_time_offset_s",
                   "cpu_pre_end_time_offset_s", "gpu_start_time_offset_s",
                   "gpu_end_time_offset_s", "gpu_release_time_offset_s",
                   "cpu_post_end_time_offset_s", "latency_s", "service_latency_s"]
        np.testing.assert_allclose(timeline[columns],
                                   [[0, 0, 2, 2, 5, 9, 14, 14, 14],
                                    [14, 14, 15, 15, 16, 17, 18, 18, 4]])
        self.assertEqual(timeline["release_time_offset_s"].tolist(), [0, 0])
        self.assertEqual(timeline["cpu_runtime_s"].tolist(), [11, 3])
        self.assertEqual(timeline["gpu_gate_runtime_s"].tolist(), [7, 2])
        # The envelope includes the GPU wait and must not count as CPU demand.
        self.assertEqual(timeline.iloc[0]["cpu_end_time_offset_s"], 14)
        self.assertEqual(timeline.iloc[0]["cpu_runtime_s"], 11)

    def test_synchronous_holds_workers_through_gpu_and_cleanup(self):
        timeline = replay([event(i) for i in range(3)], workers=2,
                          max_inflight=3, policy="synchronous")
        columns = ["admission_time_offset_s", "gpu_start_time_offset_s",
                   "gpu_end_time_offset_s", "gpu_release_time_offset_s",
                   "cpu_post_end_time_offset_s"]
        np.testing.assert_allclose(timeline[columns],
                                   [[0, 1, 3, 4, 5], [0, 4, 6, 7, 8],
                                    [5, 7, 9, 10, 11]])
        self.assertEqual(timeline["worker_id"].tolist(), [0, 1, 0])

    def test_synchronous_uses_input_gpu_order_despite_pre_completion_order(self):
        events = [event(30, 5, 1, 1, 0), event(10, 1, 1, 1, 0)]
        timeline = replay(events, workers=2, max_inflight=2, policy="synchronous")
        self.assertEqual(timeline["gpu_submit_time_offset_s"].tolist(), [5, 1])
        self.assertEqual(timeline["gpu_start_time_offset_s"].tolist(), [5, 7])
        self.assertEqual(timeline["gpu_queue_delay_s"].tolist(), [0, 6])

    def test_synchronous_batches_reproduce_head_of_line_blocking(self):
        events = [event(i, 10, 1, 0, 0) for i in range(4)]
        individual = replay(events, workers=2, max_inflight=2, policy="synchronous")
        batched = replay(events, workers=2, max_inflight=2, policy="synchronous",
                         event_batch_size=2)
        self.assertEqual(individual["latency_s"].max(), 23)
        self.assertEqual(batched["latency_s"].max(), 34)
        columns = ["worker_id", "admission_time_offset_s",
                   "cpu_pre_start_time_offset_s", "gpu_start_time_offset_s",
                   "gpu_end_time_offset_s"]
        np.testing.assert_allclose(batched[columns],
                                   [[0, 0, 0, 10, 11], [0, 11, 11, 21, 22],
                                    [1, 0, 0, 22, 23], [1, 23, 23, 33, 34]])
        self.assertEqual(batched["event_batch_size"].tolist(), [2, 2, 2, 2])
        self.assertNotEqual(individual.iloc[0]["scenario"], batched.iloc[0]["scenario"])

    def test_batch_size_one_preserves_every_default_policy(self):
        events = [event(4, 3, 1, 2, 1), event(2, 1, 3, 1, 0), event(9)]
        for policy in ("blocking", "synchronous", "async"):
            with self.subTest(policy=policy):
                kwargs = dict(workers=2, max_inflight=3, policy=policy)
                pd.testing.assert_frame_equal(replay(events, **kwargs),
                                              replay(events, event_batch_size=1, **kwargs))

    def test_synchronous_batch_nondivisible_tail_keeps_its_worker(self):
        timeline = replay([event(i, 10, 1, 0, 0) for i in range(5)], workers=2,
                          max_inflight=2, policy="synchronous", event_batch_size=2)
        self.assertEqual(timeline["worker_id"].tolist(), [0, 0, 1, 1, 0])
        self.assertEqual(timeline["admission_time_offset_s"].tolist(), [0, 11, 0, 23, 22])
        self.assertEqual(timeline["gpu_start_time_offset_s"].tolist(), [10, 21, 22, 33, 34])
        self.assertEqual(timeline["latency_s"].max(), 35)

    def test_synchronous_batches_with_one_worker_or_one_active_event(self):
        events = [event(i, 2, 1, 1, 1) for i in range(5)]
        for workers, cap in [(1, 8), (4, 1)]:
            with self.subTest(workers=workers, cap=cap):
                timeline = replay(events, workers=workers, max_inflight=cap,
                                  policy="synchronous", event_batch_size=2)
                self.assertEqual(timeline["admission_time_offset_s"].tolist(),
                                 [0, 5, 10, 15, 20])
                self.assertEqual(timeline["latency_s"].tolist(), [5, 10, 15, 20, 25])
                self.assertEqual(timeline["worker_id"].tolist(), [0, 0, 0, 0, 0])

    def test_synchronous_batches_drain_zero_phases_and_empty_input(self):
        for cap in (1, 2, 4):
            with self.subTest(cap=cap):
                timeline = replay([event(i, 0, 0, 0, 0) for i in range(7)], workers=4,
                                  max_inflight=cap, policy="synchronous", event_batch_size=3)
                self.assertEqual(timeline["latency_s"].tolist(), [0] * 7)
        empty = replay([], policy="synchronous", event_batch_size=3)
        self.assertTrue(empty.empty)
        self.assertIn("event_batch_size", empty)

    def test_synchronous_batch_random_schedules_obey_capacity_and_reservations(self):
        rng = random.Random(834)
        for workers in (1, 2, 4):
            for cap in (1, 2, 4):
                for batch in (2, 3, 7):
                    with self.subTest(workers=workers, cap=cap, batch=batch):
                        events = [event(i, *(rng.randrange(4) for _ in range(4)))
                                  for i in range(11)]
                        timeline = replay(events, workers=workers, max_inflight=cap,
                                          policy="synchronous", event_batch_size=batch)
                        repeated = replay(events, workers=workers, max_inflight=cap,
                                          policy="synchronous", event_batch_size=batch)
                        pd.testing.assert_frame_equal(timeline, repeated)
                        spans = list(zip(timeline["admission_time_offset_s"],
                                         timeline["cpu_post_end_time_offset_s"]))
                        self.assertLessEqual(self.peak(spans), min(workers, cap))
                        for _, group in timeline.groupby("worker_id"):
                            spans = list(zip(group["cpu_pre_start_time_offset_s"],
                                             group["cpu_post_end_time_offset_s"]))
                            self.assertLessEqual(self.peak(spans), 1)
                        for start in range(0, len(events), batch):
                            self.assertEqual(timeline.iloc[start:start + batch]["worker_id"].nunique(), 1)
                        self.assertTrue(np.all(timeline["gpu_start_time_offset_s"].to_numpy()[1:] >=
                                               timeline["gpu_release_time_offset_s"].to_numpy()[:-1]))
                        for _, row in timeline.iterrows():
                            boundaries = row[["admission_time_offset_s", "cpu_pre_start_time_offset_s",
                                              "cpu_pre_end_time_offset_s", "gpu_start_time_offset_s",
                                              "gpu_end_time_offset_s", "cpu_post_start_time_offset_s",
                                              "gpu_release_time_offset_s", "cpu_post_end_time_offset_s"]]
                            self.assertTrue(np.all(np.diff(boundaries.to_numpy(dtype=float)) >= 0))
                            for phase in ("pre", "post_locked", "post_unlocked"):
                                self.assertEqual(row[f"cpu_{phase}_end_time_offset_s"] -
                                                 row[f"cpu_{phase}_start_time_offset_s"],
                                                 row[f"cpu_{phase}_runtime_s"])

    def test_invalid_event_batches_and_other_policies_are_rejected(self):
        for batch in (0, -1, True, 1.5, None, "2"):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                replay([], policy="synchronous", event_batch_size=batch)
        for policy in ("blocking", "async"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                replay([], policy=policy, event_batch_size=2)

    def test_async_submits_fifo_without_future_duration_ordering(self):
        timeline = replay([event(30, 5, 1, 1, 0), event(10, 1, 1, 1, 0)],
                          workers=2, max_inflight=2, policy="async")
        self.assertEqual(timeline["gpu_start_time_offset_s"].tolist(), [5, 1])
        self.assertEqual(timeline["gpu_release_time_offset_s"].tolist(), [7, 3])

    def test_async_releases_waiting_worker_and_prioritizes_ready_post(self):
        timeline = replay([event(i) for i in range(3)], workers=1,
                          max_inflight=3, policy="async")
        columns = ["cpu_pre_start_time_offset_s", "cpu_pre_end_time_offset_s",
                   "gpu_start_time_offset_s", "gpu_end_time_offset_s",
                   "cpu_post_start_time_offset_s", "gpu_release_time_offset_s",
                   "cpu_post_end_time_offset_s"]
        np.testing.assert_allclose(timeline[columns],
                                   [[0, 1, 1, 3, 3, 4, 5],
                                    [1, 2, 4, 6, 6, 7, 8],
                                    [2, 3, 7, 9, 9, 10, 11]])
        # With a fourth pre task waiting at t=3, ready post takes the worker.
        more = replay([event(i) for i in range(4)], workers=1,
                      max_inflight=4, policy="async")
        self.assertEqual(more.iloc[3]["cpu_pre_start_time_offset_s"], 5)

    def test_async_gate_includes_wait_for_cpu_and_locked_post(self):
        events = [event(0, 1, 1, 2, 0), event(1, 5, 1, 1, 0)]
        timeline = replay(events, workers=1, max_inflight=2, policy="async")
        self.assertEqual(timeline["gpu_end_time_offset_s"].tolist(), [2, 9])
        self.assertEqual(timeline["cpu_post_start_time_offset_s"].tolist(), [6, 9])
        self.assertEqual(timeline["gpu_release_time_offset_s"].tolist(), [8, 10])
        self.assertEqual(timeline["gpu_start_time_offset_s"].tolist(), [1, 8])
        self.assertEqual(timeline["gpu_gate_runtime_s"].tolist(), [7, 2])
        self.assertEqual(timeline["gpu_runtime_s"].tolist(), [1, 1])

    def test_max_inflight_preserves_admission_delay_and_latency(self):
        timeline = replay([event(i) for i in range(3)], workers=2,
                          max_inflight=1, policy="async")
        self.assertEqual(timeline["admission_time_offset_s"].tolist(), [0, 5, 10])
        self.assertEqual(timeline["latency_s"].tolist(), [5, 10, 15])
        self.assertEqual(timeline["service_latency_s"].tolist(), [5, 5, 5])

    def test_all_policies_respect_dependencies_wip_cpu_and_gpu_capacity(self):
        # Independent interval sweeps catch worker overbooking and early gate release.
        events = [event(i, pre, gpu, locked, unlocked)
                  for i, (pre, gpu, locked, unlocked) in enumerate(
                      [(4, 2, 1, 5), (1, 3, 2, 1), (2, 1, 4, 2),
                       (3, 4, 0, 1), (1, 1, 2, 0), (2, 2, 1, 3)])]
        for policy in ("blocking", "synchronous", "async"):
            with self.subTest(policy=policy):
                timeline = replay(events, workers=2, max_inflight=3, policy=policy)
                self.assertEqual(timeline["event_id"].tolist(), list(range(6)))
                cpu_intervals = []
                worker_intervals = {}
                wip_intervals = []
                for _, row in timeline.iterrows():
                    boundaries = row[["admission_time_offset_s", "cpu_pre_start_time_offset_s",
                                      "cpu_pre_end_time_offset_s", "gpu_start_time_offset_s",
                                      "gpu_end_time_offset_s", "cpu_post_start_time_offset_s",
                                      "gpu_release_time_offset_s", "cpu_post_end_time_offset_s"]]
                    self.assertTrue(np.all(np.diff(boundaries.to_numpy(dtype=float)) >= 0))
                    for phase in ("pre", "post"):
                        begin = row[f"cpu_{phase}_start_time_offset_s"]
                        end = row[f"cpu_{phase}_end_time_offset_s"]
                        cpu_intervals.append((begin, end))
                        worker = row[f"cpu_{phase}_worker_id"]
                        worker_intervals.setdefault(worker, []).append((begin, end))
                    wip_intervals.append((row["admission_time_offset_s"],
                                          row["cpu_post_end_time_offset_s"]))
                self.assertLessEqual(self.peak(cpu_intervals), 2)
                for intervals in worker_intervals.values():
                    self.assertLessEqual(self.peak(intervals), 1)
                limit = 1 if policy == "blocking" else (2 if policy == "synchronous" else 3)
                self.assertLessEqual(self.peak(wip_intervals), limit)
                gate = timeline.sort_values("gpu_start_time_offset_s")
                self.assertTrue(np.all(gate["gpu_start_time_offset_s"].to_numpy()[1:] >=
                                       gate["gpu_release_time_offset_s"].to_numpy()[:-1]))
                if policy == "synchronous":
                    for _, group in timeline.groupby("worker_id"):
                        occupied = list(zip(group["cpu_pre_start_time_offset_s"],
                                            group["cpu_post_end_time_offset_s"]))
                        self.assertLessEqual(self.peak(occupied), 1)

    @staticmethod
    def peak(intervals):
        changes = sorted([(start, 1) for start, end in intervals if end > start] +
                         [(end, -1) for start, end in intervals if end > start])
        count = peak = 0
        for _, delta in changes:
            count += delta
            peak = max(peak, count)
        return peak

    def test_zero_durations_and_zero_photons_terminate(self):
        events = [event(0, 0, 0, 0, 0, 0), event(1, 0, 0, 2, 0, 0),
                  event(2, 0, 0, 0, 0, 0)]
        for policy in ("blocking", "synchronous", "async"):
            with self.subTest(policy=policy):
                timeline = replay(events, workers=1, max_inflight=3, policy=policy)
                self.assertEqual(timeline["latency_s"].tolist(), [0, 2, 2])
                self.assertEqual(timeline["num_photons"].tolist(), [0, 0, 0])
        many = replay([event(i, 0, 0, 0, 0, 0) for i in range(1000)],
                      workers=8, max_inflight=32, policy="async")
        self.assertEqual(many["latency_s"].sum(), 0)

    def test_empty_has_shared_timeline_schema(self):
        timeline = replay([])
        self.assertTrue(timeline.empty)
        for column in ("scenario", "event_id", "cpu_start_time_offset_s",
                       "cpu_end_time_offset_s", "gpu_start_time_offset_s",
                       "gpu_end_time_offset_s", "gpu_release_time_offset_s", "latency_s"):
            self.assertIn(column, timeline)

    def test_deterministic_without_mutating_inputs(self):
        events = [event(3), event(1), event(8)]
        original = deepcopy(events)
        first = replay(events, workers=2, max_inflight=3, policy="async")
        second = replay(events, workers=2, max_inflight=3, policy="async")
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(events, original)

    def test_replay_is_compatible_with_csv_summary_and_plot(self):
        timeline = replay([event(0), event(1)], workers=1,
                          max_inflight=2, policy="async")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.csv"
            timeline.to_csv(path, index=False)
            loaded = read_one_csv(path)
        summary = timeline_summary(loaded).iloc[0]
        self.assertEqual(summary["makespan_s"], 8)
        self.assertEqual(summary["cpu_busy_s"], 6)
        self.assertEqual(summary["gpu_busy_s"], 4)
        figure, _, plotted = plot_run_timeline(loaded, loaded.iloc[0]["scenario"])
        self.assertEqual(len(plotted), 2)
        plt.close(figure)

    def test_unrepresentable_durations_fail_as_validation_errors(self):
        with self.assertRaises(ValueError):
            replay([event(0, pre=10 ** 400)])
        with self.assertRaises(ValueError):
            replay([event(0, pre=1e308, gpu=1e308)])

    def test_invalid_inputs_raise_clear_value_errors(self):
        bad_events = [None, [None], [{}], [event(1), event(1)],
                      [event(-1)], [event(True)], [event(1, photons=-1)],
                      [event(1, photons=1.5)]]
        for field in ("cpu_pre_runtime_s", "gpu_runtime_s",
                      "cpu_post_locked_runtime_s", "cpu_post_unlocked_runtime_s"):
            for bad in (-1, float("nan"), float("inf"), "1", True, None):
                item = event(0)
                item[field] = bad
                bad_events.append([item])
        for items in bad_events:
            with self.subTest(events=items):
                with self.assertRaises(ValueError):
                    replay(items)
        for kwargs in ({"workers": 0}, {"workers": True}, {"workers": 1.5},
                       {"max_inflight": 0}, {"max_inflight": -1},
                       {"max_inflight": True}, {"max_inflight": 1.5},
                       {"policy": "event_barrier"}, {"policy": None}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    replay([], **kwargs)


if __name__ == "__main__":
    unittest.main()
