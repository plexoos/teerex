from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from teerex.analysis import (
    plot_run_timeline,
    read_many_csv,
    read_one_csv,
    timeline_summary,
)


REPO_DIR = Path(__file__).resolve().parents[1]


class TimelineAnalysisTest(unittest.TestCase):
    def test_legacy_simload_offsets_are_derived(self) -> None:
        timeline = read_one_csv(REPO_DIR / "simload_runs" / "mode_blocking.csv")

        self.assertEqual(set(timeline["scenario"]), {"mode_blocking"})
        self.assertEqual(set(timeline["dispatch_mode"]), {"blocking"})
        self.assertIn("cpu_start_time_offset_s", timeline)
        self.assertIn("gpu_end_time_offset_s", timeline)
        self.assertAlmostEqual(timeline["cpu_start_time_offset_s"].min(), 0.0)

    def test_measured_scenario_and_offsets_are_preserved(self) -> None:
        measured = pd.DataFrame(
            {
                "scenario": ["simg4ox_blocking"],
                "dispatch_mode": ["blocking"],
                "event_id": [0],
                "cpu_start_time_offset_s": [0.01],
                "cpu_end_time_offset_s": [0.04],
                "gpu_start_time_offset_s": [0.02],
                "gpu_end_time_offset_s": [0.03],
                "cpu_runtime_s": [0.02],
                "gpu_runtime_s": [0.01],
                "num_photons": [978],
            }
        )

        with TemporaryDirectory() as directory:
            measured_path = Path(directory) / "events.csv"
            measured.to_csv(measured_path, index=False)
            combined = read_many_csv(
                [REPO_DIR / "simload_runs" / "mode_async.csv", measured_path]
            )

        self.assertIn("simg4ox_blocking", set(combined["scenario"]))
        summary = timeline_summary(combined)
        measured_summary = summary.loc[summary["scenario"].eq("simg4ox_blocking")].iloc[0]
        self.assertEqual(measured_summary["num_photons"], 978)

        figure, _, timeline = plot_run_timeline(
            combined,
            ["mode_async", "simg4ox_blocking"],
        )
        self.assertEqual(len(timeline), 21)
        self.assertEqual(len(figure.axes), 2)


if __name__ == "__main__":
    unittest.main()
