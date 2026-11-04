from pathlib import Path

import pandas as pd
import pytest

from teerex.analysis import (
    plot_run_timeline,
    read_many_csv,
    read_one_csv,
    timeline_summary,
)

REPO_DIR = Path(__file__).resolve().parents[1]


def test_legacy_simload_offsets_are_derived() -> None:
    timeline = read_one_csv(REPO_DIR / "simload_runs" / "mode_blocking.csv")

    assert set(timeline["scenario"]) == {"mode_blocking"}
    assert set(timeline["dispatch_mode"]) == {"blocking"}
    assert "cpu_start_time_offset_s" in timeline
    assert "gpu_end_time_offset_s" in timeline
    assert timeline["cpu_start_time_offset_s"].min() == pytest.approx(
        0.0, rel=0, abs=5e-8
    )


def test_measured_scenario_and_offsets_are_preserved(tmp_path) -> None:
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

    measured_path = tmp_path / "events.csv"
    measured.to_csv(measured_path, index=False)
    combined = read_many_csv(
        [REPO_DIR / "simload_runs" / "mode_async.csv", measured_path]
    )

    assert "simg4ox_blocking" in set(combined["scenario"])
    summary = timeline_summary(combined)
    measured_summary = summary.loc[summary["scenario"].eq("simg4ox_blocking")].iloc[0]
    assert measured_summary["num_photons"] == 978

    figure, _, timeline = plot_run_timeline(
        combined,
        ["mode_async", "simg4ox_blocking"],
    )
    assert len(timeline) == 21
    assert len(figure.axes) == 2
