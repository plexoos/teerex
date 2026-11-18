"""Keep electron timeline views independent of the complete Simload replay data."""

from pathlib import Path


def test_full_replay_data_keeps_seven_electron_workload_views(monkeypatch):
    from notebooks.timelines_single_worker import app

    source = Path(__file__).resolve().parents[1] / "data" / "single_worker" / "events.csv.gz"
    monkeypatch.setenv("TEEREX_TIMELINE_CSV", str(source))
    _, definitions = app.run()

    events = definitions["production_events"]
    assert events.groupby("particle").size().to_dict() == {"e-": 14_000, "mu-": 8_000}

    frames = definitions["configuration_frames"]
    assert len(frames) == 7
    for (_, particle), frame in frames.items():
        assert particle == "e-"
        assert frame.particle.eq("e-").all()
        assert len(frame) == 2_000
        assert frame.groupby("replication").size().to_dict() == {
            0: 500, 1: 500, 2: 500, 3: 500,
        }

    summary = definitions["configuration_summary"]
    assert len(summary) == 7
    assert summary["Particle"].eq("e−").all()
    assert summary["Events"].eq(2_000).all()
    assert len(definitions["timeline_figures"]) == 7
    assert len(definitions["distribution_figures"]) == 7
