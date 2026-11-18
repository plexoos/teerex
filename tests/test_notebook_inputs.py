"""Verify relative defaults and replacement inputs drive each timeline loader."""

import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import pandas as pd
import pytest

from teerex.analysis import read_one_csv
from teerex.notebook_data import resolve_data_file


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("timelines_single_worker", "single_worker", {
        "csv_path": ("events.csv.gz", "production_events"),
    }),
    ("timelines_multi_worker", "multi_worker", {
        "csv_path": ("events-raw.csv.gz", "events"),
        "run_summary_path": ("run-summary.csv", "runs"),
        "summary_path": ("summary.csv", "summary"),
        "sensitivity_path": ("sensitivity.csv", "sensitivity"),
    }),
    ("timelines_multi_gpu", "multi_gpu", {
        "csv_path": ("events.csv.gz", "events"),
        "run_summary_path": ("run-summary.csv", "runs"),
        "condition_summary_path": ("condition-summary.csv", "conditions"),
        "paired_summary_path": ("paired-summary.csv", "pairs"),
        "manifest_path": ("manifest.json", "manifest"),
    }),
]


@pytest.mark.parametrize("name,bundle,inputs", CASES, ids=[case[0] for case in CASES])
def test_relative_controls_and_replacement_files(name, bundle, inputs, tmp_path, monkeypatch):
    monkeypatch.delenv("TEEREX_TIMELINE_CSV", raising=False)
    monkeypatch.delenv("TEEREX_MT_DATA_DIR", raising=False)
    notebook = importlib.import_module(f"notebooks.{name}")
    _, defaults = notebook.app.run()
    for control, (filename, _) in inputs.items():
        assert defaults[control].value == f"data/{bundle}/{filename}"

    refs = dict(browser_base=None, data_root=tmp_path, resolve_data_file=resolve_data_file, mo=mo, pd=pd)
    if name == "timelines_multi_gpu":
        refs["json"] = json
    else:
        refs["read_one_csv"] = read_one_csv
    if name == "timelines_single_worker":
        refs.pop("pd")

    for version in (1, 2):
        directory = tmp_path / f"replacement{version}"
        directory.mkdir()
        for index, (control, (filename, _)) in enumerate(inputs.items()):
            original = ROOT / "data" / bundle / filename
            target = directory / filename
            if filename.endswith(".json"):
                data = json.loads(original.read_text())
                data["replacement_marker"] = version * 10 + index
                target.write_text(json.dumps(data))
            else:
                data = pd.read_csv(original).head(2)
                data["replacement_marker"] = version * 10 + index
                data.to_csv(target, index=False)
            refs[control] = SimpleNamespace(value=f"replacement{version}/{filename}")
        _, loaded = asyncio.run(notebook.load_data.run(**refs))
        for index, (_, definition) in enumerate(inputs.values()):
            data = loaded[definition]
            if isinstance(data, dict):
                assert data["replacement_marker"] == version * 10 + index
            else:
                assert data.replacement_marker.eq(version * 10 + index).all()
