"""Exercise timeline file inputs through the real browser data loaders."""

import asyncio
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import marimo as mo
import pandas as pd
import pytest

from teerex.analysis import read_one_csv
from teerex.notebook_data import resolve_data_file


CASES = [
    ('timelines_single_worker', 'single_worker', 'production_events', 22_000, {
        'csv_path': 'events.csv.gz',
    }),
    ('timelines_multi_worker', 'multi_worker', 'events', 57_792, {
        'csv_path': 'events-raw.csv.gz', 'run_summary_path': 'run-summary.csv',
        'summary_path': 'summary.csv', 'sensitivity_path': 'sensitivity.csv',
    }),
    ('timelines_multi_gpu', 'multi_gpu', 'events', 4_032, {
        'csv_path': 'events.csv.gz', 'run_summary_path': 'run-summary.csv',
        'condition_summary_path': 'condition-summary.csv',
        'paired_summary_path': 'paired-summary.csv', 'manifest_path': 'manifest.json',
    }),
]


@pytest.mark.parametrize('name,bundle,event_definition,count,inputs', CASES,
                         ids=[case[0] for case in CASES])
def test_browser_loads_files_selected_in_inputs(name, bundle, event_definition, count,
                                               inputs, tmp_path, monkeypatch):
    notebook = importlib.import_module(f'notebooks.{name}')
    evidence = Path(__file__).resolve().parents[1] / 'data' / bundle
    base = f'https://example.github.io/teerex/{name}/'
    payloads = {base + 'alternative/' + filename: (evidence / filename).read_bytes()
                for filename in inputs.values()}

    async def pyfetch(url):
        assert url in payloads

        async def body():
            return payloads[url]

        return SimpleNamespace(ok=True, status=200, bytes=body)

    http = ModuleType('pyodide.http')
    http.pyfetch = pyfetch
    monkeypatch.setitem(sys.modules, 'pyodide', ModuleType('pyodide'))
    monkeypatch.setitem(sys.modules, 'pyodide.http', http)
    refs = dict(browser_base=base, data_root=tmp_path, resolve_data_file=resolve_data_file,
                mo=mo, pd=pd)
    refs.update({control: SimpleNamespace(value='alternative/' + filename)
                 for control, filename in inputs.items()})
    if name == 'timelines_multi_gpu':
        refs['json'] = json
    else:
        refs['read_one_csv'] = read_one_csv
    if name == 'timelines_single_worker':
        refs.pop('pd')

    _, loaded = asyncio.run(notebook.load_data.run(**refs))
    assert len(loaded[event_definition]) == count
    cached = {path.name: path.read_bytes() for path in tmp_path.glob('*/*')}
    assert cached == {filename: (evidence / filename).read_bytes() for filename in inputs.values()}
