"""Resolve portable notebook inputs against the checkout or hosted page."""

import asyncio
import gzip
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest


def test_local_paths_use_checkout_root_and_display_relative_defaults(tmp_path, monkeypatch):
    from teerex.notebook_data import relative_data_path, resolve_data_file

    root = tmp_path / "checkout"
    source = root / "docs" / "events.csv"
    source.parent.mkdir(parents=True)
    source.write_text("value\n7\n")
    monkeypatch.chdir(tmp_path)

    assert relative_data_path(source, root) == "docs/events.csv"
    assert relative_data_path("docs/events.csv", root) == "docs/events.csv"
    for value in ("docs/events.csv", str(source)):
        resolved = asyncio.run(resolve_data_file(value, root=root))
        assert pd.read_csv(resolved).value.tolist() == [7]


@pytest.mark.parametrize("path", ["", "missing.csv"])
def test_local_missing_inputs_fail_clearly(tmp_path, path):
    from teerex.notebook_data import resolve_data_file

    with pytest.raises((ValueError, FileNotFoundError)):
        asyncio.run(resolve_data_file(path, root=tmp_path))


def test_browser_paths_fetch_selected_sources_and_preserve_gzip(tmp_path, monkeypatch):
    from teerex.notebook_data import resolve_data_file

    base = "https://example.github.io/teerex/timelines_multi_gpu/"
    payloads = {
        base + "public/events.csv.gz": gzip.compress(b"value\n11\n"),
        base + "alternative/events.csv.gz": gzip.compress(b"value\n22\n"),
    }

    async def pyfetch(url):
        async def body():
            return payloads[url]

        return SimpleNamespace(ok=url in payloads, status=200 if url in payloads else 404, bytes=body)

    http = ModuleType("pyodide.http")
    http.pyfetch = pyfetch
    monkeypatch.setitem(sys.modules, "pyodide", ModuleType("pyodide"))
    monkeypatch.setitem(sys.modules, "pyodide.http", http)

    first = asyncio.run(resolve_data_file("public/events.csv.gz", root=tmp_path, browser_base=base))
    second = asyncio.run(resolve_data_file("alternative/events.csv.gz", root=tmp_path, browser_base=base))
    assert first != second
    assert pd.read_csv(first).value.tolist() == [11]
    assert pd.read_csv(second).value.tolist() == [22]

    payloads[base + "public/events.csv.gz"] = gzip.compress(b"value\n33\n")
    updated = asyncio.run(resolve_data_file("public/events.csv.gz", root=tmp_path, browser_base=base))
    assert pd.read_csv(updated).value.tolist() == [33]

    del payloads[base + "public/events.csv.gz"]
    with pytest.raises(RuntimeError, match=r"public/events\.csv\.gz.*404"):
        asyncio.run(resolve_data_file("public/events.csv.gz", root=tmp_path, browser_base=base))
