"""Portable file paths for local and browser-executed notebooks."""

import hashlib
import os
from pathlib import Path
from urllib.parse import urljoin, urlsplit


def relative_data_path(path: str | Path, root: Path) -> str:
    """Display local defaults relative to the checkout, including env overrides."""
    source = Path(path).expanduser()
    return Path(os.path.relpath(source, root)).as_posix() if source.is_absolute() else source.as_posix()


async def resolve_data_file(path: str, *, root: Path, browser_base: str | None = None) -> Path:
    """Read local paths from root, or fetch paths relative to the hosted notebook."""
    value = path.strip()
    if not value:
        raise ValueError("Enter a data file path.")
    if browser_base is None:
        source = Path(value).expanduser()
        if not source.is_absolute():
            source = root / source
        if not source.is_file():
            raise FileNotFoundError(f"Data file not found: {value}")
        return source

    from pyodide.http import pyfetch

    url = urljoin(browser_base.rstrip("/") + "/", value)
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Use a relative web path or an HTTP(S) URL.")
    response = await pyfetch(url)
    if not response.ok:
        raise RuntimeError(f"Could not load {value}: HTTP {response.status}")
    # Distinct source URLs can have the same filename. Preserve suffixes so
    # pandas still recognizes compressed CSVs, and refresh on every evaluation.
    directory = root / hashlib.sha256(url.encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / (Path(parsed.path).name or "data")
    source.write_bytes(await response.bytes())
    return source
