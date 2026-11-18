"""Export the browser-executed marimo notebooks for GitHub Pages."""

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from html import escape
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]


NOTEBOOKS = {
    "simload": "Production ratio study",
    "timelines_single_worker": "Geant4 CPU/GPU timelines: single worker",
    "timelines_multi_worker": "Geant4 CPU/GPU timelines: multiple workers",
    "timelines_multi_gpu": "Geant4 CPU/GPU timelines: multiple GPUs",
}

# Keep old bookmarks working while canonical routes match notebook filenames.
LEGACY_SITE_PATHS = {
    "ratio_timelines": "timelines_single_worker",
    "mt_timelines": "timelines_multi_worker",
}


def main():
    site = ROOT / "_site"
    shutil.rmtree(site, ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="teerex-pages-") as temporary:
        for name in NOTEBOOKS:
            evidence_name = {
                "timelines_multi_worker": "multi_worker",
                "timelines_multi_gpu": "multi_gpu",
            }.get(name, "single_worker")
            evidence = ROOT / "data" / evidence_name
            source = Path(temporary) / name
            package = source / "teerex"
            package.mkdir(parents=True)
            (package / "__init__.py").touch()
            # Ship only the pure-Python modules used by the notebooks.
            for module in ("analysis.py", "replay.py", "notebook_data.py"):
                shutil.copy2(ROOT / "teerex" / module, package / module)

            public = source / "public"
            public.mkdir()
            if name == "simload":
                manifest = json.loads((evidence / "manifest.json").read_text())
                with zipfile.ZipFile(public / "study.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
                    for filename in sorted({"manifest.json", *manifest["files"]}):
                        snapshot.write(evidence / filename, filename)
            elif name == "timelines_single_worker":
                shutil.copy2(evidence / "events.csv.gz", public)
            elif name == "timelines_multi_gpu":
                for filename in (
                    "manifest.json", "events.csv.gz", "run-summary.csv",
                    "condition-summary.csv", "paired-summary.csv",
                ):
                    shutil.copy2(evidence / filename, public)
            else:
                for filename in ("events-raw.csv.gz", "run-summary.csv", "summary.csv", "sensitivity.csv"):
                    shutil.copy2(evidence / filename, public)

            notebook = source / f"{name}.py"
            shutil.copy2(ROOT / "notebooks" / notebook.name, notebook)
            subprocess.run(
                [sys.executable, "-m", "marimo", "export", "html-wasm",
                 str(notebook), "-o", str(site / name), "--mode", "run"],
                check=True,
            )

    (site / ".nojekyll").touch()
    cards = "\n".join(
        f'<article class="card" aria-labelledby="{escape(name)}-title">'
        f'<h2 class="card-header" id="{escape(name)}-title">{escape(title)}</h2>'
        f'<div class="card-body"><code>{escape(name)}.py</code>'
        f'<a class="card-link" href="./{escape(name)}/" '
        f'aria-label="Open notebook: {escape(title)}">Open notebook</a></div></article>'
        for name, title in NOTEBOOKS.items()
        if name != "simload"
    )
    template = Template(
        (Path(__file__).parent / "templates" / "gallery.html").read_text(encoding="utf-8")
    )
    (site / "index.html").write_text(
        template.substitute(cards=cards),
        encoding="utf-8",
    )
    for old, current in LEGACY_SITE_PATHS.items():
        directory = site / old
        directory.mkdir()
        target = f"../{current}/"
        (directory / "index.html").write_text(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="0; url={target}">'
            f'<link rel="canonical" href="{target}">'
            '<title>Notebook moved</title></head><body>'
            f'<p><a href="{target}">Open {escape(NOTEBOOKS[current])}</a></p>'
            f'<script>location.replace({json.dumps(target)} + location.search + location.hash);</script>'
            '</body></html>',
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
