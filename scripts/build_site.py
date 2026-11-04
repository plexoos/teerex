"""Export the browser-executed marimo notebooks for GitHub Pages."""

import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


NOTEBOOKS = {
    "simload": "Production ratio study",
    "ratio_timelines": "Measured CPU/GPU timelines",
}


def main():
    site = ROOT / "_site"
    shutil.rmtree(site, ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="teerex-pages-") as temporary:
        evidence = ROOT / "docs" / "study"
        for name in NOTEBOOKS:
            source = Path(temporary) / name
            package = source / "teerex"
            package.mkdir(parents=True)
            (package / "__init__.py").touch()
            # Ship only the pure-Python modules used by the notebooks.
            for module in ("analysis.py", "replay.py"):
                shutil.copy2(ROOT / "teerex" / module, package / module)

            public = source / "public"
            public.mkdir()
            if name == "simload":
                with zipfile.ZipFile(public / "study.zip", "w", zipfile.ZIP_DEFLATED) as snapshot:
                    for path in sorted(evidence.iterdir()):
                        if path.suffix in {".csv", ".json", ".gz"}:
                            snapshot.write(path, path.name)
            else:
                shutil.copy2(evidence / "events.csv.gz", public)

            notebook = source / f"{name}.py"
            shutil.copy2(ROOT / "notebooks" / notebook.name, notebook)
            subprocess.run(
                [sys.executable, "-m", "marimo", "export", "html-wasm",
                 str(notebook), "-o", str(site / name), "--mode", "run"],
                check=True,
            )

    (site / ".nojekyll").touch()
    links = "\n".join(
        f'<li><a href="./{name}/">{title}</a></li>'
        for name, title in NOTEBOOKS.items()
    )
    (site / "index.html").write_text(
        f'<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>Teerex notebooks</title><h1>Teerex notebooks</h1><ul>{links}</ul>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
