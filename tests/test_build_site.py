"""Check that every notebook and its evidence reaches the Pages artifact."""

import json
from html.parser import HTMLParser
from pathlib import Path
import zipfile

from scripts import build_site


class GalleryParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cards = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.current = {"text": "", "links": []}
            self.cards.append(self.current)
        if tag == "a" and self.current is not None:
            self.current["links"].append(dict(attrs).get("href"))

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"] += data

    def handle_endtag(self, tag):
        if tag == "article":
            self.current = None


def test_site_includes_all_notebooks_and_evidence(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for name in ("data", "notebooks", "teerex"):
        (tmp_path / name).symlink_to(root / name, target_is_directory=True)
    monkeypatch.setattr(build_site, "ROOT", tmp_path)

    build_site.main()

    site = tmp_path / "_site"
    index = (site / "index.html").read_text()
    assert (site / ".nojekyll").is_file()
    pages = {
        "simload": "simload",
        "timelines_single_worker": "timelines_single_worker",
        "timelines_multi_worker": "timelines_multi_worker",
        "timelines_multi_gpu": "timelines_multi_gpu",
    }
    gallery = GalleryParser()
    gallery.feed(index)
    assert len(gallery.cards) == 3
    for card, name in zip(gallery.cards, list(pages)[1:], strict=True):
        assert f"{name}.py" in card["text"]
        assert card["links"] == [f"./{name}/"]
    assert "simload" not in index
    assert set(pages) == {path.stem for path in (root / "notebooks").glob("*.py")}
    for notebook, route in pages.items():
        assert (f'href="./{route}/"' in index) == (notebook != "simload")
        assert (site / route / "index.html").is_file()

    evidence = {
        "timelines_single_worker": ("single_worker", ("events.csv.gz",)),
        "timelines_multi_worker": ("multi_worker", (
            "events-raw.csv.gz", "run-summary.csv", "summary.csv", "sensitivity.csv",
        )),
        "timelines_multi_gpu": ("multi_gpu", (
            "manifest.json", "events.csv.gz", "run-summary.csv",
            "condition-summary.csv", "paired-summary.csv",
        )),
    }
    for route, (bundle, filenames) in evidence.items():
        for name in filenames:
            assert (site / route / "public" / name).read_bytes() == (
                root / "data" / bundle / name
            ).read_bytes()

    for old, current in {
        "ratio_timelines": "timelines_single_worker",
        "mt_timelines": "timelines_multi_worker",
    }.items():
        redirect = (site / old / "index.html").read_text()
        assert f'content="0; url=../{current}/"' in redirect
        assert f'href="../{current}/"' in redirect
        assert f'href="./{old}/"' not in index

    bundle = root / "data" / "single_worker"
    manifest = json.loads((bundle / "manifest.json").read_text())
    with zipfile.ZipFile(site / "simload" / "public" / "study.zip") as archive:
        assert set(archive.namelist()) == {"manifest.json", *manifest["files"]}
        for name in archive.namelist():
            assert archive.read(name) == (bundle / name).read_bytes()
