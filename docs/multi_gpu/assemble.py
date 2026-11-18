"""Rebuild the notebook data from the charged-native-multi-gpu-v1 archive."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(archive, output):
    plan = json.loads((archive / "plan.json").read_text())
    verification = json.loads((archive / "verification.json").read_text())
    if not verification["passed"]:
        raise ValueError("Source campaign verification failed")

    source_hashes = {}
    num_events = 0
    raw_header = None
    with (output / "events.csv.gz").open("wb") as compressed:
        with gzip.GzipFile(fileobj=compressed, filename="", mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                for run in plan["runs"]:
                    run_id = run["run_id"]
                    directory = archive / "runs" / run_id
                    meta = json.loads((directory / "run.json").read_text())
                    events = directory / "events.csv"
                    digest = sha256(events)
                    if run_id in source_hashes or meta["run_id"] != run_id or meta["status"] != "complete":
                        raise ValueError(f"Invalid or repeated run {run_id}")
                    if digest != meta["file_hashes"]["events.csv"]:
                        raise ValueError(f"{run_id}: timing CSV hash differs from the measured run")
                    source_hashes[run_id] = digest
                    with events.open(encoding="utf-8", newline="") as source:
                        reader = csv.reader(source)
                        header = next(reader)
                        if raw_header is None:
                            raw_header = header
                            writer.writerow(["run_id", *header])
                        if header != raw_header or "run_id" in header:
                            raise ValueError(f"{run_id}: inconsistent timing columns")
                        count = 0
                        for row in reader:
                            writer.writerow([run_id, *row])
                            count += 1
                        if count != plan["raw_events"]:
                            raise ValueError(f"{run_id}: expected {plan['raw_events']} raw events, found {count}")
                        num_events += count

    copies = ("run-summary.csv", "condition-summary.csv", "paired-summary.csv", "profile/kernel-overlap.json")
    for name in copies:
        target = output / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(archive / name, target)
    manifest = {
        "source_archive": "charged-native-multi-gpu-v1",
        "simphony_commit": plan["commit"],
        "teerex_commit_at_collection": plan["teerex_commit"],
        "runs": len(source_hashes),
        "raw_events": num_events,
        "warmup_completions_per_run": plan["warmup_events"],
        "geant4_threads": plan["threads"],
        "gpus": [
            dict(gpu, index=int(gpu["index"]))
            for gpu in csv.DictReader(plan["gpus"].splitlines(), skipinitialspace=True)
        ],
        "source_plan_sha256": sha256(archive / "plan.json"),
        "source_verification_sha256": sha256(archive / "verification.json"),
        "source_event_csv_sha256_by_run": source_hashes,
        "sha256": {name: sha256(output / name) for name in ("events.csv.gz", *copies)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="study directory or extracted timings.zip")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "multi_gpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as scratch:
        staging = Path(scratch)
        assemble(args.archive, staging)
        for path in staging.rglob("*"):
            if path.is_file():
                target = args.output / path.relative_to(staging)
                target.parent.mkdir(exist_ok=True)
                path.replace(target)
