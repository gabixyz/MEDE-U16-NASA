"""Load and validate the two FD002 source views, without fitting models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NASA_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
HASHES = {
    "train_FD002.txt": "dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f",
    "test_FD002.txt": "de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02",
    "RUL_FD002.txt": "c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b",
}
KEYS = ["split", "engine_id", "cycle"]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i:02d}" for i in range(1, 22)]
SENSOR_INFO = [
    ("T2", "degR"), ("T24", "degR"), ("T30", "degR"), ("T50", "degR"),
    ("P2", "psia"), ("P15", "psia"), ("P30", "psia"),
    ("Nf", "rpm"), ("Nc", "rpm"), ("epr", "dimensionless"),
    ("Ps30", "psia"), ("phi", "pps/psi"), ("NRf", "rpm"),
    ("NRc", "rpm"), ("BPR", "dimensionless"), ("farB", "dimensionless"),
    ("htBleed", "not specified (NASA table: --)"), ("Nf_dmd", "rpm"),
    ("PCNfR_dmd", "rpm"), ("W31", "lbm/s"), ("W32", "lbm/s"),
]


@dataclass(frozen=True)
class Observation:
    engine_id: int
    cycle: int
    settings: tuple[float, ...]
    sensors: tuple[float, ...]


def positive_integer(text: str, context: str) -> int:
    value = float(text)
    if not math.isfinite(value) or value < 1 or not value.is_integer():
        raise ValueError(f"{context}: expected a positive integer, got {text!r}")
    return int(value)


def read_observations(path: Path) -> list[Observation]:
    rows = []
    keys = set()
    last_cycle: dict[int, int] = {}
    with path.open(encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            context = f"{path.name}:{line_number}"
            fields = line.split()
            if len(fields) != 26:
                raise ValueError(f"{context}: expected 26 fields, found {len(fields)}")
            engine = positive_integer(fields[0], context)
            cycle = positive_integer(fields[1], context)
            values = tuple(float(value) for value in fields[2:])
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{context}: non-finite or missing measurement")
            if (engine, cycle) in keys:
                raise ValueError(f"{context}: duplicate engine-cycle key")
            expected_cycle = last_cycle.get(engine, 0) + 1
            if cycle != expected_cycle:
                raise ValueError(
                    f"{context}: expected cycle {expected_cycle} for engine {engine}, got {cycle}"
                )
            keys.add((engine, cycle))
            last_cycle[engine] = cycle
            rows.append(Observation(engine, cycle, values[:3], values[3:]))
    if not rows:
        raise ValueError(f"{path.name}: empty dataset")
    if sorted(last_cycle) != list(range(1, len(last_cycle) + 1)):
        raise ValueError(f"{path.name}: engine IDs must be contiguous from 1")
    return rows


def read_rul(path: Path, engine_count: int) -> list[int]:
    values = []
    with path.open(encoding="ascii") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if len(fields) != 1:
                raise ValueError(f"{path.name}:{line_number}: expected one RUL value")
            values.append(positive_integer(fields[0], f"{path.name}:{line_number}"))
    if len(values) != engine_count:
        raise ValueError(f"{path.name}: {len(values)} labels for {engine_count} engines")
    return values


def check_hashes(raw_dir: Path) -> dict[str, str]:
    actual = {}
    for name, expected in HASHES.items():
        digest = hashlib.sha256((raw_dir / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"{name}: SHA-256 differs from the CP1 frozen input")
        actual[name] = digest
    return actual


def download(raw_dir: Path) -> None:
    with urllib.request.urlopen(NASA_URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        selected = {}
        for name, expected in HASHES.items():
            matches = [entry for entry in archive.namelist() if Path(entry).name == name]
            if len(matches) != 1:
                raise ValueError(f"Archive must contain exactly one {name}")
            contents = archive.read(matches[0])
            if hashlib.sha256(contents).hexdigest() != expected:
                raise ValueError(f"Downloaded {name} does not match the frozen checksum")
            selected[name] = contents
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, contents in selected.items():
        (raw_dir / name).write_bytes(contents)


def write_csv(path: Path, headers: list[str], rows) -> None:
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def summarize(rows: list[Observation]) -> dict:
    lengths = Counter(row.engine_id for row in rows)
    settings = list(zip(*(row.settings for row in rows)))
    sensors = list(zip(*(row.sensors for row in rows)))
    return {
        "rows": len(rows),
        "engines": len(lengths),
        "trajectory_cycles_min": min(lengths.values()),
        "trajectory_cycles_max": max(lengths.values()),
        "missing_values": 0,
        "duplicate_keys": 0,
        "internal_cycle_gaps": 0,
        "settings_ranges": {
            name: [min(values), max(values)] for name, values in zip(SETTINGS, settings)
        },
        "sensor_ranges": {
            name: [min(values), max(values)] for name, values in zip(SENSORS, sensors)
        },
        "engine_lengths": {str(key): value for key, value in sorted(lengths.items())},
    }


def run(raw_dir: Path, out_dir: Path) -> dict:
    hashes = check_hashes(raw_dir)
    splits = {
        "train": read_observations(raw_dir / "train_FD002.txt"),
        "test": read_observations(raw_dir / "test_FD002.txt"),
    }
    summaries = {split: summarize(rows) for split, rows in splits.items()}
    expected = {"train": (53759, 260), "test": (33991, 259)}
    for split, (row_count, engine_count) in expected.items():
        if (summaries[split]["rows"], summaries[split]["engines"]) != (row_count, engine_count):
            raise ValueError(f"{split}: unexpected FD002 row or engine count")
    rul = read_rul(raw_dir / "RUL_FD002.txt", summaries["test"]["engines"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_csv(
            out_dir / f"{split}_sensors.csv", KEYS + SENSORS,
            ([split, row.engine_id, row.cycle, *row.sensors] for row in rows),
        )
        write_csv(
            out_dir / f"{split}_operating.csv", KEYS + SETTINGS,
            ([split, row.engine_id, row.cycle, *row.settings] for row in rows),
        )
    write_csv(
        out_dir / "evaluation_only_test_rul.csv", ["split", "engine_id", "rul_at_last_cycle"],
        (["test", engine, value] for engine, value in enumerate(rul, 1)),
    )
    report = {
        "dataset": "NASA C-MAPSS FD002 (legacy archive)",
        "source_url": NASA_URL,
        "checkpoint": "CP1 / 2026-10-19",
        "sha256": hashes,
        "source_views": {"S1": SENSORS, "S2": SETTINGS},
        "join_key": KEYS,
        "splits": summaries,
        "test_rul": {"count": len(rul), "min": min(rul), "max": max(rul)},
        "schema": [
            {"name": name, "symbol": symbol, "unit": unit}
            for name, (symbol, unit) in zip(SENSORS, SENSOR_INFO)
        ],
        "notes": [
            "Two complementary views of one dataset, not independent datasets.",
            "Train and test engine IDs are separate namespaces.",
            "Test trajectories are intentionally truncated; this is not missing-data imputation.",
            "No regime labels, normalization, windows, PCA or autoencoder are fitted in CP1.",
            "RUL labels are kept in an evaluation-only file, never in the source views.",
            "Setting units are not explicit in the distributed readme; values remain unchanged.",
            "The readme's final sensor numbering is a typo: 26 fields contain 21 sensors.",
        ],
    }
    (out_dir / "quality_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--download", action="store_true", help="Download the pinned NASA inputs")
    args = parser.parse_args()
    if args.download:
        download(args.raw_dir)
    report = run(args.raw_dir, args.out_dir)
    for split, summary in report["splits"].items():
        print(f"{split}: {summary['rows']} rows / {summary['engines']} engines / S1=21 / S2=3")
    print(f"Report: {args.out_dir / 'quality_report.json'}")


if __name__ == "__main__":
    main()
