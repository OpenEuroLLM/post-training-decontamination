import csv
from collections import defaultdict
from pathlib import Path

MATCH_COLUMNS = ("summary_id", "training_row_id", "benchmark_row_id")


def prepare_matches(path: Path, summary_ids: set[int]) -> None:
    if not path.exists():
        if summary_ids:
            raise FileNotFoundError(f"Missing matches file: {path}")
        with path.open("x", newline="") as file:
            csv.writer(file).writerow(MATCH_COLUMNS)
        return

    with path.open("r+b") as file:
        if file.readline().strip() != ",".join(MATCH_COLUMNS).encode():
            raise ValueError(f"Invalid matches header: {path}")
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            # A scan is committed only after its summary row is published.
            if not line.endswith(b"\n") or int(line.split(b",", 1)[0]) not in summary_ids:
                file.truncate(offset)
                break


def load_training_ids(path: Path, summary_ids: set[int]) -> dict[int, set[int]]:
    ids = defaultdict(set)
    with path.open(newline="") as file:
        rows = csv.reader(file)
        if tuple(next(rows, ())) != MATCH_COLUMNS:
            raise ValueError(f"Invalid matches header: {path}")
        for row in rows:
            summary_id, training_row_id, _ = map(int, row)
            if summary_id not in summary_ids:
                raise ValueError(f"Unknown summary ID {summary_id} in {path}. Resume the search first.")
            ids[summary_id].add(training_row_id)
    return dict(ids)
