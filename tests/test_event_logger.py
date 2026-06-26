"""Unit tests for the CSV event logger."""

import csv

from event_logger import EventLogger


def test_writes_header_and_rows(tmp_path) -> None:
    path = tmp_path / "events.csv"
    with EventLogger(str(path)) as logger:
        logger.log(10, "roi_intrusion", "center=(305,265)")
        logger.log(12, "motion", "ratio=0.07")
    assert logger.count == 2

    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert rows[0]["event_type"] == "roi_intrusion"
    assert rows[1]["frame_idx"] == "12"
    assert rows[0]["timestamp"]  # non-empty ISO timestamp


def test_appends_without_duplicate_header(tmp_path) -> None:
    path = tmp_path / "events.csv"
    with EventLogger(str(path)) as a:
        a.log(1, "motion", "")
    with EventLogger(str(path)) as b:
        b.log(2, "motion", "")

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3  # one header + two data rows
    assert lines[0].startswith("timestamp")
