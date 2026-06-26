"""Day 3 - append-only CSV event logger.

Records anomaly events (ROI intrusions, motion) with a wall-clock timestamp and
frame index so a run can be audited after the fact. The file is opened in append
mode and a header is written only when the file is new.
"""

from __future__ import annotations

import csv
import os
from datetime import UTC, datetime
from typing import Any

FIELDNAMES = ["timestamp", "frame_idx", "event_type", "detail"]


class EventLogger:
    """Append anomaly events to a CSV file."""

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        is_new = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        # Line-buffered so events are flushed promptly during long runs.
        self._file = open(csv_path, "a", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        if is_new:
            self._writer.writeheader()
        self.count = 0

    def log(self, frame_idx: int, event_type: str, detail: Any = "") -> None:
        """Write a single event row with a UTC ISO-8601 timestamp."""
        self._writer.writerow(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "frame_idx": frame_idx,
                "event_type": event_type,
                "detail": detail,
            }
        )
        self.count += 1

    def close(self) -> None:
        """Flush and close the underlying file."""
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> EventLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


if __name__ == "__main__":
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "events_demo.csv")
    with EventLogger(path) as logger:
        logger.log(10, "roi_intrusion", "center=(305,265) score=0.95")
        logger.log(12, "motion", "ratio=0.07")
    print(f"[TEST] wrote {logger.count} events -> {path}")
