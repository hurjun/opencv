"""Day 3 - ROI intrusion rule engine.

A detection counts as an intrusion when its bounding-box *centre* falls inside a
rectangular region of interest (ROI). Using the centre point (rather than any
box overlap) gives a simple, stable geofence/keep-out rule that maps directly
onto industrial-safety and fleet keep-out-zone checks.

This module is pure logic with no OpenCV/torch dependency, which makes it cheap
to unit-test.
"""

from __future__ import annotations

from typing import Any

Box = tuple[int, int, int, int]
Detection = dict[str, Any]


def box_center(bbox: list[int] | Box) -> tuple[int, int]:
    """Return the integer ``(cx, cy)`` centre of a ``[x1, y1, x2, y2]`` box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def is_in_roi(bbox: list[int] | Box, roi: Box) -> bool:
    """Return ``True`` if the box centre lies strictly inside ``roi``."""
    cx, cy = box_center(bbox)
    rx1, ry1, rx2, ry2 = roi
    return rx1 < cx < rx2 and ry1 < cy < ry2


def check_intrusion(detections: list[Detection], roi: Box) -> list[Detection]:
    """Return the subset of ``detections`` whose centre is inside ``roi``.

    Each returned detection is the original dict augmented with its ``center``,
    so callers can annotate or log it without recomputing.
    """
    intruders: list[Detection] = []
    for det in detections:
        if is_in_roi(det["bbox"], roi):
            enriched = dict(det)
            enriched["center"] = box_center(det["bbox"])
            intruders.append(enriched)
    return intruders


if __name__ == "__main__":
    roi: Box = (200, 150, 450, 380)
    sample = [
        {"bbox": [220, 160, 310, 370], "score": 0.95},  # centre inside ROI
        {"bbox": [50, 100, 140, 300], "score": 0.71},  # centre outside ROI
    ]
    hits = check_intrusion(sample, roi)
    print(f"[TEST] {len(hits)}/{len(sample)} detections inside ROI {roi}")
    for h in hits:
        print(f"  intruder center={h['center']} score={h['score']}")
