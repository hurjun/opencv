"""Unit tests for the ROI intrusion rule engine (pure logic)."""

import roi_guard as rg

ROI = (200, 150, 450, 380)


def test_box_center() -> None:
    assert rg.box_center([200, 100, 300, 200]) == (250, 150)


def test_is_in_roi_inside() -> None:
    assert rg.is_in_roi([220, 160, 310, 370], ROI) is True


def test_is_in_roi_outside() -> None:
    assert rg.is_in_roi([50, 100, 140, 300], ROI) is False


def test_is_in_roi_on_edge_is_excluded() -> None:
    # Strict inequality: a centre exactly on the border is not "inside".
    bbox = [ROI[0] * 2 - 1, 0, 1, 0]  # cx == ROI[0]
    assert rg.box_center(bbox)[0] == ROI[0]
    assert rg.is_in_roi(bbox, ROI) is False


def test_check_intrusion_filters_and_enriches() -> None:
    dets = [
        {"bbox": [220, 160, 310, 370], "score": 0.95},
        {"bbox": [50, 100, 140, 300], "score": 0.71},
    ]
    hits = rg.check_intrusion(dets, ROI)
    assert len(hits) == 1
    assert hits[0]["score"] == 0.95
    assert hits[0]["center"] == (265, 265)
    # Original detections must not be mutated.
    assert "center" not in dets[0]
