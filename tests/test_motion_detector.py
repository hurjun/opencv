"""Unit tests for the MOG2 motion detector."""

import numpy as np
from motion_detector import MotionDetector


def test_static_frames_report_no_motion() -> None:
    det = MotionDetector(motion_threshold=0.01)
    static = np.zeros((120, 160, 3), dtype=np.uint8)
    is_motion = True
    for _ in range(10):
        _, _, is_motion = det.apply(static)
    assert is_motion is False


def test_moving_block_triggers_motion() -> None:
    det = MotionDetector(motion_threshold=0.01)
    static = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(10):
        det.apply(static)

    moving = static.copy()
    moving[40:80, 60:100] = 255  # bright block appears
    _, ratio, is_motion = det.apply(moving)
    assert ratio > 0.0
    assert is_motion is True
