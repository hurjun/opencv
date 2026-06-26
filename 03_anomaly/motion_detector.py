"""Day 3 - background-subtraction motion detector.

Wraps ``cv2.BackgroundSubtractorMOG2`` to flag frames that contain significant
motion. It is a cheap, model-free signal that runs every frame and complements
the (more expensive) detector: motion can gate when the detector runs, or be
logged on its own.
"""

from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    """Detect motion via the fraction of foreground pixels per frame."""

    def __init__(
        self,
        motion_threshold: float = 0.02,
        history: int = 200,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
    ) -> None:
        """Create the detector.

        Args:
            motion_threshold: foreground-pixel fraction above which a frame is
                considered "in motion" (``0.02`` == 2% of the frame).
            history: number of frames used to build the background model.
            var_threshold: MOG2 variance threshold (lower = more sensitive).
            detect_shadows: whether MOG2 marks shadows (value 127) separately.
        """
        self.motion_threshold = motion_threshold
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )

    def apply(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Update the background model and score motion for ``frame_bgr``.

        Returns ``(mask, motion_ratio, is_motion)`` where ``mask`` is the binary
        foreground mask, ``motion_ratio`` is the foreground-pixel fraction, and
        ``is_motion`` is ``motion_ratio >= motion_threshold``.
        """
        mask = self._subtractor.apply(frame_bgr)
        # Drop shadow pixels (value 127) so only hard foreground (255) counts.
        foreground = (mask == 255).astype(np.uint8)
        motion_ratio = float(foreground.mean())
        return mask, motion_ratio, motion_ratio >= self.motion_threshold


if __name__ == "__main__":
    # Smoke check: a static frame then a frame with a bright moving block.
    detector = MotionDetector(motion_threshold=0.01)
    static = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(5):
        detector.apply(static)

    moving = static.copy()
    moving[40:80, 60:100] = 255
    _, ratio, is_motion = detector.apply(moving)
    print(f"[TEST] motion_ratio={ratio:.3f}  is_motion={is_motion}")
