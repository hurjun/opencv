"""Day 1 - image preprocessing primitives (resize / crop / letterbox).

These are the two core preprocessing operations every detection pipeline needs:

* ``resize`` - models expect a fixed input size, and downscaling speeds up inference.
* ``crop``   - analysing only a region of interest (ROI) reduces compute and can
  improve accuracy by removing irrelevant background.

Key conventions:

* ``cv2.resize`` interpolation: use ``INTER_AREA`` for shrinking and
  ``INTER_LINEAR``/``INTER_CUBIC`` for enlarging.
* NumPy slicing is ``img[y1:y2, x1:x2]`` - rows (y) come before columns (x).
* Aspect-ratio-preserving resize avoids distorting object shapes.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

OUTPUT_DIR = "data/output"


def resize_fixed(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to an exact ``(width, height)``; aspect ratio may change."""
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_by_scale(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize by a uniform ``scale`` factor (e.g. ``0.5`` halves each side)."""
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def resize_keep_aspect(img: np.ndarray, target_width: int) -> np.ndarray:
    """Resize to ``target_width`` while preserving the aspect ratio."""
    h, w = img.shape[:2]
    ratio = target_width / w
    target_height = int(h * ratio)
    return cv2.resize(
        img, (target_width, target_height), interpolation=cv2.INTER_LINEAR
    )


def crop_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop a region of interest, clamping coordinates to the image bounds.

    NumPy indexes as ``img[y1:y2, x1:x2]`` (row=y, column=x). Coordinates that
    fall outside the frame are clipped so the call never raises.
    """
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return img[y1:y2, x1:x2]


def letterbox(img: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Resize keeping aspect ratio, then centre on a square padded canvas.

    This is the standard preprocessing used by YOLO-style detectors: the longest
    side is scaled to ``target_size`` and the remainder is zero-padded, so the
    output is always ``(target_size, target_size, 3)`` with no shape distortion.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas


def _make_gradient(height: int = 480, width: int = 640) -> np.ndarray:
    """Build a synthetic BGR gradient image (no external assets needed)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        img[i, :] = [i // 2, 80, 255 - i // 2]
    return img


def demo() -> None:
    """Run each preprocessing op on a synthetic image and save the results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = _make_gradient()

    fixed = resize_fixed(img, 320, 320)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "resize_fixed.jpg"), fixed)
    print(f"[1] fixed resize:      {img.shape} -> {fixed.shape}")

    scaled = resize_by_scale(img, 0.5)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "resize_scaled.jpg"), scaled)
    print(f"[2] scale resize 0.5:  {img.shape} -> {scaled.shape}")

    kept = resize_keep_aspect(img, target_width=400)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "resize_aspect.jpg"), kept)
    print(f"[3] keep aspect:       {img.shape} -> {kept.shape}")

    cropped = crop_roi(img, x1=200, y1=100, x2=500, y2=380)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "crop_roi.jpg"), cropped)
    print(f"[4] ROI crop:          {img.shape} -> {cropped.shape}")

    lb = letterbox(img, target_size=640)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "letterbox.jpg"), lb)
    print(f"[5] letterbox 640:     {img.shape} -> {lb.shape}")

    print(f"\n[done] results written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    demo()
