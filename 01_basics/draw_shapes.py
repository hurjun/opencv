"""Day 1 - annotation / visualization helpers.

The detection pipeline's visualization layer is just a composition of these
helpers: bounding boxes, labels, centre points, ROI overlays and an info bar.

Key conventions:

* OpenCV coordinates: origin ``(0, 0)`` is top-left, x grows right, y grows down.
* Colours are BGR tuples, e.g. red is ``(0, 0, 255)``.
* ``thickness=-1`` fills a shape; a positive value is the border thickness.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

OUTPUT_DIR = "data/output"

Color = tuple[int, int, int]
Box = tuple[int, int, int, int]


def make_blank_canvas(height: int = 480, width: int = 640) -> np.ndarray:
    """Create a black ``(H, W, 3)`` uint8 canvas."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def draw_bounding_box(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label: str,
    score: float,
    color: Color = (0, 255, 0),
) -> None:
    """Draw a detection box with a filled label tag above it."""
    cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)

    text = f"{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # Filled tag behind the text improves readability over busy backgrounds.
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color=color, thickness=-1)
    cv2.putText(
        img,
        text,
        (x1, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 0, 0),
        thickness=1,
    )


def draw_center_point(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Color = (0, 0, 255),
) -> None:
    """Mark the box centre - the reference point used for ROI intrusion checks."""
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (cx, cy), radius=5, color=color, thickness=-1)


def draw_roi_zone(img: np.ndarray, roi: Box, color: Color = (0, 255, 255)) -> None:
    """Draw a translucent region-of-interest overlay with a solid border."""
    rx1, ry1, rx2, ry2 = roi

    # Blend a filled rectangle over a copy so the fill stays semi-transparent.
    overlay = img.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color=color, thickness=-1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)  # 25% opacity

    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color=color, thickness=2)
    cv2.putText(
        img,
        "ROI ZONE",
        (rx1 + 4, ry1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def draw_info_bar(img: np.ndarray, frame_idx: int, fps: float | None = None) -> None:
    """Draw a frame-number / FPS readout in the bottom-left corner."""
    h = img.shape[0]
    text = f"Frame: {frame_idx}"
    if fps is not None:
        text += f"  FPS: {fps:.1f}"
    cv2.putText(
        img,
        text,
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )


def demo() -> None:
    """Render a synthetic scene illustrating the visualization layer.

    No webcam or model is involved - the boxes are hard-coded sample detections,
    coloured green when their centre falls inside the ROI and grey otherwise.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = make_blank_canvas()

    roi: Box = (200, 150, 450, 380)
    draw_roi_zone(img, roi)

    detections = [
        {"bbox": (220, 160, 310, 370), "label": "person", "score": 0.95},
        {"bbox": (320, 180, 410, 360), "label": "person", "score": 0.82},
        {"bbox": (50, 100, 140, 300), "label": "person", "score": 0.71},
    ]

    in_roi_color: Color = (0, 255, 0)
    out_roi_color: Color = (128, 128, 128)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx1, ry1, rx2, ry2 = roi
        in_roi = rx1 < cx < rx2 and ry1 < cy < ry2

        color = in_roi_color if in_roi else out_roi_color
        draw_bounding_box(img, x1, y1, x2, y2, det["label"], det["score"], color)
        draw_center_point(img, x1, y1, x2, y2)

    draw_info_bar(img, frame_idx=0, fps=30.0)

    out_path = os.path.join(OUTPUT_DIR, "draw_demo.jpg")
    cv2.imwrite(out_path, img)
    print(f"[done] saved: {out_path}")


if __name__ == "__main__":
    demo()
