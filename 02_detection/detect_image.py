"""Day 2 - run person detection on a single image and visualize the result.

Usage (from the repository root)::

    python 02_detection/detect_image.py path/to/image.jpg

The package directories are numbered (``01_basics`` ...), which are not valid
Python module names, so the sibling modules are imported by adding their
directories to ``sys.path`` instead of using package imports.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../02_detection
_ROOT = os.path.dirname(_HERE)  # repository root
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "01_basics"))

from draw_shapes import draw_bounding_box, draw_center_point, draw_info_bar  # noqa: E402
from model_loader import detect, load_model  # noqa: E402


def run(image_path: str, output_dir: str = "data/output") -> int:
    """Detect people in ``image_path`` and save an annotated copy.

    Returns the number of people detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] could not read image: {image_path}")
        print("  -> generate one first, e.g. python 01_basics/draw_shapes.py")
        sys.exit(1)

    print(f"[INFO] loaded image: {image_path}  shape={img.shape}")

    model, device = load_model()
    detections = detect(model, device, img)
    print(f"[INFO] people detected: {len(detections)}")

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        draw_bounding_box(img, x1, y1, x2, y2, "person", det["score"])
        draw_center_point(img, x1, y1, x2, y2)
        print(f"  [{i + 1}] bbox={det['bbox']}  score={det['score']}")

    draw_info_bar(img, frame_idx=0)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "detected_image.jpg")
    cv2.imwrite(out_path, img)
    print(f"[done] result saved: {out_path}")
    return len(detections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people in a single image.")
    parser.add_argument(
        "image",
        nargs="?",
        default="data/output/frame_00000.jpg",
        help="path to the input image",
    )
    parser.add_argument(
        "--output", default="data/output", help="directory to write the result into"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.image, args.output)
