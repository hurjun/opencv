"""Regenerate the README detection demo image (assets/detection_demo.png).

This is a small, throwaway capture helper. It runs the *real* project code
end-to-end on a real photograph of a person and saves the annotated output:

* input  : ``grace_hopper.jpg`` shipped with matplotlib (a pinned dependency),
            a public-domain US Navy photo of Adm. Grace Hopper -- so the demo is
            fully reproducible offline, with no network download and no image
            committed to this repo beyond the generated result.
* model  : ``02_detection/model_loader.py`` -> real Faster R-CNN ResNet-50 FPN
            (downloads ~160 MB of weights on first run, then caches them).
* drawing: ``01_basics/draw_shapes.py`` -> the same boxes / labels / centre-point
            / info-bar visualization layer the detection drivers use.

The boxes and the "person 1.00" label are produced by the model, not hand-placed.
The full-resolution annotation is then downscaled with the project's own
``resize_keep_aspect`` helper so the committed PNG stays small (< 250 KB).

Run from the repository root::

    python scripts/capture_detection_demo.py
"""

from __future__ import annotations

import os
import sys

import cv2
import matplotlib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "01_basics"))
sys.path.insert(0, os.path.join(_ROOT, "02_detection"))

from draw_shapes import draw_bounding_box, draw_center_point, draw_info_bar  # noqa: E402
from model_loader import detect, load_model  # noqa: E402
from resize_crop import resize_keep_aspect  # noqa: E402

OUT_PATH = os.path.join(_ROOT, "assets", "detection_demo.png")


def _sample_image_path() -> str:
    """Path to matplotlib's bundled grace_hopper.jpg (a real person photo)."""
    return os.path.join(
        os.path.dirname(matplotlib.__file__),
        "mpl-data",
        "sample_data",
        "grace_hopper.jpg",
    )


def main() -> None:
    src = _sample_image_path()
    img = cv2.imread(src)
    if img is None:
        raise SystemExit(f"could not read sample image: {src}")
    print(f"[INFO] input: {src}  shape={img.shape}")

    model, device = load_model()
    detections = detect(model, device, img)
    print(f"[INFO] people detected: {len(detections)}")

    # Same visualization the detect_image.py driver produces.
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        draw_bounding_box(img, x1, y1, x2, y2, "person", det["score"])
        draw_center_point(img, x1, y1, x2, y2)
        print(f"  [{i + 1}] bbox={det['bbox']}  score={det['score']}")
    draw_info_bar(img, frame_idx=0)

    # Downscale the real annotated frame so the committed asset stays small.
    small = resize_keep_aspect(img, target_width=320)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    # PNG level-9 compression; this is a small downscaled frame so it stays tiny.
    cv2.imwrite(OUT_PATH, small, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"[done] wrote {OUT_PATH}  ({small.shape[1]}x{small.shape[0]}, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
