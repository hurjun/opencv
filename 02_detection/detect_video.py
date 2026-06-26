"""Day 2 - run person detection on a video/webcam stream and save the result.

Key concepts:

* Detection is expensive, so the model only runs every ``--detect-every`` frames
  and the most recent boxes are reused in between (a common throughput trick).
* Per-frame FPS is measured with ``time`` for a rough throughput readout.
* ``cv2.VideoWriter`` saves the annotated stream to disk.

Usage (from the repository root)::

    python 02_detection/detect_video.py --source 0 --max-frames 300
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../02_detection
_ROOT = os.path.dirname(_HERE)  # repository root
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "01_basics"))

from draw_shapes import draw_bounding_box, draw_center_point, draw_info_bar  # noqa: E402
from model_loader import detect, load_model  # noqa: E402


def make_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    """Create a ``VideoWriter`` matching the capture's resolution and FPS."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))


def run(
    source: int | str,
    output_path: str,
    detect_every: int,
    max_frames: int,
) -> None:
    """Process a stream: detect, annotate, and write an output video."""
    model, device = load_model()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] could not open source: {source}")
        sys.exit(1)

    writer = make_writer(cap, output_path)

    frame_idx = 0
    detections: list = []  # reused between detection frames
    fps_display = 0.0

    print(f"[INFO] processing up to {max_frames} frames (Ctrl+C to stop)")
    try:
        while frame_idx < max_frames:
            t_start = time.time()

            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % detect_every == 0:
                detections = detect(model, device, frame)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                draw_bounding_box(frame, x1, y1, x2, y2, "person", det["score"])
                draw_center_point(frame, x1, y1, x2, y2)

            draw_info_bar(frame, frame_idx, fps=fps_display)
            writer.write(frame)

            elapsed = time.time() - t_start
            fps_display = 1.0 / elapsed if elapsed > 0 else 0.0

            if frame_idx % 30 == 0:
                print(
                    f"  frame={frame_idx:04d}  persons={len(detections)}  "
                    f"FPS={fps_display:.1f}"
                )
            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user")
    finally:
        cap.release()
        writer.release()
        print(f"[done] {frame_idx} frames -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect people in a video/webcam.")
    parser.add_argument(
        "--source", default="0", help="camera index (e.g. 0) or video file path"
    )
    parser.add_argument(
        "--output", default="data/output/result.mp4", help="output video path"
    )
    parser.add_argument(
        "--detect-every", type=int, default=5, help="run detection every Nth frame"
    )
    parser.add_argument(
        "--max-frames", type=int, default=300, help="stop after this many frames"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src: int | str = int(args.source) if args.source.isdigit() else args.source
    run(src, args.output, args.detect_every, args.max_frames)
