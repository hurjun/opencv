"""Day 1 - read a video/webcam stream and extract frames to disk.

Key concepts:

* ``cv2.VideoCapture`` opens a video file or camera index as an input stream.
* ``cap.read()`` returns ``(ok, frame)``; ``ok`` is ``False`` at end-of-stream.
* A frame is an ``(H, W, 3)`` uint8 BGR NumPy array.

Run from the repository root, e.g.::

    python 01_basics/read_video.py --source 0 --every 30 --output data/output

Note: captured frames are written to ``data/output`` which is git-ignored, so
personal webcam imagery is never committed.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2


def open_capture(source: int | str) -> cv2.VideoCapture:
    """Open a ``VideoCapture`` and print basic stream info."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] could not open source: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("[INFO] capture opened")
    print(
        f"       resolution: {width}x{height}  fps: {fps:.1f}  frames: {total_frames}"
    )
    return cap


def extract_frames(
    cap: cv2.VideoCapture, output_dir: str, save_every: int
) -> tuple[int, int]:
    """Read frames sequentially and save every ``save_every``-th one."""
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:  # end of stream or read failure
            break

        if frame_idx % save_every == 0:
            filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"  saved: {filename}  shape={frame.shape}  dtype={frame.dtype}")

        frame_idx += 1

    print(f"\n[done] saved {saved_count}/{frame_idx} frames -> {output_dir}/")
    return frame_idx, saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video/webcam.")
    parser.add_argument(
        "--source",
        default="0",
        help="camera index (e.g. 0) or path to a video file",
    )
    parser.add_argument(
        "--every", type=int, default=30, help="save every Nth frame"
    )
    parser.add_argument(
        "--output", default="data/output", help="directory to write frames into"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # A bare integer string means a camera index; anything else is a file path.
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    cap = open_capture(source)
    try:
        extract_frames(cap, args.output, args.every)
    finally:
        # Always release - otherwise the device/file can stay locked.
        cap.release()
        print("[INFO] VideoCapture released")


if __name__ == "__main__":
    main()
