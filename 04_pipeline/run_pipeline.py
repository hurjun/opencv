"""Day 4 - end-to-end pipeline: capture -> detect -> ROI rule -> log -> output.

Ties the Day 1-3 modules into a single loop:

    read frame -> (every Nth frame) detect people -> flag ROI intrusions
        -> background-subtraction motion -> log events to CSV
        -> annotate -> write output video

Run from the repository root, e.g.::

    python 04_pipeline/run_pipeline.py --source 0 --roi 200 150 450 380
    python 04_pipeline/run_pipeline.py --source clip.mp4 --no-detect   # motion only

``--no-detect`` skips the detector entirely (and never imports torch), so the
motion + logging path can run on machines without the model weights.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../04_pipeline
_ROOT = os.path.dirname(_HERE)  # repository root
for _sub in ("01_basics", "02_detection", "03_anomaly"):
    sys.path.insert(0, os.path.join(_ROOT, _sub))

from draw_shapes import (  # noqa: E402
    draw_bounding_box,
    draw_center_point,
    draw_info_bar,
    draw_roi_zone,
)
from event_logger import EventLogger  # noqa: E402
from motion_detector import MotionDetector  # noqa: E402
from roi_guard import box_center, is_in_roi  # noqa: E402

INTRUDER_COLOR = (0, 0, 255)  # red - centre inside ROI
SAFE_COLOR = (128, 128, 128)  # grey - outside ROI


def run(
    source: int | str,
    roi: tuple[int, int, int, int],
    output_path: str,
    events_path: str,
    detect_every: int,
    max_frames: int,
    use_detector: bool,
) -> None:
    """Run the full pipeline over a stream."""
    detect = load_model = None
    model = device = None
    if use_detector:
        # Imported lazily so --no-detect never requires torch.
        from model_loader import detect, load_model  # noqa: E402

        model, device = load_model()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] could not open source: {source}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    motion = MotionDetector()
    logger = EventLogger(events_path)

    frame_idx = 0
    detections: list = []
    fps_display = 0.0

    print(f"[INFO] pipeline running (roi={roi}, detector={'on' if use_detector else 'off'})")
    try:
        while frame_idx < max_frames:
            t_start = time.time()

            ok, frame = cap.read()
            if not ok:
                break

            if use_detector and frame_idx % detect_every == 0:
                detections = detect(model, device, frame)

            intruders = [d for d in detections if is_in_roi(d["bbox"], roi)]
            intruder_ids = {id(d) for d in intruders}

            _, motion_ratio, is_motion = motion.apply(frame)

            # Annotate.
            draw_roi_zone(frame, roi)
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                in_roi = id(det) in intruder_ids
                color = INTRUDER_COLOR if in_roi else SAFE_COLOR
                draw_bounding_box(frame, x1, y1, x2, y2, "person", det["score"], color)
                draw_center_point(frame, x1, y1, x2, y2)
            draw_info_bar(frame, frame_idx, fps=fps_display)
            writer.write(frame)

            # Log events.
            for intr in intruders:
                logger.log(
                    frame_idx,
                    "roi_intrusion",
                    f"center={box_center(intr['bbox'])} score={intr['score']}",
                )
            if is_motion:
                logger.log(frame_idx, "motion", f"ratio={motion_ratio:.3f}")

            elapsed = time.time() - t_start
            fps_display = 1.0 / elapsed if elapsed > 0 else 0.0
            if frame_idx % 30 == 0:
                print(
                    f"  frame={frame_idx:04d}  persons={len(detections)}  "
                    f"intruders={len(intruders)}  motion={is_motion}  "
                    f"FPS={fps_display:.1f}"
                )
            frame_idx += 1
    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user")
    finally:
        cap.release()
        writer.release()
        logger.close()
        print(
            f"[done] {frame_idx} frames -> {output_path}; "
            f"{logger.count} events -> {events_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full vision pipeline.")
    parser.add_argument(
        "--source", default="0", help="camera index (e.g. 0) or video file path"
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=[200, 150, 450, 380],
        help="rectangular region of interest",
    )
    parser.add_argument(
        "--output", default="data/output/pipeline_result.mp4", help="output video path"
    )
    parser.add_argument(
        "--events", default="data/output/events.csv", help="event-log CSV path"
    )
    parser.add_argument(
        "--detect-every", type=int, default=5, help="run detection every Nth frame"
    )
    parser.add_argument(
        "--max-frames", type=int, default=300, help="stop after this many frames"
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="skip the detector (motion + logging only; does not require torch)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src: int | str = int(args.source) if args.source.isdigit() else args.source
    run(
        source=src,
        roi=tuple(args.roi),
        output_path=args.output,
        events_path=args.events,
        detect_every=args.detect_every,
        max_frames=args.max_frames,
        use_detector=not args.no_detect,
    )
