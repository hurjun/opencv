"""
detect_video.py — Day 2

웹캠 영상을 실시간으로 읽어 person 탐지를 실행하고,
탐지 결과가 그려진 영상을 data/output/에 저장한다.

핵심 개념:
  - 모델 추론은 느리므로 매 N프레임마다 실행 (DETECT_EVERY)
  - FPS 측정: time 모듈로 프레임 처리 시간 계산
  - VideoWriter: 결과 영상 파일로 저장
"""

import cv2
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from 02_detection.model_loader import load_model, detect
from 01_basics.draw_shapes import draw_bounding_box, draw_center_point, draw_info_bar

# ── 설정 ──────────────────────────────────────────────
VIDEO_SOURCE  = 0                       # 웹캠
OUTPUT_PATH   = "data/output/result.mp4"
DETECT_EVERY  = 5                       # 매 N프레임마다 탐지 실행 (속도 조절)
MAX_FRAMES    = 300                     # 최대 처리 프레임 수 (Ctrl+C 대신 자동 종료)
# ──────────────────────────────────────────────────────


def make_writer(cap, output_path):
    """VideoCapture와 같은 해상도로 VideoWriter를 생성한다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))


def run():
    model, device = load_model()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        sys.exit(1)

    writer = make_writer(cap, OUTPUT_PATH)

    frame_idx  = 0
    detections = []          # 최근 탐지 결과를 재사용 (DETECT_EVERY 간격)
    fps_display = 0.0

    print(f"[INFO] 영상 처리 시작 (최대 {MAX_FRAMES}프레임)")
    print("  Ctrl+C로 조기 종료 가능")

    try:
        while frame_idx < MAX_FRAMES:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # 매 DETECT_EVERY 프레임마다 탐지 실행
            if frame_idx % DETECT_EVERY == 0:
                detections = detect(model, device, frame)

            # 탐지 결과 시각화
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                draw_bounding_box(frame, x1, y1, x2, y2, "person", det["score"])
                draw_center_point(frame, x1, y1, x2, y2)

            draw_info_bar(frame, frame_idx, fps=fps_display)
            writer.write(frame)

            elapsed = time.time() - t_start
            fps_display = 1.0 / elapsed if elapsed > 0 else 0

            if frame_idx % 30 == 0:
                print(f"  frame={frame_idx:04d}  persons={len(detections)}  FPS={fps_display:.1f}")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단")
    finally:
        cap.release()
        writer.release()
        print(f"[완료] {frame_idx}프레임 처리 → {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
