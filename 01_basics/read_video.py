"""
read_video.py — Day 1

영상을 읽고 매 30프레임마다 이미지로 저장한다.

핵심 개념:
  - cv2.VideoCapture: 영상/웹캠 입력 스트림
  - cap.read(): (성공여부, 프레임) 튜플 반환
  - 프레임은 (H, W, 3) uint8 BGR NumPy 배열
"""

import cv2
import os
import sys

# ── 설정 ──────────────────────────────────────────────
VIDEO_SOURCE = 0                   # 맥북 내장 웹캠
OUTPUT_DIR   = "data/output"
SAVE_EVERY   = 30                  # N프레임마다 저장
# ──────────────────────────────────────────────────────


def open_capture(source):
    """VideoCapture를 열고 기본 정보를 출력한다."""
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] 영상을 열 수 없습니다: {source}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] 영상 열기 성공")
    print(f"       해상도: {width}x{height}  FPS: {fps:.1f}  총 프레임: {total_frames}")

    return cap


def extract_frames(cap, output_dir, save_every):
    """프레임을 순서대로 읽으면서 save_every 간격으로 저장한다."""
    os.makedirs(output_dir, exist_ok=True)

    frame_idx   = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        # ret=False → 영상 끝 또는 읽기 실패
        if not ret:
            break

        if frame_idx % save_every == 0:
            filename = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f"  저장: {filename}  |  shape={frame.shape}  dtype={frame.dtype}")

        frame_idx += 1

    print(f"\n[완료] 총 {frame_idx}프레임 중 {saved_count}장 저장 → {output_dir}/")
    return frame_idx, saved_count


def main():
    cap = open_capture(VIDEO_SOURCE)

    try:
        extract_frames(cap, OUTPUT_DIR, SAVE_EVERY)
    finally:
        # 반드시 해제 — 안 하면 다음 실행 시 파일 잠금 문제
        cap.release()
        print("[INFO] VideoCapture 해제 완료")


if __name__ == "__main__":
    main()
