"""
detect_image.py — Day 2

저장된 이미지 파일 하나에 대해 탐지를 실행하고 결과를 시각화한다.
read_video.py로 저장한 프레임 이미지를 입력으로 사용한다.
"""

import sys
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from 02_detection.model_loader import load_model, detect
from 01_basics.draw_shapes import draw_bounding_box, draw_center_point, draw_info_bar

# ── 설정 ──────────────────────────────────────────────
INPUT_IMAGE = "data/output/frame_00000.jpg"  # read_video.py 로 저장한 프레임
OUTPUT_DIR  = "data/output"
# ──────────────────────────────────────────────────────


def run(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 이미지를 찾을 수 없습니다: {image_path}")
        print("  → 먼저 read_video.py를 실행해서 프레임을 저장하세요.")
        sys.exit(1)

    print(f"[INFO] 이미지 로드: {image_path}  shape={img.shape}")

    model, device = load_model()
    detections = detect(model, device, img)
    print(f"[INFO] 탐지된 사람: {len(detections)}명")

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        draw_bounding_box(img, x1, y1, x2, y2, "person", det["score"])
        draw_center_point(img, x1, y1, x2, y2)
        print(f"  [{i+1}] bbox={det['bbox']}  score={det['score']}")

    draw_info_bar(img, frame_idx=0)

    out_path = os.path.join(OUTPUT_DIR, "detected_image.jpg")
    cv2.imwrite(out_path, img)
    print(f"[완료] 결과 저장: {out_path}")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_IMAGE
    run(image_path)
