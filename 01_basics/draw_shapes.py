"""
draw_shapes.py — Day 1

샘플 이미지를 만들고 그 위에 탐지 결과처럼 도형과 텍스트를 그린다.
실제 탐지 파이프라인에서 시각화는 이 함수들의 조합이다.

핵심 개념:
  - OpenCV 좌표계: 좌상단이 (0, 0), x→오른쪽, y→아래쪽
  - BGR 컬러: (Blue, Green, Red) 순서. 예) 빨강=(0,0,255)
  - thickness=-1 → 채우기(filled), 양수 → 테두리 두께
"""

import cv2
import numpy as np
import os

OUTPUT_DIR = "data/output"


def make_blank_canvas(height=480, width=640):
    """검정 배경 캔버스 생성. 이미지 = (H, W, 3) uint8 배열."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def draw_bounding_box(img, x1, y1, x2, y2, label, score, color=(0, 255, 0)):
    """탐지 결과 박스 + 라벨을 그린다."""
    # 사각형 테두리
    cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)

    # 라벨 배경 (텍스트 가독성 향상)
    text = f"{label} {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color=color, thickness=-1)

    # 라벨 텍스트 (배경 위에 검정 글씨)
    cv2.putText(img, text, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                color=(0, 0, 0), thickness=1)


def draw_center_point(img, x1, y1, x2, y2, color=(0, 0, 255)):
    """바운딩 박스 중심점 표시 — ROI 침입 판단에 쓰이는 기준점."""
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(img, (cx, cy), radius=5, color=color, thickness=-1)


def draw_roi_zone(img, roi, color=(0, 255, 255)):
    """관심 구역(ROI)을 반투명하게 표시한다."""
    rx1, ry1, rx2, ry2 = roi

    # 반투명 오버레이 — 원본을 복사한 뒤 채운 사각형을 그리고 합성
    overlay = img.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), color=color, thickness=-1)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)  # 25% 투명도

    # 테두리는 불투명하게
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color=color, thickness=2)
    cv2.putText(img, "ROI ZONE", (rx1 + 4, ry1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def draw_info_bar(img, frame_idx, fps=None):
    """좌하단에 프레임 번호 / FPS 정보를 표시한다."""
    h = img.shape[0]
    text = f"Frame: {frame_idx}"
    if fps is not None:
        text += f"  FPS: {fps:.1f}"
    cv2.putText(img, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def demo():
    """가상의 탐지 결과를 그려서 저장한다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img = make_blank_canvas()

    # ROI 구역
    roi = (200, 150, 450, 380)
    draw_roi_zone(img, roi)

    # 가상 탐지 결과 3개
    detections = [
        {"bbox": (220, 160, 310, 370), "label": "person", "score": 0.95},
        {"bbox": (320, 180, 410, 360), "label": "person", "score": 0.82},
        {"bbox": (50,  100, 140, 300), "label": "person", "score": 0.71},
    ]

    colors = {
        "in_roi":  (0, 255, 0),   # 초록 — ROI 내부
        "out_roi": (128, 128, 128) # 회색 — ROI 외부
    }

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        rx1, ry1, rx2, ry2 = roi
        in_roi = rx1 < cx < rx2 and ry1 < cy < ry2

        color = colors["in_roi"] if in_roi else colors["out_roi"]
        draw_bounding_box(img, x1, y1, x2, y2, det["label"], det["score"], color)
        draw_center_point(img, x1, y1, x2, y2)

    draw_info_bar(img, frame_idx=0, fps=30.0)

    out_path = os.path.join(OUTPUT_DIR, "draw_demo.jpg")
    cv2.imwrite(out_path, img)
    print(f"[완료] 저장: {out_path}")


if __name__ == "__main__":
    demo()
