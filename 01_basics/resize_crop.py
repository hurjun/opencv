"""
resize_crop.py — Day 1

전처리의 두 핵심 연산: 크기 조정(resize)과 관심 영역 크롭(crop).

파이프라인에서 이 두 연산이 필요한 이유:
  - resize: 모델은 고정 입력 크기를 요구하거나, 속도를 위해 축소 필요
  - crop:   ROI만 따로 분석하면 연산량 감소 + 정확도 향상 가능

핵심 개념:
  - cv2.resize 보간법(interpolation): 축소=AREA, 확대=LINEAR or CUBIC
  - NumPy 슬라이싱: img[y1:y2, x1:x2] — (행, 열) 순서 주의
  - 종횡비(aspect ratio) 유지 리사이즈
"""

import cv2
import numpy as np
import os

OUTPUT_DIR = "data/output"


def resize_fixed(img, width, height):
    """고정 크기로 리사이즈. 종횡비가 달라질 수 있다."""
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_by_scale(img, scale):
    """비율로 리사이즈. scale=0.5 → 가로세로 절반."""
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def resize_keep_aspect(img, target_width):
    """가로 기준으로 종횡비를 유지하며 리사이즈."""
    h, w = img.shape[:2]
    ratio = target_width / w
    target_height = int(h * ratio)
    return cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def crop_roi(img, x1, y1, x2, y2):
    """
    관심 영역(ROI)을 잘라낸다.
    img[y1:y2, x1:x2] — NumPy는 (행=y, 열=x) 순서임을 주의.
    좌표가 이미지 경계를 벗어나면 자동으로 클리핑된다.
    """
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return img[y1:y2, x1:x2]


def letterbox(img, target_size=640):
    """
    종횡비를 유지하면서 정사각형 캔버스 중앙에 배치 (letterbox).
    YOLO 계열 모델에 입력할 때 자주 쓰는 전처리 방식.
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 검정 패딩 캔버스
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_y = (target_size - new_h) // 2
    pad_x = (target_size - new_w) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    return canvas


def demo():
    """각 함수 결과를 저장해서 비교한다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 샘플 이미지: 640x480 그레이디언트 배경
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(480):
        img[i, :] = [i // 2, 80, 255 - i // 2]   # 색상 그레이디언트

    # 1. 고정 크기 리사이즈
    fixed = resize_fixed(img, 320, 320)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "resize_fixed.jpg"), fixed)
    print(f"[1] 고정 리사이즈: {img.shape} → {fixed.shape}")

    # 2. 비율 리사이즈
    scaled = resize_by_scale(img, 0.5)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "resize_scaled.jpg"), scaled)
    print(f"[2] 비율 리사이즈(0.5): {img.shape} → {scaled.shape}")

    # 3. 종횡비 유지
    kept = resize_keep_aspect(img, target_width=400)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "resize_aspect.jpg"), kept)
    print(f"[3] 종횡비 유지: {img.shape} → {kept.shape}")

    # 4. ROI 크롭
    cropped = crop_roi(img, x1=200, y1=100, x2=500, y2=380)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "crop_roi.jpg"), cropped)
    print(f"[4] ROI 크롭: {img.shape} → {cropped.shape}")

    # 5. Letterbox
    lb = letterbox(img, target_size=640)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "letterbox.jpg"), lb)
    print(f"[5] Letterbox(640): {img.shape} → {lb.shape}")

    print(f"\n[완료] 결과 이미지 → {OUTPUT_DIR}/")


if __name__ == "__main__":
    demo()
