"""
model_loader.py — Day 2

Faster R-CNN 모델을 로드하고, BGR 프레임을 받아 person 탐지 결과를 반환한다.

핵심 개념:
  - torchvision 사전학습 모델: COCO 데이터셋 80개 클래스 학습됨
  - COCO label 1 = person
  - 모델 입력: RGB float32 텐서 (0.0~1.0), 출력: boxes/labels/scores 딕셔너리
  - BGR(OpenCV) → RGB(PyTorch) 변환 필수
"""

import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

PERSON_LABEL   = 1      # COCO 데이터셋에서 사람 클래스 번호
SCORE_THRESHOLD = 0.5   # 이 값 미만의 탐지 결과는 버림


def load_model():
    """
    Faster R-CNN ResNet-50 FPN 모델을 로드한다.
    첫 실행 시 가중치를 다운로드하므로 시간이 걸린다 (~170MB).
    """
    print("[INFO] 모델 로딩 중...")
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()  # 추론 모드 — 드롭아웃/배치정규화 비활성화

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[INFO] 모델 로드 완료 (device: {device})")
    return model, device


def preprocess(frame_bgr):
    """
    OpenCV BGR 프레임 → PyTorch 텐서로 변환.
      1. BGR → RGB (OpenCV와 PyTorch의 채널 순서 차이)
      2. (H,W,3) uint8 → (3,H,W) float32, 값 범위 0~1
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = F.to_tensor(frame_rgb)  # 자동으로 /255 정규화
    return tensor


def detect(model, device, frame_bgr, score_threshold=SCORE_THRESHOLD):
    """
    BGR 프레임을 받아 person 탐지 결과를 반환한다.

    Returns:
        list of dict: [{"bbox": [x1,y1,x2,y2], "score": 0.95}, ...]
        bbox 좌표는 원본 프레임 기준 픽셀값 (int)
    """
    tensor = preprocess(frame_bgr).to(device)

    with torch.no_grad():               # 그래디언트 계산 끄기 → 속도/메모리 절약
        predictions = model([tensor])[0]

    results = []
    for box, label, score in zip(
        predictions["boxes"],
        predictions["labels"],
        predictions["scores"]
    ):
        if label.item() != PERSON_LABEL:
            continue
        if score.item() < score_threshold:
            continue

        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        results.append({
            "bbox":  [x1, y1, x2, y2],
            "score": round(score.item(), 3),
        })

    return results


if __name__ == "__main__":
    # 동작 확인: 검정 더미 프레임으로 추론 테스트
    import numpy as np
    model, device = load_model()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detect(model, device, dummy)
    print(f"[TEST] 더미 프레임 탐지 결과: {result}")
    print("model_loader.py 정상 동작 확인")
