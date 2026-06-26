"""Day 2 - load Faster R-CNN and run person detection on BGR frames.

Key concepts:

* The torchvision model is pretrained on COCO (80 classes); ``label == 1`` is
  the *person* class.
* Model input is an RGB float tensor in ``[0, 1]``; output is a dict of
  ``boxes`` / ``labels`` / ``scores``.
* OpenCV frames are BGR, so a BGR -> RGB conversion is required before inference.

The pretrained weights (~160 MB) are downloaded by torchvision on the first call
to :func:`load_model`. Nothing is downloaded at import time, so this module is
safe to import in CI.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F

PERSON_LABEL = 1  # COCO class id for "person"
SCORE_THRESHOLD = 0.5  # discard detections below this confidence

Detection = dict[str, Any]


def load_model() -> tuple[torch.nn.Module, str]:
    """Load Faster R-CNN ResNet-50 FPN in eval mode on the best device.

    Downloads the pretrained weights on first use (~160 MB).
    """
    print("[INFO] loading model ...")
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()  # inference mode: disable dropout / batch-norm updates

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"[INFO] model ready (device: {device})")
    return model, device


def preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    """Convert an OpenCV BGR frame to a model-ready RGB tensor.

    1. BGR -> RGB (OpenCV vs. PyTorch channel order).
    2. ``(H, W, 3)`` uint8 -> ``(3, H, W)`` float32 normalised to ``[0, 1]``.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return F.to_tensor(frame_rgb)  # also divides by 255


def detect(
    model: torch.nn.Module,
    device: str,
    frame_bgr: np.ndarray,
    score_threshold: float = SCORE_THRESHOLD,
) -> list[Detection]:
    """Detect people in a BGR frame.

    Returns a list of ``{"bbox": [x1, y1, x2, y2], "score": float}`` dicts with
    pixel coordinates in the original frame, keeping only person detections above
    ``score_threshold``.
    """
    tensor = preprocess(frame_bgr).to(device)

    with torch.no_grad():  # no gradients -> faster, lower memory
        predictions = model([tensor])[0]

    results: list[Detection] = []
    for box, label, score in zip(
        predictions["boxes"],
        predictions["labels"],
        predictions["scores"],
        strict=True,  # torchvision returns equal-length boxes/labels/scores
    ):
        if label.item() != PERSON_LABEL:
            continue
        if score.item() < score_threshold:
            continue
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        results.append(
            {"bbox": [int(x1), int(y1), int(x2), int(y2)], "score": round(score.item(), 3)}
        )
    return results


if __name__ == "__main__":
    # Smoke check: run inference on a black dummy frame (expect 0 people).
    model, device = load_model()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detect(model, device, dummy)
    print(f"[TEST] dummy-frame detections: {result}")
    print("model_loader.py OK")
