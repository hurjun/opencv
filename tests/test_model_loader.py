"""Tests for the Faster R-CNN wrapper.

The lightweight preprocessing test runs anywhere torch is installed. The full
inference test downloads ~160 MB of weights, so it only runs when
``RUN_MODEL_TESTS=1`` is set (CI keeps it off to avoid network downloads).
"""

import os

import numpy as np
import pytest

pytest.importorskip("torch")  # skip the whole module if torch is unavailable

import model_loader as ml  # noqa: E402

_RUN_MODEL = os.environ.get("RUN_MODEL_TESTS") == "1"


def test_preprocess_produces_chw_unit_tensor() -> None:
    # No weights needed: only the BGR->RGB->tensor conversion is exercised.
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    tensor = ml.preprocess(frame)
    assert tuple(tensor.shape) == (3, 48, 64)
    assert float(tensor.max()) <= 1.0
    assert float(tensor.min()) >= 0.0


@pytest.mark.skipif(
    not _RUN_MODEL, reason="set RUN_MODEL_TESTS=1 to download weights and run"
)
def test_detect_on_blank_frame_is_empty() -> None:
    model, device = ml.load_model()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    assert ml.detect(model, device, frame) == []
