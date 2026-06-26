"""Smoke tests for the Day 1 preprocessing + visualization utilities.

These run a synthetic image through every preprocessing op and assert the output
shapes, with no model or external assets required.
"""

import draw_shapes as ds
import numpy as np
import resize_crop as rc


def _img(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_resize_fixed_shape() -> None:
    assert rc.resize_fixed(_img(), 320, 240).shape == (240, 320, 3)


def test_resize_by_scale_halves() -> None:
    assert rc.resize_by_scale(_img(480, 640), 0.5).shape == (240, 320, 3)


def test_resize_keep_aspect_preserves_ratio() -> None:
    assert rc.resize_keep_aspect(_img(480, 640), 320).shape == (240, 320, 3)


def test_crop_roi_basic() -> None:
    assert rc.crop_roi(_img(480, 640), 100, 50, 300, 250).shape == (200, 200, 3)


def test_crop_roi_clamps_out_of_bounds() -> None:
    # Coordinates outside the frame are clipped, never raising.
    assert rc.crop_roi(_img(480, 640), -50, -50, 1000, 1000).shape == (480, 640, 3)


def test_letterbox_is_square_uint8() -> None:
    out = rc.letterbox(_img(480, 640), 640)
    assert out.shape == (640, 640, 3)
    assert out.dtype == np.uint8


def test_letterbox_pads_and_centers() -> None:
    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    out = rc.letterbox(img, 640)
    assert out[0].sum() == 0  # top padding row is black
    assert out[-1].sum() == 0  # bottom padding row is black
    assert out[320].sum() > 0  # centre row carries the content


def test_draw_helpers_render_something() -> None:
    img = ds.make_blank_canvas(200, 200)
    ds.draw_bounding_box(img, 10, 20, 80, 120, "person", 0.9)
    ds.draw_center_point(img, 10, 20, 80, 120)
    ds.draw_roi_zone(img, (5, 5, 150, 150))
    ds.draw_info_bar(img, 3, fps=12.5)
    assert img.shape == (200, 200, 3)
    assert img.sum() > 0
