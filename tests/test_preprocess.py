"""Tests for host/draw.py preprocess(). Run with: pytest tests/test_preprocess.py"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "host"))
from draw import preprocess  # noqa: E402


def blank():
    """An empty canvas as DrawingApp creates it."""
    return Image.new("L", (280, 280), 0)


def ones_grid(x):
    """The 1-pixels of a (784,) output as (rows, cols) index arrays."""
    return np.nonzero(x.reshape(28, 28))


def draw_seven(dx, dy):
    """A "7" drawn with the app's brush width, shifted by (dx, dy)."""
    img = blank()
    pen = ImageDraw.Draw(img)
    points = [(40 + dx, 40 + dy), (120 + dx, 40 + dy), (70 + dx, 160 + dy)]
    pen.line(points, fill=255, width=20, joint="curve")
    for x, y in points:
        pen.ellipse([x - 10, y - 10, x + 10, y + 10], fill=255)
    return img


def test_empty_image_returns_none():
    assert preprocess(blank()) is None


def test_off_center_rectangle_is_boxed_and_centered():
    img = blank()
    ImageDraw.Draw(img).rectangle([20, 30, 119, 89], fill=255)

    x = preprocess(img)

    assert x.shape == (784,)
    assert x.dtype == np.uint8
    assert set(np.unique(x)) <= {0, 1}
    rows, cols = ones_grid(x)
    assert rows.max() - rows.min() + 1 <= 20
    assert cols.max() - cols.min() + 1 <= 20
    assert abs(rows.mean() - 14) <= 1
    assert abs(cols.mean() - 14) <= 1


def test_tall_thin_stroke_is_20_pixels_high():
    img = blank()
    ImageDraw.Draw(img).rectangle([135, 40, 145, 239], fill=255)

    rows, _ = ones_grid(preprocess(img))

    assert rows.max() - rows.min() + 1 == 20


def test_output_does_not_depend_on_position():
    a = preprocess(draw_seven(0, 0))
    b = preprocess(draw_seven(110, 90))

    assert a.sum() > 0
    np.testing.assert_array_equal(a, b)
