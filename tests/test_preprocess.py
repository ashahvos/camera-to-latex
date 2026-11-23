import numpy as np

from pipeline.preprocess import binarize, normalize_crop, orient, place_glyph


def test_binarize_finds_dark_ink():
    img = np.full((60, 80), 200, np.uint8)
    img[20:40, 30:34] = 40
    b = binarize(img)
    assert b[30, 31] == 1
    assert b[5, 5] == 0
    assert 60 <= b.sum() <= 140


def test_binarize_handles_uneven_light():
    x = np.linspace(120, 230, 120)
    img = np.tile(x, (60, 1)).astype(np.uint8)
    img[25:35, 10:13] = (img[25:35, 10:13] * 0.3).astype(np.uint8)
    img[25:35, 100:103] = (img[25:35, 100:103] * 0.3).astype(np.uint8)
    b = binarize(img)
    assert b[30, 11] == 1 and b[30, 101] == 1
    assert b[:, 40:90].sum() == 0


def test_crop_keeps_relative_size():
    cap = np.ones((20, 12), bool)
    dot = np.ones((3, 3), bool)
    a = normalize_crop(cap, scale=20)
    b = normalize_crop(dot, scale=20)
    assert a.shape == b.shape == (32, 32)
    assert a.sum() > 10 * b.sum()
    ys, xs = np.nonzero(b > 0.2)
    assert abs(ys.mean() - 15.5) < 2 and abs(xs.mean() - 15.5) < 2


def test_big_symbol_still_fits():
    big = np.ones((60, 20), bool)
    c = normalize_crop(big, scale=20)
    assert c[0].sum() > 0 or c[1].sum() > 0          # reaches the top: window grew to fit


def test_orient_flips_and_rotates():
    img = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(orient(img, flip_h=True, flip_v=False, rotate_deg=0, roi=None), img[:, ::-1])
    assert orient(img, flip_h=False, flip_v=False, rotate_deg=90, roi=None).shape == (4, 3)


def test_place_glyph_uses_height_ratio():
    glyph = np.ones((28, 20), bool)
    small = place_glyph(glyph, 0.5)
    tall = place_glyph(glyph, 1.0)
    assert tall.sum() > 2 * small.sum()
