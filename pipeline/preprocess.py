"""Image clean-up shared by every data source.

Live frames, sheet photos, synthetic renders and external datasets all go
through binarize() and normalize_crop(), so the classifier sees crops made
the same way in training and in use.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config


def to_gray(img: np.ndarray) -> np.ndarray:
    """Any image -> 2-D uint8 grayscale."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def orient(gray: np.ndarray, flip_h=None, flip_v=None, rotate_deg=None, roi=None) -> np.ndarray:
    """Apply the fixed flips, rotation and region of interest from config.py."""
    flip_h = config.FLIP_H if flip_h is None else flip_h
    flip_v = config.FLIP_V if flip_v is None else flip_v
    rotate_deg = config.ROTATE_DEG if rotate_deg is None else rotate_deg
    roi = config.ROI if roi is None else roi
    out = gray
    if flip_h:
        out = cv2.flip(out, 1)
    if flip_v:
        out = cv2.flip(out, 0)
    rot = int(rotate_deg) % 360
    if rot == 90:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif rot == 270:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rot != 0:
        raise ValueError("ROTATE_DEG must be 0, 90, 180 or 270")
    if roi:
        x, y, w, h = roi
        out = out[y:y + h, x:x + w]
    return np.ascontiguousarray(out)


def _odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def binarize(gray: np.ndarray, blur=None, block=None, c=None,
             open_k=None, close_k=None, weak_c="config") -> np.ndarray:
    """Dark ink on light paper -> uint8 mask with ink = 1.

    An adaptive threshold copes with uneven light across the page: each pixel
    is compared with a Gaussian-weighted average of its neighbourhood, and is
    ink if it is more than c grey levels darker. With hysteresis (weak_c set),
    fainter pixels (more than weak_c darker) are kept too, but only where they
    touch confident ink: thin strokes of Computer Modern letters stay joined
    to their letter without lighter specks of paper turning into ink.
    """
    blur = config.BLUR_KSIZE if blur is None else blur
    block = config.THRESH_BLOCK if block is None else block
    c = config.THRESH_C if c is None else c
    weak_c = config.THRESH_WEAK_C if weak_c == "config" else weak_c
    open_k = config.OPEN_KSIZE if open_k is None else open_k
    close_k = config.CLOSE_KSIZE if close_k is None else close_k

    g = to_gray(gray)
    if blur and blur > 1:
        k = _odd(blur)
        g = cv2.GaussianBlur(g, (k, k), 0)
    block = max(3, _odd(block))
    th = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, c)
    if weak_c is not None and weak_c < c:
        weak = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block, weak_c)
        n, labels = cv2.connectedComponents(weak, connectivity=8)
        keep = np.zeros(n, bool)
        keep[np.unique(labels[th > 0])] = True
        keep[0] = False
        th = keep[labels].astype(np.uint8)
    if open_k and open_k > 1:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    if close_k and close_k > 1:
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    return th


def normalize_crop(mask: np.ndarray, scale: float, size=None, window_scale=None) -> np.ndarray:
    """Turn one symbol's ink mask into a size x size float crop in [0, 1].

    The window is a square of side max(WINDOW_SCALE * scale, longest side + 4)
    centred on the symbol, where scale is the line's capital height in pixels.
    Because the window follows the line and not the symbol, a period stays a
    small dot and a capital fills most of the crop: size is kept as a clue.
    """
    size = size or config.CROP_SIZE
    window_scale = window_scale or config.WINDOW_SCALE
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim != 2 or m.size == 0:
        return np.zeros((size, size), np.float32)
    h, w = m.shape
    side = int(np.ceil(max(window_scale * float(scale), max(h, w) + 4)))
    canvas = np.zeros((side, side), np.float32)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = m
    interp = cv2.INTER_AREA if side >= size else cv2.INTER_LINEAR
    out = cv2.resize(canvas, (size, size), interpolation=interp)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def place_glyph(mask: np.ndarray, height_ratio: float, ref_scale: float = 24.0,
                size=None, window_scale=None) -> np.ndarray:
    """Crop for a glyph from a 'fill the box' dataset (EMNIST-style).

    The glyph is resized so its height is height_ratio * ref_scale pixels, as
    if it sat in a camera line whose capital height is ref_scale, then it goes
    through normalize_crop like any live symbol.
    """
    m = np.asarray(mask, dtype=np.float32)
    ys, xs = np.nonzero(m > 0.5)
    if len(ys) == 0:
        return np.zeros((size or config.CROP_SIZE,) * 2, np.float32)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    f = max(1.0, height_ratio * ref_scale) / h
    nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    small = cv2.resize(m, (nw, nh), interpolation=interp) >= 0.5
    return normalize_crop(small, ref_scale, size=size, window_scale=window_scale)


def crop_to_uint8(crop: np.ndarray) -> np.ndarray:
    """float crop (ink = 1) -> uint8 (ink = 255), the format of the .npz data files."""
    return np.clip(np.round(crop * 255.0), 0, 255).astype(np.uint8)


def crop_from_uint8(img: np.ndarray) -> np.ndarray:
    return to_gray(img).astype(np.float32) / 255.0


def save_crop_png(path, crop: np.ndarray) -> None:
    """Crops are stored as dark ink on white so the folders are easy to browse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), 255 - crop_to_uint8(crop))


def load_crop_png(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"cannot read {path}")
    if img.shape != (config.CROP_SIZE, config.CROP_SIZE):
        img = cv2.resize(img, (config.CROP_SIZE, config.CROP_SIZE), interpolation=cv2.INTER_AREA)
    return (255 - img).astype(np.float32) / 255.0
