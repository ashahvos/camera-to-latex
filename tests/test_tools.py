"""Sheet cropping and dataset import on tiny synthetic inputs."""
import csv

import cv2
import numpy as np

from pipeline.preprocess import load_crop_png
from pipeline.symbols import load_symbols
from tools.crop_sheet import crop_photo
from tools.import_datasets import Collector, import_chars74k, import_tmnist

LAYOUT = {"cols": 2, "rows": 2, "pitch_x_cm": 2.0, "pitch_y_cm": 2.0, "mark_cm": 0.5, "mark_gap_cm": 0.25,
          "blocks": []}
BLOCK = {"id": "Z1", "cells": [
    {"row": 0, "col": 0, "member": "digit_1", "class": "digit_1", "style": "text"},
    {"row": 0, "col": 1, "member": "digit_2", "class": "digit_2", "style": "text"},
    {"row": 1, "col": 0, "member": "upper_A", "class": "upper_A", "style": "text"},
    {"row": 1, "col": 1, "member": "upper_B", "class": "upper_B", "style": "text"}]}
WIDTHS = {(0, 0): 4, (0, 1): 8, (1, 0): 12, (1, 1): 16}     # px, so each cell's crop is recognisable


def draw_block(rotate180=False):
    ppc = 20.0                                   # pixels per cm
    img = np.full((240, 320), 220, np.uint8)
    ox, oy = 110, 70                             # grid top-left in pixels
    m, g = 0.5 * ppc, 0.25 * ppc
    W, H = 2 * 2.0 * ppc, 2 * 2.0 * ppc
    tl = (int(ox - g - m), int(oy - g - m))
    cv2.rectangle(img, tl, (int(tl[0] + m), int(tl[1] + m)), 20, 2)               # hollow
    for x, y in ((ox + W + g, oy - g - m), (ox + W + g, oy + H + g), (ox - g - m, oy + H + g)):
        cv2.rectangle(img, (int(x), int(y)), (int(x + m), int(y + m)), 20, -1)
    for (r, c), w in WIDTHS.items():
        cx, cy = ox + (c + 0.5) * 2.0 * ppc, oy + (r + 0.5) * 2.0 * ppc
        cv2.rectangle(img, (int(cx - w / 2), int(cy - 10)), (int(cx + w / 2), int(cy + 10)), 20, -1)
    if rotate180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def test_crop_sheet_assigns_cells(tmp_path):
    table = load_symbols()
    for rot in (False, True):
        photo = tmp_path / f"Z1_{int(rot) + 1}.png"
        cv2.imwrite(str(photo), draw_block(rot))
        saved, empty, scale = crop_photo(photo, BLOCK, LAYOUT, table, tmp_path / "crops", None)
        assert saved == 4 and not empty
        assert 18 <= scale <= 23
    ink = {}
    for f in (tmp_path / "crops").rglob("*.png"):
        ink.setdefault(f.parent.name, []).append(load_crop_png(f).sum())
    order = [np.mean(ink[c]) for c in ("digit_1", "digit_2", "upper_A", "upper_B")]
    assert order == sorted(order)                # widths 4 < 8 < 12 < 16 px kept their labels


def test_import_tmnist_and_chars74k(tmp_path):
    table = load_symbols()
    root = tmp_path / "tmnist94"
    root.mkdir()
    with open(root / "94_character_TMNIST.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["names", "labels"] + [str(i) for i in range(1, 785)])
        for ch in ("A", "a", "~"):
            px = np.zeros((28, 28), int)
            px[4:24, 8:20] = 255
            w.writerow(["SomeFont", ch] + px.ravel().tolist())
    col = Collector(table, max_per_class=10)
    import_tmnist(root, col)
    assert len(col.X) == 3
    assert {table.classes[i] for i in col.y} == {"upper_A", "lower_a", "tilde"}

    croot = tmp_path / "chars74k_fnt" / "English" / "Fnt" / "Sample011"
    croot.mkdir(parents=True)
    img = np.full((128, 128), 255, np.uint8)
    img[30:100, 40:90] = 0
    cv2.imwrite(str(croot / "img011-00001.png"), img)
    col = Collector(table, max_per_class=10)
    import_chars74k(tmp_path / "chars74k_fnt", col)
    assert [table.classes[i] for i in col.y] == ["upper_A"]
