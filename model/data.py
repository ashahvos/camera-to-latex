"""Load training crops from every source into one (X, y) form.

  X: uint8 array (N, 32, 32), ink = 255     y: int class index (order of symbols.csv)

Sources
  data/synthetic/*.npz      model/make_synthetic.py
  data/external/*.npz       tools/import_datasets.py
  data/crops_real/<class>/  tools/crop_sheet.py (PNG, dark ink on white);
                            file names <member>__<photo>__r<row>c<col>.png
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

import config
from pipeline.preprocess import crop_to_uint8, load_crop_png
from pipeline.symbols import SymbolTable


def load_npz(path: Path, table: SymbolTable):
    """(X, y) with labels remapped to the current symbols.csv; unknown classes dropped."""
    d = np.load(path, allow_pickle=False)
    X, y = d["X"], d["y"].astype(int)
    saved = [str(c) for c in d["classes"]] if "classes" in d.files else table.classes
    remap = np.array([table.class_index.get(c, -1) for c in saved])
    y = remap[y]
    keep = y >= 0
    return X[keep], y[keep]


def load_real_crops(root: Path, table: SymbolTable):
    """(X, y, photo ids) from data/crops_real/<class>/*.png."""
    X, y, groups = [], [], []
    root = Path(root)
    if not root.is_dir():
        return np.zeros((0, config.CROP_SIZE, config.CROP_SIZE), np.uint8), np.zeros(0, int), []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ci = table.class_index.get(cls_dir.name)
        if ci is None:
            print(f"skipping {cls_dir}: not a class in symbols.csv")
            continue
        for f in sorted(cls_dir.glob("*.png")):
            X.append(crop_to_uint8(load_crop_png(f)))
            y.append(ci)
            parts = f.stem.split("__")
            groups.append(parts[1] if len(parts) >= 3 else f.stem)
    if not X:
        return np.zeros((0, config.CROP_SIZE, config.CROP_SIZE), np.uint8), np.zeros(0, int), []
    return np.stack(X), np.array(y), groups


def holdout_mask(groups, fraction: float = 0.2) -> np.ndarray:
    """True for crops whose photo is held out (a stable choice: same photos every run)."""
    return np.array([(zlib.crc32(g.encode()) % 1000) < fraction * 1000 for g in groups], bool)
