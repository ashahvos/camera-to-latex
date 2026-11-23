"""Turn downloaded datasets into training crops (data/external/<name>.npz).

  python -m tools.import_datasets chars74k_fnt tmnist94
  python -m tools.import_datasets hasy --max-per-class 300
  python -m model.train --external            # then train with them

Each dataset's labels are mapped onto symbols.csv (unmapped classes are
skipped and listed). These datasets scale every glyph to fill its image,
which throws away size: an 'o' looks like an 'O' and a period like a ball. So
each glyph is resized to the height that member really has relative to a
capital (measured on the synthetic glyph bank, data/synthetic/member_stats.json)
and then cropped exactly like a live symbol. Members that are only dots or
lines (. , - _) are skipped: their shape alone says too little.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.classify import HEIGHT_RATIO
from pipeline.preprocess import crop_to_uint8, place_glyph
from pipeline.symbols import load_symbols

HASY_ALIASES = {
    "\\mathds{R}": "bbR", "\\mathds{N}": "bbN", "\\mathds{Z}": "bbZ", "\\mathds{Q}": "bbQ",
    "\\mathds{C}": "bbC", "\\varnothing": "emptyset", "\\sqrt{}": "sqrt", "\\le": "leq",
    "\\ge": "geq", "\\ne": "neq", "\\to": "rightarrow", "\\mid": "vert", "|": "vert",
    "\\|": "parallel", "\\lbrace": "lbrace", "\\rbrace": "rbrace", "\\epsilon": "epsilon",
    "\\varepsilon": "varepsilon", "\\phi": "phi", "\\varphi": "varphi", "\\with": None,
}
KAGGLE_MATH = {
    "!": "exclam", "(": "lparen", ")": "rparen", "+": "plus", ",": "comma", "-": "minus",
    "=": "equal", "[": "lbracket", "]": "rbracket", "{": "lbrace", "}": "rbrace",
    "Delta": "Delta", "alpha": "alpha", "beta": "beta", "gamma": "gamma", "lambda": "lambda",
    "mu": "mu", "phi": "varphi", "pi": "pi", "sigma": "sigma", "theta": "theta",
    "ascii_124": "vert", "div": "div", "exists": "exists", "forall": "forall",
    "forward_slash": "slash", "geq": "geq", "gt": "greater", "in": "in", "infty": "infty",
    "int": "int", "leq": "leq", "lt": "less", "neq": "neq", "pm": "pm", "prime": "prime",
    "rightarrow": "rightarrow", "sqrt": "sqrt", "sum": "sum", "times": "times",
}


def height_ratios(table) -> dict:
    path = Path(config.SYNTH_DIR) / "member_stats.json"
    stats = json.loads(path.read_text()) if path.exists() else {}
    if not stats:
        print("note: data/synthetic/member_stats.json missing (run make_synthetic first); "
              "using typical heights per height type")
    return {m.name: stats.get(m.name, {}).get("h", HEIGHT_RATIO[m.height]) for m in table.member_list}


class Collector:
    def __init__(self, table, max_per_class: int, seed: int = 0):
        self.table = table
        self.ratios = height_ratios(table)
        self.max = max_per_class
        self.X, self.y = [], []
        self.count: dict[str, int] = {}
        self.skipped: dict[str, int] = {}
        self.rng = np.random.default_rng(seed)

    def add(self, ink_mask: np.ndarray, member_name: str | None, label: str) -> None:
        m = self.table.members.get(member_name) if member_name else None
        if m is None:
            self.skipped[label] = self.skipped.get(label, 0) + 1
            return
        if m.height in ("dot", "line") or self.count.get(m.cls, 0) >= self.max:
            return
        if ink_mask.sum() < 4:
            return
        ref = float(self.rng.uniform(20, 30))          # pretend capital height, px
        crop = place_glyph(ink_mask, self.ratios[m.name] * self.rng.uniform(0.92, 1.08), ref_scale=ref)
        self.X.append(crop_to_uint8(crop))
        self.y.append(self.table.class_index[m.cls])
        self.count[m.cls] = self.count.get(m.cls, 0) + 1

    def save(self, name: str) -> None:
        out = Path(config.EXTERNAL_DIR)
        out.mkdir(parents=True, exist_ok=True)
        if not self.X:
            print(f"{name}: nothing imported")
            return
        np.savez_compressed(out / f"{name}.npz", X=np.stack(self.X), y=np.array(self.y, np.int16),
                            classes=np.array(self.table.classes))
        print(f"{name}: {len(self.X)} crops in {len(self.count)} classes -> {out / (name + '.npz')}")
        if self.skipped:
            top = sorted(self.skipped.items(), key=lambda kv: -kv[1])[:12]
            print(f"  labels not in symbols.csv (skipped): {', '.join(k for k, _ in top)}"
                  f"{' ...' if len(self.skipped) > 12 else ''}")


def _dark_ink(img: np.ndarray) -> np.ndarray:
    """Grayscale image with dark ink on light paper -> bool ink mask (Otsu)."""
    _, th = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return th.astype(bool)


def import_chars74k(root: Path, col: Collector) -> None:
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    cmap = col.table.char_map()
    dirs = sorted(root.rglob("Sample0[0-9][0-9]"))
    if not dirs:
        raise FileNotFoundError(f"no Sample0xx folders under {root}")
    for d in dirs:
        ch = chars[int(d.name[-3:]) - 1]
        files = sorted(d.glob("*.png"))
        col.rng.shuffle(files)
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                col.add(_dark_ink(img), cmap.get(ch).name if ch in cmap else None, ch)


def import_tmnist(root: Path, col: Collector) -> None:
    cmap = col.table.char_map()
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV file under {root}")
    with open(files[0], newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        li = header.index("labels") if "labels" in header else 1
        first_px = li + 1
        rows = list(reader)
    col.rng.shuffle(rows)
    for r in rows:
        ch = r[li]
        px = np.array(r[first_px:first_px + 784], dtype=np.float32).reshape(28, 28)
        if px.max() <= 1.0:
            px *= 255
        ink = px > 127 if px.mean() < 128 else px < 128      # handle either polarity
        col.add(ink, cmap[ch].name if ch in cmap else None, ch)


def import_hasy(root: Path, col: Collector) -> None:
    labels = list(root.rglob("hasy-data-labels.csv"))
    if not labels:
        raise FileNotFoundError(f"no hasy-data-labels.csv under {root}")
    base = labels[0].parent
    lmap = {k: v.name for k, v in col.table.latex_map().items()}
    with open(labels[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    col.rng.shuffle(rows)
    for r in rows:
        tex = r["latex"]
        member = HASY_ALIASES.get(tex, lmap.get(tex))
        if member is None and len(tex) == 1:
            cm = col.table.char_map().get(tex)
            member = cm.name if cm else None
        img = cv2.imread(str(base / r["path"]), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            col.add(_dark_ink(img), member, tex)


def import_kaggle_math(root: Path, col: Collector) -> None:
    cmap = col.table.char_map()
    dirs = [d for d in root.rglob("*") if d.is_dir() and any(d.glob("*.jpg"))]
    if not dirs:
        raise FileNotFoundError(f"no class folders with .jpg files under {root}")
    for d in sorted(dirs):
        name = d.name
        member = KAGGLE_MATH.get(name)
        if member is None and len(name) == 1:
            member = cmap[name].name if name in cmap else None
        files = sorted(d.glob("*.jpg"))
        col.rng.shuffle(files)
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                col.add(_dark_ink(img), member, name)


IMPORTERS = {"chars74k_fnt": import_chars74k, "tmnist94": import_tmnist,
             "hasy": import_hasy, "kaggle_math": import_kaggle_math}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", choices=sorted(IMPORTERS))
    ap.add_argument("--root", type=Path, default=Path(config.DOWNLOADS_DIR))
    ap.add_argument("--max-per-class", type=int, default=1000)
    args = ap.parse_args(argv)
    table = load_symbols()
    for name in args.names:
        col = Collector(table, args.max_per_class)
        try:
            IMPORTERS[name](args.root / name, col)
        except FileNotFoundError as exc:
            print(f"{name}: {exc} (run python -m tools.download_datasets {name} first)")
            continue
        col.save(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
