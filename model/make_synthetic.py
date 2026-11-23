"""Synthetic training crops: every symbol drawn in many fonts, then put through a fake camera.

  python -m model.make_synthetic                    # 400 crops per class -> data/synthetic/train.npz
  python -m model.make_synthetic --per-class 100    # quick set for testing
  python -m model.make_synthetic --list-fonts       # which fonts were found

Step 1, the glyph bank (cached in data/synthetic/glyph_bank.pkl): each member
of symbols.csv is drawn large in every font that has it: matplotlib mathtext
font sets ("m:" and "r:" renders) and font files ("t:" renders) found on this
machine: Latin Modern and TeX Gyre from the TeX installation, DejaVu and STIX
from matplotlib, common system fonts, and anything in data/fonts/. Latin
Modern / Computer Modern count three times, because LaTeX prints with them.

Step 2, the fake camera: a glyph is scaled so the line's capital height is
18-32 px (30 % of the time drawn at script size), slightly rotated, sheared
and thickened/thinned, printed with random paper and ink levels, uneven light,
blur and noise, then binarized and cropped by the same code as live frames.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.preprocess import binarize, crop_to_uint8, normalize_crop
from pipeline.segment import find_blobs, merge_parts
from pipeline.symbols import load_symbols

BANK_PX = 96              # font size used for the bank
MATH_FONTSETS = {"cm": 3, "stix": 1, "stixsans": 1, "dejavusans": 1, "dejavuserif": 1}

# (file name, weight). Missing files are skipped; the first match wins.
FONT_FILES = [
    ("lmroman10-regular.otf", 3), ("lmroman10-italic.otf", 3), ("lmroman10-bold.otf", 1),
    ("lmsans10-regular.otf", 1), ("lmmono10-regular.otf", 1), ("latinmodern-math.otf", 3),
    ("texgyretermes-regular.otf", 1), ("texgyretermes-italic.otf", 1),
    ("texgyreheros-regular.otf", 1), ("texgyrepagella-regular.otf", 1),
    ("texgyrepagella-italic.otf", 1), ("texgyreschola-regular.otf", 1),
    ("texgyrebonum-regular.otf", 1), ("texgyretermes-math.otf", 1), ("texgyrepagella-math.otf", 1),
    ("texgyredejavu-math.otf", 1), ("DejaVuMathTeXGyre.ttf", 1), ("STIXTwoMath-Regular.otf", 1),
    ("STIXTwoMath.otf", 1), ("STIXTwoText-Regular.otf", 1), ("LibertinusMath-Regular.otf", 1),
    ("LibertinusSerif-Regular.otf", 1),
    ("DejaVuSans.ttf", 1), ("DejaVuSans-Oblique.ttf", 1), ("DejaVuSans-Bold.ttf", 1),
    ("DejaVuSerif.ttf", 1), ("DejaVuSerif-Italic.ttf", 1), ("DejaVuSansMono.ttf", 1),
    ("STIXGeneral.ttf", 1), ("STIXGeneralItalic.ttf", 1),
    ("LiberationSerif-Regular.ttf", 1), ("LiberationSerif-Italic.ttf", 1),
    ("LiberationSans-Regular.ttf", 1), ("FreeSerif.ttf", 1), ("FreeSerifItalic.ttf", 1),
    ("FreeSans.ttf", 1),
    ("Times New Roman.ttf", 1), ("times.ttf", 1), ("Arial.ttf", 1), ("arial.ttf", 1),
    ("Georgia.ttf", 1), ("georgia.ttf", 1), ("Helvetica.ttc", 1), ("cambria.ttc", 1),
    ("Cambria Math.ttf", 1), ("calibri.ttf", 1), ("Calibri.ttf", 1),
]
FONT_DIRS = [
    "/usr/share/fonts", "/usr/local/share/fonts", "~/.fonts", "~/.local/share/fonts",
    "/Library/Fonts", "/System/Library/Fonts", "~/Library/Fonts", "C:/Windows/Fonts",
]


# ------------------------------------------------------------------ fonts ---
def _index_dirs(dirs) -> dict[str, str]:
    index: dict[str, str] = {}
    for d in dirs:
        d = Path(os.path.expanduser(str(d)))
        if not d.is_dir():
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".ttf", ".otf", ".ttc")):
                    index.setdefault(f.lower(), os.path.join(root, f))
    return index


def _kpsewhich(name: str) -> str | None:
    exe = shutil.which("kpsewhich")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, name], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None
    return out or None


def find_fonts() -> list[tuple[str, int]]:
    import matplotlib
    mpl_dir = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    index = _index_dirs([mpl_dir] + FONT_DIRS)
    found, seen = [], set()
    for name, weight in FONT_FILES:
        path = index.get(name.lower()) or (_kpsewhich(name) if name.endswith(".otf") else None)
        if path and os.path.basename(path).lower() not in seen:
            seen.add(os.path.basename(path).lower())
            found.append((path, weight))
    extra = Path(config.FONTS_DIR)
    if extra.is_dir():
        for p in sorted(extra.iterdir()):
            if p.suffix.lower() in (".ttf", ".otf", ".ttc") and p.name.lower() not in seen:
                found.append((str(p), 2))
    return found


# --------------------------------------------------------------- drawing ---
def _tight(img: np.ndarray, pad: int = 2):
    ys, xs = np.nonzero(img > 40)
    if len(ys) == 0:
        return None
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + 1 + pad)
    return np.ascontiguousarray(img[y0:y1, x0:x1])


def _ink_height(img) -> float | None:
    ys, _ = np.nonzero(img > 128)
    return float(ys.max() - ys.min() + 1) if len(ys) else None


_PARSER = None


def draw_mathtext(expr: str, fontset: str, size_pt: float = 70) -> np.ndarray | None:
    global _PARSER
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import MathTextParser
    if _PARSER is None:
        _PARSER = MathTextParser("agg")
    try:
        r = _PARSER.parse(f"${expr}$", dpi=100, prop=FontProperties(size=size_pt, math_fontfamily=fontset))
    except Exception:
        return None
    img = np.asarray(r.image, dtype=np.uint8)
    return _tight(img)


def keep_largest(img: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((img > 128).astype(np.uint8), connectivity=8)
    if n <= 2:
        return img
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_WIDTH] * stats[1:, cv2.CC_STAT_HEIGHT]))
    keep = cv2.dilate((labels == big).astype(np.uint8), np.ones((3, 3), np.uint8))
    return _tight(img * keep)


class FontRenderer:
    def __init__(self, path: str):
        from PIL import ImageFont
        self.path = path
        self.font = ImageFont.truetype(path, BANK_PX)
        try:
            from fontTools.ttLib import TTFont
            tt = TTFont(path, lazy=True, fontNumber=0)
            self.cmap = set(tt.getBestCmap() or {})
        except Exception:
            self.cmap = None

    def covers(self, text: str) -> bool:
        return self.cmap is None or all(ord(ch) in self.cmap for ch in text)

    def draw(self, text: str) -> np.ndarray | None:
        from PIL import Image, ImageDraw
        if not self.covers(text):
            return None
        w = BANK_PX * (2 + 2 * len(text))
        im = Image.new("L", (w, BANK_PX * 3), 0)
        ImageDraw.Draw(im).text((BANK_PX // 2, BANK_PX // 2), text, font=self.font, fill=255)
        return _tight(np.asarray(im))


def build_bank(table, verbose=True) -> list[dict]:
    """Every member in every font that has it, drawn large, ink = 255."""
    bank = []
    fonts = [(FontRenderer(p), w) for p, w in find_fonts()]
    fonts = [(f, w) for f, w in fonts if f.covers("H")]
    caps = {}
    for fs in MATH_FONTSETS:
        ref = draw_mathtext(r"\mathrm{H}", fs)
        caps["m:" + fs] = _ink_height(ref) if ref is not None else None
    for f, _ in fonts:
        ref = f.draw("H")
        caps[f.path] = _ink_height(ref) if ref is not None else None
    for m in table.member_list:
        for spec in m.render:
            kind, body = spec[:1], spec[2:]
            if kind in ("m", "r"):
                for fs, weight in MATH_FONTSETS.items():
                    img = draw_mathtext(body, fs)
                    if img is not None and kind == "r":
                        img = keep_largest(img)
                    if img is not None and caps["m:" + fs]:
                        bank.append({"member": m.name, "source": f"mathtext:{fs}", "img": img,
                                     "cap": caps["m:" + fs], "weight": weight})
            elif kind == "t":
                for f, weight in fonts:
                    img = f.draw(body)
                    if img is not None and caps.get(f.path):
                        bank.append({"member": m.name, "source": os.path.basename(f.path), "img": img,
                                     "cap": caps[f.path], "weight": weight})
    if verbose:
        per = {}
        for b in bank:
            per[b["member"]] = per.get(b["member"], 0) + 1
        thin = sorted(per.items(), key=lambda kv: kv[1])[:8]
        missing = [m.name for m in table.member_list if m.name not in per]
        print(f"glyph bank: {len(bank)} drawings from {len(fonts)} font files + {len(MATH_FONTSETS)} "
              f"mathtext sets; fewest: {thin}; missing: {missing}")
    return bank


def member_stats(bank) -> dict:
    """Median height and width of each member, in capital heights."""
    acc = {}
    for b in bank:
        ys, xs = np.nonzero(b["img"] > 128)
        if len(ys) == 0:
            continue
        acc.setdefault(b["member"], []).append(((np.ptp(ys) + 1) / b["cap"], (np.ptp(xs) + 1) / b["cap"]))
    return {m: {"h": float(np.median([v[0] for v in vals])), "w": float(np.median([v[1] for v in vals]))}
            for m, vals in acc.items()}


# ------------------------------------------------------------ fake camera ---
SCALE_RANGE = (18, 32)    # line capital height in camera pixels (the rig gives about 24)


def fake_camera(glyph: np.ndarray, cap0: float, rng: np.random.Generator, script_p: float = 0.3,
                scale_range=SCALE_RANGE):
    """High-res glyph (ink = 255) -> (32x32 crop as uint8, merged_ok) or None."""
    S = rng.uniform(*scale_range)                             # line capital height, px
    if rng.random() < script_p:
        size = S * rng.uniform(0.62, 0.8)
    else:
        size = S * rng.uniform(0.93, 1.07)
    f = size / cap0
    g = glyph.astype(np.float32) / 255.0
    k = int(rng.choice([-1, 0, 0, 0, 1, 2]))
    if k:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        g = cv2.dilate(g, ker, iterations=k) if k > 0 else cv2.erode(g, ker, iterations=-k)
        if g.max() < 0.5:
            return None
    sigma = 0.45 / f
    if sigma > 0.6:
        g = cv2.GaussianBlur(g, (0, 0), sigma)
    h, w = g.shape
    angle = float(np.clip(rng.normal(0, 1.5), -4, 4))
    shear = float(np.clip(rng.normal(0, 0.06), -0.15, 0.15))
    margin = int(1.3 * S) + 4
    out_w = int(np.ceil(w * f * 1.2)) + 2 * margin
    out_h = int(np.ceil(h * f * 1.2)) + 2 * margin
    a = np.deg2rad(angle)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    Sh = np.array([[1.0, -shear], [0.0, 1.0]])
    A = f * R @ Sh
    c_in = np.array([w / 2.0, h / 2.0])
    c_out = np.array([out_w / 2.0, out_h / 2.0]) + rng.uniform(-1, 1, 2)
    M = np.hstack([A, (c_out - A @ c_in)[:, None]])
    ink = cv2.warpAffine(g, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderValue=0)
    if ink.max() < 0.3:
        return None
    paper = rng.uniform(140, 245)
    dark = rng.uniform(0, min(110, paper - 60))
    img = paper - (paper - dark) * np.clip(ink, 0, 1)
    yy, xx = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    gx, gy = rng.uniform(-0.25, 0.25, 2)
    img *= 1 + gx * (xx / out_w - 0.5) + gy * (yy / out_h - 0.5)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    img += rng.normal(0, rng.uniform(1, 7), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    binary = binarize(img)
    labels, blobs, _ = find_blobs(binary)
    foot = cv2.dilate((ink > 0.25).astype(np.uint8), np.ones((5, 5), np.uint8))
    keep = [b for b in blobs if foot[b.box.y0:b.box.y1, b.box.x0:b.box.x1].any()]
    if not keep:
        return None
    merged_ok = len(merge_parts(keep, S)) == 1
    ids = [i for b in keep for i in b.labels]
    x0 = min(b.box.x0 for b in keep)
    y0 = min(b.box.y0 for b in keep)
    x1 = max(b.box.x1 for b in keep)
    y1 = max(b.box.y1 for b in keep)
    mask = np.isin(labels[y0:y1, x0:x1], ids)
    crop = normalize_crop(mask, S * rng.uniform(0.88, 1.12))
    return crop_to_uint8(crop), merged_ok


# ----------------------------------------------------------- generation ---
_BANK = None
_BY_MEMBER = None


def _init_worker(bank_path: str):
    global _BANK, _BY_MEMBER
    with open(bank_path, "rb") as fh:
        _BANK = pickle.load(fh)
    _BY_MEMBER = {}
    for b in _BANK:
        _BY_MEMBER.setdefault(b["member"], []).append(b)
    cv2.setNumThreads(1)


def _make_class(job):
    cls_index, members, n, seed = job
    rng = np.random.default_rng(seed)
    members = [m for m in members if m in _BY_MEMBER]
    X, M, merge_stats = [], [], {}
    if not members:
        return cls_index, np.zeros((0, config.CROP_SIZE, config.CROP_SIZE), np.uint8), [], merge_stats
    # every member of a look-alike class gets at least half a class's worth of crops
    per_member = max(int(np.ceil(n / len(members))), n // 2)
    plan = [m for m in members for _ in range(per_member)]
    rng.shuffle(plan)
    target = len(plan)
    tries = 0
    while len(X) < target and tries < target * 5:
        m = plan[len(X) % len(plan)] if len(X) < len(plan) else members[0]
        tries += 1
        entries = _BY_MEMBER[m]
        w = np.array([e["weight"] for e in entries], float)
        e = entries[int(rng.choice(len(entries), p=w / w.sum()))]
        r = fake_camera(e["img"], e["cap"], rng)
        if r is None:
            continue
        crop, ok = r
        X.append(crop)
        M.append(m)
        s = merge_stats.setdefault(m, [0, 0])
        s[0] += 1
        s[1] += int(ok)
    return cls_index, np.stack(X) if X else np.zeros((0, 32, 32), np.uint8), M, merge_stats


def save_preview(X, y, classes, path: Path, per_class: int = 10):
    """One row per class: its name and the first few crops (for a quick visual check)."""
    rows = []
    cell = config.CROP_SIZE + 2
    for ci, name in enumerate(classes):
        idx = np.nonzero(y == ci)[0][:per_class]
        row = np.full((cell, 110 + cell * per_class), 255, np.uint8)
        cv2.putText(row, name[:14], (2, cell - 10), cv2.FONT_HERSHEY_PLAIN, 0.9, 0, 1)
        for k, i in enumerate(idx):
            row[1:1 + config.CROP_SIZE, 110 + k * cell:110 + k * cell + config.CROP_SIZE] = 255 - X[i]
        rows.append(row)
    pages = [rows[i:i + 40] for i in range(0, len(rows), 40)]
    for p, page in enumerate(pages):
        cv2.imwrite(str(path.with_name(f"{path.stem}_{p + 1}.png")), np.vstack(page))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=400)
    ap.add_argument("--out", type=Path, default=Path(config.SYNTH_DIR))
    ap.add_argument("--name", default="train", help="output file name (without .npz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--rebuild-bank", action="store_true")
    ap.add_argument("--list-fonts", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args(argv)

    if args.list_fonts:
        for p, w in find_fonts():
            print(f"weight {w}  {p}")
        return 0
    table = load_symbols()
    args.out.mkdir(parents=True, exist_ok=True)
    bank_path = args.out / "glyph_bank.pkl"
    t0 = time.time()
    if args.rebuild_bank or not bank_path.exists():
        bank = build_bank(table)
        with open(bank_path, "wb") as fh:
            pickle.dump(bank, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (args.out / "member_stats.json").write_text(json.dumps(member_stats(bank), indent=1))
        print(f"bank built in {time.time() - t0:.0f} s -> {bank_path}")

    jobs = [(ci, [m.name for m in table.members_of(c)], args.per_class, args.seed * 100003 + ci)
            for ci, c in enumerate(table.classes)]
    results = []
    t1 = time.time()
    if args.workers > 1:
        # "spawn": forking a process that already runs OpenCV threads can hang
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=_init_worker,
                                 initargs=(str(bank_path),)) as ex:
            results = list(ex.map(_make_class, jobs))
    else:
        _init_worker(str(bank_path))
        results = [_make_class(j) for j in jobs]
    X = np.concatenate([r[1] for r in results])
    y = np.concatenate([np.full(len(r[1]), r[0], np.int16) for r in results])
    members = [m for r in results for m in r[2]]
    member_idx = np.array([table.member_index[m] for m in members], np.int16)
    out = args.out / f"{args.name}.npz"
    np.savez_compressed(out, X=X, y=y, member=member_idx, classes=np.array(table.classes),
                        members=np.array([m.name for m in table.member_list]))
    print(f"{len(X)} crops for {len(table.classes)} classes in {time.time() - t1:.0f} s -> {out}")

    merge = {}
    for r in results:
        for m, (n, ok) in r[3].items():
            merge[m] = (n, ok)
    bad = sorted(((ok / n, m) for m, (n, ok) in merge.items() if n and ok / n < 0.9))
    if bad:
        print("segmentation check - members often split into pieces (tune MERGE_* in config.py):")
        for frac, m in bad[:15]:
            print(f"  {m:16s} kept whole in {frac:.0%} of drawings")
    short = [table.classes[r[0]] for r in results if len(r[1]) < args.per_class]
    if short:
        print(f"classes with fewer crops than asked: {short}")
    if not args.no_preview:
        save_preview(X, y, table.classes, args.out / "preview.png")
        print(f"preview pages: {args.out}/preview_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
