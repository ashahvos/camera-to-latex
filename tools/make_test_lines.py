"""The test set: printable sheets, photo names, and synthetic "photos" for testing without the board.

  python -m tools.make_test_lines --pdf       # test_set/test_lines.pdf (print it, 100 % scale)
  python -m tools.make_test_lines --names     # test_set/photo_names.txt for pipeline.capture --series
  python -m tools.make_test_lines --render    # test_set/synthetic/<id>.png (fake camera photos)

Every line of test_set/lines.tsv is typeset at 40 pt (capitals about 1 cm)
from its expected LaTeX, one line per 4.6 cm band, with its id small at the
right margin, far enough from the line to stay out of the camera's view.
Math lines are printed in display style: fractions get full-size numbers
and more space around their bars, and sums get their limits above and below.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

LINES_TSV = Path(config.TEST_SET_DIR) / "lines.tsv"
FONT_PT = 40
CAP_CM = 0.683 * FONT_PT * 2.54 / 72.27      # Latin Modern capital height at 40 pt (about 0.96 cm)

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{exscale}
"""


def read_lines(path: Path = LINES_TSV, split: str = "all") -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["id"] and not r["id"].startswith("#")]
    if split != "all":
        rows = [r for r in rows if r["split"] == split]
    return rows


def body(row) -> str:
    return row["latex"] if row["mode"] == "text" else r"$\displaystyle " + row["latex"] + "$"


def run_pdflatex(tex: str, workdir: Path, name: str) -> Path:
    (workdir / f"{name}.tex").write_text(tex)
    r = subprocess.run([config.PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
                       cwd=workdir, capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        raise RuntimeError("pdflatex failed:\n" + r.stdout[-2000:])
    return workdir / f"{name}.pdf"


def make_pdf(rows, out: Path) -> None:
    per_page = 5
    parts = [PREAMBLE, r"\usepackage[letterpaper,margin=1.5cm]{geometry}", r"\pagestyle{empty}",
             r"\setlength{\parindent}{0pt}", r"\newsavebox{\linebox}", r"\begin{document}"]
    for i, row in enumerate(rows):
        if i and i % per_page == 0:
            parts.append(r"\newpage")
        parts.append(r"\sbox{\linebox}{\fontsize{%d}{%d}\selectfont %s}" % (FONT_PT, FONT_PT * 1.2, body(row)))
        parts.append(r"\ifdim\wd\linebox>12.5cm\typeout{WIDE LINE %s}\fi" % row["id"])
        parts.append(r"\begin{minipage}[c][4.6cm][c]{\linewidth}\usebox{\linebox}\hfill"
                     r"{\footnotesize\ttfamily %s}\end{minipage}\par" % row["id"])
    parts.append(r"\end{document}")
    with tempfile.TemporaryDirectory() as tmp:
        pdf = run_pdflatex("\n".join(parts), Path(tmp), "test_lines")
        log = (Path(tmp) / "test_lines.log").read_text(errors="replace")
        wide = [ln.split()[-1] for ln in log.splitlines() if ln.startswith("WIDE LINE")]
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, out)
    print(f"{out}: {len(rows)} lines, {per_page} per page")
    if wide:
        print(f"warning: wider than 12.5 cm (may not fit the camera's view): {wide}")


def render_lines(rows, dpi: int):
    """Each line alone on a page -> list of grayscale images (ink dark), same order."""
    parts = [PREAMBLE, r"\usepackage[paperwidth=30cm,paperheight=7cm,margin=1cm]{geometry}",
             r"\pagestyle{empty}", r"\setlength{\parindent}{0pt}", r"\begin{document}"]
    for row in rows:
        parts.append(r"\vspace*{\fill}{\fontsize{%d}{%d}\selectfont %s}\vspace*{\fill}\newpage"
                     % (FONT_PT, FONT_PT * 1.2, body(row)))
    parts.append(r"\end{document}")
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm (poppler) is needed for --render")
    images = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pdf = run_pdflatex("\n".join(parts), tmp, "lines")
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", str(pdf), str(tmp / "p")], check=True)
        pages = sorted(tmp.glob("p-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
        for p in pages:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            ys, xs = np.nonzero(img < 128)
            images.append(img[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return images


def fake_photo(line: np.ndarray, px_per_cm: float, rng, render_px_per_cm: float, other=None):
    """A rendered line (ink dark, high resolution) -> 320 x 240 camera-like frame."""
    W, H = config.FRAME_W, config.FRAME_H
    f = px_per_cm / render_px_per_cm
    ink = (255 - line.astype(np.float32)) / 255.0
    big = np.zeros((H * 4, W * 4), np.float32)          # page around the line, in frame pixels x4

    def paste(img, cx, cy):
        im = cv2.resize(img, None, fx=4 * f, fy=4 * f, interpolation=cv2.INTER_AREA)
        h, w = im.shape
        x0, y0 = int(cx * 4 - w / 2), int(cy * 4 - h / 2)
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(big.shape[1], x0 + w), min(big.shape[0], y0 + h)
        if xb > xa and yb > ya:
            big[ya:yb, xa:xb] = np.maximum(big[ya:yb, xa:xb], im[ya - y0:yb - y0, xa - x0:xb - x0])

    cx, cy = W / 2 + rng.uniform(-10, 10), H / 2 + rng.uniform(-8, 8)
    paste(ink, cx, cy)
    if other is not None:                               # the neighbouring printed line, 4.6 cm away
        oink = (255 - other.astype(np.float32)) / 255.0
        paste(oink, cx + rng.uniform(-20, 20), cy + rng.choice([-1, 1]) * 4.6 * px_per_cm)
    angle = rng.uniform(-2, 2)
    M = cv2.getRotationMatrix2D((big.shape[1] / 2, big.shape[0] / 2), angle, 1.0)
    big = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]))
    ink_f = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
    paper = rng.uniform(150, 230)
    dark = rng.uniform(20, 70)
    img = paper - (paper - dark) * np.clip(ink_f, 0, 1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gx, gy = rng.uniform(-0.25, 0.25, 2)
    img *= 1 + gx * (xx / W - 0.5) + gy * (yy / H - 0.5)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 0.8))
    img += rng.normal(0, rng.uniform(2, 5), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--names", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--split", default="all", choices=("all", "test", "dev"))
    ap.add_argument("--cap-px", type=float, default=24.0, help="capital height in the fake photos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(config.TEST_SET_DIR) / "synthetic")
    args = ap.parse_args(argv)
    rows = read_lines(split=args.split)
    if not (args.pdf or args.names or args.render):
        ap.error("choose --pdf, --names and/or --render")
    if args.pdf:
        make_pdf(rows, Path(config.TEST_SET_DIR) / "test_lines.pdf")
    if args.names:
        path = Path(config.TEST_SET_DIR) / "photo_names.txt"
        path.write_text("\n".join(r["id"] for r in rows) + "\n")
        print(f"{path}: {len(rows)} names (photos go to test_set/images/<id>.png)")
    if args.render:
        rng = np.random.default_rng(args.seed)
        px_per_cm = args.cap_px / CAP_CM
        dpi = int(round(4 * px_per_cm * 2.54))          # render at 4x the camera's resolution
        imgs = render_lines(rows, dpi)
        args.out.mkdir(parents=True, exist_ok=True)
        for i, (row, img) in enumerate(zip(rows, imgs)):
            other = imgs[(i + 7) % len(imgs)] if rng.random() < 0.5 else None
            photo = fake_photo(img, px_per_cm, rng, dpi / 2.54, other)
            cv2.imwrite(str(args.out / f"{row['id']}.png"), photo)
        print(f"{len(imgs)} fake photos ({args.cap_px:.0f} px capitals) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
