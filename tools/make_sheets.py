"""Data-collection sheets: every symbol printed in a grid the board photographs.

  python -m tools.make_sheets            # sheets/sheets.pdf, sheets/layout.json, sheets/photo_names.txt

Layout: blocks of 7 x 5 cells (1.6 x 1.7 cm), one symbol per cell at 40 pt,
with a square at each corner of the grid (the top-left one hollow) so
tools/crop_sheet.py can find the grid in a photo even if the page is tilted
or upside down. The 182 members fill 6 blocks; spare cells repeat symbols that
are easy to confuse. Three sets are printed with different cell orders:
  A  text style where a symbol has one (upright letters)
  B  math style (italic letters, as in formulas)
  C  a random mix
18 blocks, 2 per page. Photograph each block 3 times (move the page a little
and change the light between shots): 54 photos, about 9 crops per symbol.
Print at 100 % ("actual size"), not "fit to page".
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.symbols import load_symbols

COLS, ROWS = 7, 5
PITCH_X, PITCH_Y = 1.6, 1.7          # cm
MARK = 0.5                           # cm, side of a corner square
MARK_GAP = 0.25                      # cm between the grid and a square
FONT_PT = 40
SETS = ("A", "B", "C")
SHEETS_DIR = Path(config.ROOT) / "sheets"
# Extra copies for the spare cells: symbols that are easy to mix up.
HARD = ["lower_l", "upper_I", "digit_1", "vert", "lower_o", "upper_O", "digit_0", "comma",
        "apostrophe", "period", "cdot", "minus", "underscore", "lower_x", "times", "digit_5",
        "lower_s", "lower_v", "lower_u", "lower_n", "lower_h", "upper_W", "lower_w", "lower_m",
        "alpha", "lower_a", "lower_y", "lower_g", "digit_9", "lower_q", "theta", "f_math"]


def cell_tex(member, style: str) -> str:
    if style == "text" and member.latex_text:
        return member.latex_text
    return "$" + member.latex_math + "$"


def build_layout(table, seed: int = 0) -> dict:
    rng = random.Random(seed)
    per_block = COLS * ROWS
    names = [m.name for m in table.member_list]
    n_blocks = -(-len(names) // per_block)
    spare = n_blocks * per_block - len(names)
    extra = [h for h in HARD if h in table.members][:spare]
    blocks = []
    for s in SETS:
        cells = names + extra
        cells += names[:max(0, n_blocks * per_block - len(cells))]
        rng.shuffle(cells)
        for b in range(n_blocks):
            chunk = cells[b * per_block:(b + 1) * per_block]
            out = []
            for k, name in enumerate(chunk):
                m = table.members[name]
                if s == "A":
                    style = "text" if m.latex_text else "math"
                elif s == "B":
                    style = "math"
                else:
                    style = rng.choice(["text", "math"]) if m.latex_text else "math"
                out.append({"row": k // COLS, "col": k % COLS, "member": name, "class": m.cls, "style": style})
            blocks.append({"id": f"{s}{b + 1}", "cells": out})
    return {"cols": COLS, "rows": ROWS, "pitch_x_cm": PITCH_X, "pitch_y_cm": PITCH_Y,
            "mark_cm": MARK, "mark_gap_cm": MARK_GAP, "font_pt": FONT_PT, "blocks": blocks}


def block_tikz(block, table) -> str:
    W, H = COLS * PITCH_X, ROWS * PITCH_Y
    lines = [r"\begin{tikzpicture}[x=1cm,y=-1cm]"]
    o = MARK_GAP + MARK
    lines.append(r"\draw[line width=1mm] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                 % (-o + 0.05, -o + 0.05, -MARK_GAP - 0.05, -MARK_GAP - 0.05))       # hollow top-left
    for (x, y) in ((W + MARK_GAP, -o), (W + MARK_GAP, H + MARK_GAP), (-o, H + MARK_GAP)):
        lines.append(r"\fill (%.3f,%.3f) rectangle ++(%.3f,%.3f);" % (x, y, MARK, MARK))
    for c in block["cells"]:
        m = table.members[c["member"]]
        x = (c["col"] + 0.5) * PITCH_X
        y = (c["row"] + 0.5) * PITCH_Y
        lines.append(r"\node[inner sep=0pt] at (%.3f,%.3f) {\fontsize{%d}{%d}\selectfont %s};"
                     % (x, y, FONT_PT, int(FONT_PT * 1.2), cell_tex(m, c["style"])))
    lines.append(r"\node[anchor=north west,font=\footnotesize\ttfamily] at (%.3f,%.3f) {block %s};"
                 % (-o, H + o + 0.15, block["id"]))
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)


def make_pdf(layout, table, out: Path) -> None:
    parts = [r"\documentclass[11pt]{article}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}",
             r"\usepackage{amsmath,amssymb}", r"\usepackage{exscale}", r"\usepackage{tikz}",
             r"\usepackage[letterpaper,margin=1.2cm]{geometry}", r"\pagestyle{empty}",
             r"\setlength{\parindent}{0pt}", r"\begin{document}"]
    for i, block in enumerate(layout["blocks"]):
        if i % 2 == 0:
            if i:
                parts.append(r"\newpage")
            parts.append(r"{\small Camera-to-LaTeX data sheet. Print at 100\,\%. Photograph each block "
                         r"3 times with all four corner squares in view; the hollow square goes top left.}"
                         r"\par\vspace{0.3cm}")
        parts.append(r"\begin{center}" + block_tikz(block, table) + r"\end{center}\vspace{0.2cm}")
    parts.append(r"\end{document}")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "sheets.tex").write_text("\n".join(parts))
        r = subprocess.run([config.PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", "sheets.tex"],
                           cwd=tmp, capture_output=True, text=True, errors="replace")
        if r.returncode != 0:
            raise RuntimeError("pdflatex failed:\n" + r.stdout[-2500:])
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp / "sheets.pdf", out)
        shutil.copy(tmp / "sheets.tex", out.with_suffix(".tex"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--photos-per-block", type=int, default=3)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args(argv)
    table = load_symbols()
    layout = build_layout(table, args.seed)
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    (SHEETS_DIR / "layout.json").write_text(json.dumps(layout, indent=1))
    names = [f"{b['id']}_{k + 1}" for b in layout["blocks"] for k in range(args.photos_per_block)]
    (SHEETS_DIR / "photo_names.txt").write_text("\n".join(names) + "\n")
    if not args.no_pdf:
        make_pdf(layout, table, SHEETS_DIR / "sheets.pdf")
    n_cells = sum(len(b["cells"]) for b in layout["blocks"])
    print(f"{len(layout['blocks'])} blocks, {n_cells} cells, {len(names)} photos to take -> {SHEETS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
