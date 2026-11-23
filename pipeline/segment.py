"""Split a binarized photo of one printed line into symbols.

Steps (each is a function so it can be tested and tuned on its own):
  1. find_blobs         connected components of ink, noise and edge blobs dropped
  2. estimate_scale     rough capital height of the line, in pixels
  3. select_line        keep the row of ink closest to the middle of the photo
  4. mark_fraction_bars thin bars with symbols above and below them
  5. merge_parts        join the pieces of one symbol (i, j, =, :, ;, !, ?, %, ...)
  6. build_symbols      one Symbol (box + own ink mask) per merged group
"""
from __future__ import annotations

import numpy as np
import cv2

import config
from .types import Blob, Box, Symbol


# ---------------------------------------------------------------- 1. blobs ---
def find_blobs(binary: np.ndarray):
    """Return (label image, kept blobs, blobs dropped because they touch the edge)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    H, W = binary.shape
    total = float(H * W)
    kept, edge = [], []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < config.MIN_BLOB_AREA:
            continue
        if area > config.MAX_BLOB_FRACTION * total or w * h > 0.9 * total:
            continue
        blob = Blob(Box(x, y, x + w, y + h), area, [i])
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            edge.append(blob)
        else:
            kept.append(blob)
    return labels, kept, edge


# ---------------------------------------------------------------- 2. scale ---
def estimate_scale(blobs) -> float:
    """First guess of the capital height, from blob heights alone.

    Dots and bars are ignored (anything under 40 % of the 90th-percentile
    height); the median of the taller half of what is left lands on capitals,
    digits and ascenders in ordinary text. classify.refine_scale() improves it
    once the classes are known.
    """
    hs = np.array([b.box.h for b in blobs if b.kind == "symbol"], dtype=float)
    if len(hs) == 0:
        return float(config.DEFAULT_SCALE)
    p90 = np.percentile(hs, 90)
    cand = np.sort(hs[hs >= 0.4 * p90])
    upper = cand[len(cand) // 2:]
    return float(max(4.0, np.median(upper)))


# ----------------------------------------------------------------- 3. line ---
def select_line(blobs, scale: float, shape) -> list:
    """Keep only the row of ink nearest the middle of the photo.

    Blobs are grouped into rows by their vertical extent (gaps under 0.3
    line scales bridged). A thin bar with a row just above and a row just
    below it (a display-style fraction) joins those rows into one. Among rows
    near the centre, the one with the most ink wins.
    """
    if len(blobs) <= 1:
        return list(blobs)
    bars = [b for b in blobs if _is_bar(b, scale) and b.box.w >= 0.4 * scale]
    rest = [b for b in blobs if not any(b is x for x in bars)]
    pad = 0.3 * scale
    rows = []
    for b in sorted(rest, key=lambda b: b.box.y0):
        if rows and b.box.y0 <= rows[-1]["y1"] + pad:
            rows[-1]["y1"] = max(rows[-1]["y1"], b.box.y1)
            rows[-1]["blobs"].append(b)
        else:
            rows.append({"y0": b.box.y0, "y1": b.box.y1, "blobs": [b]})
    for bar in sorted(bars, key=lambda b: -b.box.w):
        def overlaps(r):
            return any(bar.box.x_overlap(b.box) > 0 for b in r["blobs"])
        above = [r for r in rows if r["y1"] <= bar.box.y0 + 1 and bar.box.y0 - r["y1"] <= 1.0 * scale and overlaps(r)]
        below = [r for r in rows if r["y0"] >= bar.box.y1 - 1 and r["y0"] - bar.box.y1 <= 1.0 * scale and overlaps(r)]
        inside = [r for r in rows if r["y0"] <= bar.box.cy <= r["y1"]]
        if above and below:
            a = max(above, key=lambda r: r["y1"])
            b = min(below, key=lambda r: r["y0"])
            merged = {"y0": min(a["y0"], b["y0"]), "y1": max(a["y1"], b["y1"]),
                      "blobs": a["blobs"] + b["blobs"] + [bar]}
            rows = [r for r in rows if r is not a and r is not b] + [merged]
        elif inside:
            inside[0]["blobs"].append(bar)
        else:
            near = min(rows, key=lambda r: min(abs(r["y0"] - bar.box.cy), abs(r["y1"] - bar.box.cy)),
                       default=None)
            if near is not None and min(abs(near["y0"] - bar.box.cy), abs(near["y1"] - bar.box.cy)) <= scale:
                near["blobs"].append(bar)
                near["y0"], near["y1"] = min(near["y0"], bar.box.y0), max(near["y1"], bar.box.y1)
            else:
                rows.append({"y0": bar.box.y0, "y1": bar.box.y1, "blobs": [bar]})
    rows.sort(key=lambda r: r["y0"])
    # rows that now overlap (a fraction merged next to a normal row) become one
    merged_rows = []
    for r in rows:
        if merged_rows and r["y0"] <= merged_rows[-1]["y1"]:
            merged_rows[-1]["y1"] = max(merged_rows[-1]["y1"], r["y1"])
            merged_rows[-1]["blobs"] += r["blobs"]
        else:
            merged_rows.append(dict(r))
    rows = merged_rows
    if len(rows) == 1:
        return rows[0]["blobs"]
    cy = shape[0] / 2.0

    def dist(r):
        if r["y0"] <= cy <= r["y1"]:
            return 0.0
        return min(abs(cy - r["y0"]), abs(cy - r["y1"]))

    near = [r for r in rows if dist(r) <= 0.25 * shape[0]]
    if near:
        best = max(near, key=lambda r: sum(b.area for b in r["blobs"]))
    else:
        best = min(rows, key=dist)
    return best["blobs"]


# ------------------------------------------------------- 4. fraction bars ---
def _is_bar(b: Blob, scale: float) -> bool:
    w, h = b.box.w, b.box.h
    return w >= config.FRACTION_BAR_ASPECT * h and h <= 0.25 * scale


def mark_fraction_bars(blobs, scale: float) -> None:
    """Mark thin bars that have a real symbol right above and right below.

    Must run before merge_parts so a numerator is never glued to its bar. A
    neighbour counts only if it is at least FRACTION_NEIGHBOR_MIN line scales
    tall (so the second bar of '=' or the dots of a division sign don't), and
    no other bar sits between it and the candidate.
    """
    bars = [b for b in blobs if _is_bar(b, scale) and b.box.w >= 0.4 * scale]
    max_gap = config.FRACTION_MAX_GAP * scale
    min_h = config.FRACTION_NEIGHBOR_MIN * scale
    all_bars = [b for b in blobs if _is_bar(b, scale)]

    def blocked(bar, other):
        lo, hi = sorted([bar.box.cy, other.box.cy])
        for c in all_bars:
            if c is bar or c is other:
                continue
            if lo < c.box.cy < hi and c.box.x_overlap(other.box) > 0 and c.box.x_overlap(bar.box) > 0:
                return True
        return False

    for bar in bars:
        above = below = False
        for o in blobs:
            if o is bar or o.box.h < min_h:
                continue
            if bar.box.x_overlap(o.box) < 0.5 * o.box.w:
                continue
            if o.box.y1 <= bar.box.y0 + 1 and bar.box.y0 - o.box.y1 <= max_gap:
                above = above or not blocked(bar, o)
            elif o.box.y0 >= bar.box.y1 - 1 and o.box.y0 - bar.box.y1 <= max_gap:
                below = below or not blocked(bar, o)
        if above and below:
            bar.kind = "fracbar"


# ----------------------------------------------------------------- 5. merge ---
def _is_minor(b: Blob, scale: float) -> bool:
    """Dots, small rings, thin bars and waves: pieces that may belong to a bigger symbol."""
    return max(b.box.w, b.box.h) <= 0.4 * scale or (b.box.h <= 0.4 * scale and b.box.w >= 2 * b.box.h)


def _inside(a: Blob, b: Blob) -> float:
    """Share of a's box that lies inside b's box."""
    return a.box.x_overlap(b.box) * a.box.y_overlap(b.box) / max(1, a.box.w * a.box.h)


def merge_parts(blobs, scale: float) -> list:
    """Group blobs that form one symbol. Returns a list of lists of blobs.

    Rules (scale = line scale, the capital height):
      * stacked: every minor piece joins its nearest partner that overlaps it
        horizontally (by MERGE_OVERLAP of the narrower one), is at most 0.6
        scales away vertically, keeps the joined height under MERGE_MAX_HEIGHT
        and is not across a fraction bar (i, j, =, :, ;, !, ?, Xi, approx, <=);
      * inside: a small piece within a bigger piece's height, centred inside
        its width and at least 30 % inside its box, joins it (the rings of %,
        the bar of Theta); a superscript ring next to a 0 stays separate;
      * side by side: two small ticks next to each other (").
    """
    syms = [b for b in blobs if b.kind == "symbol"]
    fracbars = [b for b in blobs if b.kind == "fracbar"]
    parent = list(range(len(syms)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    def across_bar(a, b):
        lo, hi = sorted([a.box.cy, b.box.cy])
        return any(lo < f.box.cy < hi and f.box.x_overlap(a.box) > 0 and f.box.x_overlap(b.box) > 0
                   for f in fracbars)

    max_gap = 0.6 * scale
    max_h = config.MERGE_MAX_HEIGHT * scale
    for i, m in enumerate(syms):
        if not _is_minor(m, scale):
            continue
        best, best_cost = None, float("inf")
        for j, o in enumerate(syms):
            if i == j:
                continue
            ov = m.box.x_overlap(o.box) / max(1, min(m.box.w, o.box.w))
            if ov < config.MERGE_OVERLAP:
                continue
            gap = m.box.y_gap(o.box)
            if gap > max_gap:
                continue
            if max(m.box.y1, o.box.y1) - min(m.box.y0, o.box.y0) > max_h:
                continue
            if across_bar(m, o):
                continue
            cost = gap + abs(m.box.cx - o.box.cx)
            if cost < best_cost:
                best, best_cost = j, cost
        if best is not None:
            union(i, best)

    # small pieces inside (or half inside) a bigger piece's box
    for i, a in enumerate(syms):
        if max(a.box.w, a.box.h) > 0.55 * scale:
            continue
        best, best_area = None, 0
        for j, b in enumerate(syms):
            if i == j or b.box.h * 0.6 < a.box.h or b.box.w * b.box.h <= best_area:
                continue
            tol = max(1.0, 0.1 * scale)
            if a.box.y0 < b.box.y0 - tol or a.box.y1 > b.box.y1 + tol or not (b.box.x0 <= a.box.cx <= b.box.x1):
                continue
            if _inside(a, b) >= 0.3 and not across_bar(a, b):
                best, best_area = j, b.box.w * b.box.h
        if best is not None:
            union(i, best)

    # side by side: the two ticks of "
    # (<< >> and the two bars of the parallel sign are joined later, in layout)
    order = sorted(range(len(syms)), key=lambda i: syms[i].box.x0)
    for a, b in zip(order, order[1:]):
        A, B = syms[a].box, syms[b].box
        if A.y_overlap(B) < 0.7 * min(A.h, B.h) or min(A.h, B.h) < 0.75 * max(A.h, B.h):
            continue
        if (max(A.h, B.h) <= 0.45 * scale and A.h >= 1.2 * A.w and B.h >= 1.2 * B.w
                and B.x0 - A.x1 <= 0.15 * scale):
            union(a, b)

    groups: dict[int, list] = {}
    for i in range(len(syms)):
        groups.setdefault(find(i), []).append(syms[i])
    out = list(groups.values()) + [[f] for f in fracbars]
    out.sort(key=lambda g: min(b.box.x0 for b in g))
    return out


# --------------------------------------------------------------- 6. symbols ---
def build_symbols(labels: np.ndarray, groups) -> list:
    out = []
    for g in groups:
        box = Box.union_all(b.box for b in g)
        ids = [i for b in g for i in b.labels]
        sub = labels[box.y0:box.y1, box.x0:box.x1]
        mask = np.isin(sub, ids)
        kind = "fracbar" if any(b.kind == "fracbar" for b in g) else "symbol"
        out.append(Symbol(box=box, mask=mask, kind=kind))
    return out


def segment(binary: np.ndarray):
    """Binarized photo -> (symbols left to right, line scale, cut_off flag)."""
    labels, blobs, edge = find_blobs(binary)
    scale = estimate_scale(blobs)
    blobs = select_line(blobs, scale, binary.shape)
    scale = estimate_scale(blobs)
    mark_fraction_bars(blobs, scale)
    groups = merge_parts(blobs, scale)
    symbols = build_symbols(labels, groups)
    cut_off = False
    if symbols and edge:
        ys0 = min(s.box.y0 for s in symbols) - scale
        ys1 = max(s.box.y1 for s in symbols) + scale
        cut_off = any(e.box.h <= 3 * scale and e.box.w <= 4 * scale
                      and e.box.y1 >= ys0 and e.box.y0 <= ys1 for e in edge)
    return symbols, scale, cut_off
