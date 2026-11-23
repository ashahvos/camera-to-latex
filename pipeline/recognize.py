"""One photo of one printed line -> LaTeX. Used by the app and the evaluation."""
from __future__ import annotations

import time

import numpy as np

import config
from .classify import Classifier, apply_predictions, refine_scale
from .layout import layout_line
from .preprocess import binarize, normalize_crop, orient, to_gray
from .segment import segment
from .symbols import load_symbols
from .types import Box, LineResult, Symbol


def _classify(symbols, scale, classifier):
    syms = [s for s in symbols if s.kind == "symbol"]
    for s in syms:
        s.crop = normalize_crop(s.mask, scale)
    if syms:
        probs = classifier.predict(np.stack([s.crop for s in syms]))
        apply_predictions(syms, probs, classifier.classes)


def _new_symbol(mask: np.ndarray, x0: int, y0: int, scale: float, probs, classes):
    ys, xs = np.nonzero(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    box = Box(x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
    s = Symbol(box=box, mask=mask, crop=normalize_crop(mask, scale))
    apply_predictions([s], probs[None], classes)
    return s


def repair(symbols, scale: float, classifier):
    """Second chances for uncertain symbols, decided by the classifier itself.

    At about 24 px per capital, thin strokes can break (n -> two stems) and
    letters 1 px apart can touch (rn). Two adjacent pieces that are both
    uncertain are joined if the joined shape is recognised confidently;
    a wide uncertain blob is cut at its thinnest column if both halves are.
    """
    classes = classifier.classes
    syms = sorted([s for s in symbols if s.kind == "symbol"], key=lambda s: s.box.x0)
    others = [s for s in symbols if s.kind != "symbol"]
    need = config.REPAIR_CONF

    # join pairs
    i = 0
    while i + 1 < len(syms):
        a, b = syms[i], syms[i + 1]
        box = a.box.union(b.box)
        if (max(a.conf, b.conf) < config.CONF_MIN and b.box.x0 - a.box.x1 <= 0.15 * scale
                and a.box.y_overlap(b.box) >= 0.5 * min(a.box.h, b.box.h)
                and box.w <= 1.5 * scale and box.h <= 1.6 * scale):
            mask = np.zeros((box.h, box.w), bool)
            for s in (a, b):
                mask[s.box.y0 - box.y0:s.box.y1 - box.y0, s.box.x0 - box.x0:s.box.x1 - box.x0] |= s.mask
            p = classifier.predict(normalize_crop(mask, scale)[None])[0]
            if p.max() >= need and p.max() > max(a.conf, b.conf):
                syms[i:i + 2] = [_new_symbol(mask, box.x0, box.y0, scale, p, classes)]
                continue
        i += 1

    # cut wide blobs
    out = []
    for s in syms:
        w = s.box.w
        if s.conf >= config.CONF_MIN or w < 0.6 * scale or w < 0.5 * s.box.h:
            out.append(s)
            continue
        cols = s.mask.sum(axis=0)
        cuts = sorted(range(max(1, int(0.2 * w)), min(w - 1, int(0.8 * w) + 1)), key=lambda c: cols[c])[:3]
        best = None
        for c in cuts:
            left, right = s.mask[:, :c], s.mask[:, c + 1:]
            if left.sum() < 2 * config.MIN_BLOB_AREA or right.sum() < 2 * config.MIN_BLOB_AREA:
                continue
            pl, pr = classifier.predict(np.stack([normalize_crop(_trim(left), scale),
                                                  normalize_crop(_trim(right), scale)]))
            score = min(pl.max(), pr.max())
            if score >= need and score > s.conf + 0.1 and (best is None or score > best[0]):
                best = (score, c, pl, pr)
        if best is None:
            out.append(s)
            continue
        _, c, pl, pr = best
        out.append(_new_symbol(s.mask[:, :c], s.box.x0, s.box.y0, scale, pl, classes))
        out.append(_new_symbol(s.mask[:, c + 1:], s.box.x0 + c + 1, s.box.y0, scale, pr, classes))
    return sorted(out + others, key=lambda s: s.box.x0)


def _trim(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def recognize(image: np.ndarray, classifier: Classifier, mode: str = "auto",
              oriented: bool = False) -> LineResult:
    """Run the whole pipeline on one photo (raw camera frame or saved image).

    Two passes: crops are cut with a first guess of the line scale, then cut
    again if the classes show the guess was off by more than 15 %.
    """
    t0 = time.perf_counter()
    gray = to_gray(image)
    if not oriented:
        gray = orient(gray)
    binary = binarize(gray)
    symbols, scale, cut_off = segment(binary)
    notes = []
    if not symbols:
        return LineResult("", "text" if mode == "auto" else mode, [], scale, cut_off, ["no ink found"])
    _classify(symbols, scale, classifier)
    better = refine_scale(symbols, load_symbols())
    if better and abs(better / scale - 1.0) > 0.15:
        notes.append(f"line scale {scale:.1f} -> {better:.1f} px")
        scale = better
        _classify(symbols, scale, classifier)
    if config.REPAIR:
        n0 = len(symbols)
        symbols = repair(symbols, scale, classifier)
        if len(symbols) != n0:
            notes.append(f"repair: {n0} -> {len(symbols)} symbols")
    lay = layout_line(symbols, mode=mode, scale=scale)
    if cut_off:
        notes.append("ink touches the photo edge: part of the line may be missing")
    notes.append(f"{(time.perf_counter() - t0) * 1000:.0f} ms")
    return LineResult(lay.latex, lay.mode, symbols, scale, cut_off, notes)


def draw_result(gray: np.ndarray, result: LineResult, zoom: int = 2) -> np.ndarray:
    """Colour picture of the photo with a box per symbol (for the app window).

    green = confident, red = worth checking, blue = fraction bar.
    """
    import cv2
    img = cv2.cvtColor(to_gray(gray), cv2.COLOR_GRAY2BGR)
    if zoom != 1:
        img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    for s in result.symbols:
        b = s.box
        color = (255, 128, 0) if s.kind == "fracbar" else ((0, 0, 255) if s.flagged else (0, 200, 0))
        cv2.rectangle(img, (b.x0 * zoom, b.y0 * zoom), (b.x1 * zoom - 1, b.y1 * zoom - 1), color, 1)
        label = s.member or s.cls or ""
        if label and s.kind == "symbol":
            cv2.putText(img, label[:8], (b.x0 * zoom, max(10, b.y0 * zoom - 3)),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1, cv2.LINE_AA)
    return img
