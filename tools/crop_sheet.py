"""Cut photos of the data sheets into labelled training crops.

  python -m tools.crop_sheet                  # every data/sheets/<block>_<k>.png
  python -m tools.crop_sheet --debug          # also save data/sheets/debug/<photo>.png

For each photo: find the four corner squares (the hollow one is top left, so
a page photographed upside down still works; if its hole is too small to
see, the page is taken to be upright), map the grid with a
perspective transform, give each piece of ink to the cell its centre falls
in, and save one crop per cell to data/crops_real/<class>/<member>__<photo>__r<row>c<col>.png.
Crops are cut exactly like live symbols (same binarization, same
normalize_crop), with the line scale taken from the capital-height symbols
in the same photo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.preprocess import binarize, normalize_crop, orient, save_crop_png, to_gray
from pipeline.segment import estimate_scale, find_blobs
from pipeline.symbols import load_symbols

LAYOUT = Path(config.ROOT) / "sheets" / "layout.json"


class SheetError(RuntimeError):
    pass


def find_marks(labels: np.ndarray, stats: np.ndarray, cents: np.ndarray, scale: float,
               aspect: float | None = None):
    """The four corner squares: ids and centres in grid order (TL, TR, BR, BL), and whether
    the hollow (top-left) square was recognised.

    A mark is a square-ish blob about half a capital tall whose outline fills
    most of its box. Round symbols (o, 0, theta) can look like that in a small
    photo, so the marks are picked by position: the candidates farthest out
    towards the four corners of the photo. They must form the grid's rectangle
    (similar sizes, opposite sides alike, the right proportions); if not, a
    corner is out of view or something else square-ish is in the picture.

    The hollow square marks the grid's top left, so a page that was turned
    around still crops correctly. At about 20 px per capital its hole is only a
    few pixels wide and can fill in; when no square is clearly hollow, the page
    is taken to be upright (hollow square at the top left of the photo).
    """
    cands = []
    for i in range(1, len(stats)):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (0.7 <= w / max(h, 1) <= 1.43) or not (0.3 * scale <= max(w, h) <= 0.8 * scale):
            continue
        mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 1, thickness=-1)
        outer = int(filled.sum())
        if outer < 0.7 * w * h:
            continue
        cands.append((i, cents[i], area / max(outer, 1), w * h))
    if len(cands) < 4:
        raise SheetError(f"found {len(cands)} corner squares, need 4 (all corners in view?)")
    xs = np.array([c[1][0] for c in cands])
    ys = np.array([c[1][1] for c in cands])
    # farthest towards the photo's top left, top right, bottom right, bottom left
    pick = [int(np.argmin(xs + ys)), int(np.argmax(xs - ys)), int(np.argmax(xs + ys)), int(np.argmin(xs - ys))]
    if len(set(pick)) != 4:
        raise SheetError("corner squares are not at four corners (all corners in view? page straight?)")
    quad = [cands[k] for k in pick]
    sizes = [c[3] for c in quad]
    if max(sizes) > 2.25 * min(sizes):
        raise SheetError("corner squares have very different sizes (something else in view?)")
    pts = np.array([c[1] for c in quad], np.float32)
    top, right = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])
    bottom, left = np.linalg.norm(pts[3] - pts[2]), np.linalg.norm(pts[0] - pts[3])
    if min(top, bottom) < 0.8 * max(top, bottom) or min(left, right) < 0.8 * max(left, right):
        raise SheetError("corner squares do not form a rectangle (a corner out of view, or another block in view?)")
    if aspect:
        ratio = max(top + bottom, left + right) / max(1e-6, min(top + bottom, left + right))
        want = max(aspect, 1.0 / aspect)
        if not (want / 1.2 <= ratio <= want * 1.2):
            raise SheetError("corner squares do not match the grid's shape (another block or object in view?)")
    fills = [c[2] for c in quad]
    order = np.argsort(fills)
    hollow_clear = bool(fills[order[0]] <= 0.93 and fills[order[1]] - fills[order[0]] >= 0.05)
    k = int(order[0]) if hollow_clear else 0
    quad = quad[k:] + quad[:k]
    return [c[0] for c in quad], np.array([c[1] for c in quad], np.float32), hollow_clear


def crop_photo(path: Path, block: dict, layout: dict, table, out_dir: Path, debug_dir: Path | None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SheetError("cannot read photo")
    gray = orient(to_gray(img))
    cols, rows = layout["cols"], layout["rows"]
    px, py = layout["pitch_x_cm"], layout["pitch_y_cm"]
    o = layout["mark_gap_cm"] + layout["mark_cm"] / 2
    aspect = (cols * px + 2 * o) / (rows * py + 2 * o)
    for _ in range(2):
        binary = binarize(gray)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
        rough = estimate_scale(find_blobs(binary)[1])
        mark_ids, pts, hollow_seen = find_marks(labels, stats, cents, rough, aspect)
        # page turned by 90/180/270 degrees: turn the photo back so the symbols are upright
        v = pts[1] - pts[0]
        quarter = int(round(np.degrees(np.arctan2(v[1], v[0])) / 90.0)) % 4
        if quarter == 0:
            break
        gray = cv2.rotate(gray, {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180,
                                 3: cv2.ROTATE_90_CLOCKWISE}[quarter])
    Wc, Hc = cols * px, rows * py
    grid_pts = np.array([(-o, -o), (Wc + o, -o), (Wc + o, Hc + o), (-o, Hc + o)], np.float32)
    to_grid = cv2.getPerspectiveTransform(pts, grid_pts)

    # each piece of ink goes to the cell its centre falls in
    centres = cv2.perspectiveTransform(cents[1:].reshape(-1, 1, 2).astype(np.float32), to_grid).reshape(-1, 2)
    cells: dict[tuple, list] = {}
    for k, (gx, gy) in enumerate(centres, start=1):
        if k in mark_ids or stats[k, cv2.CC_STAT_AREA] < config.MIN_BLOB_AREA:
            continue
        c, r = int(np.floor(gx / px)), int(np.floor(gy / py))
        if 0 <= c < cols and 0 <= r < rows:
            cells.setdefault((r, c), []).append(k)

    by_pos = {(cl["row"], cl["col"]): cl for cl in block["cells"]}
    masks = {}
    for pos, ids in cells.items():
        x0 = min(stats[k, 0] for k in ids)
        y0 = min(stats[k, 1] for k in ids)
        x1 = max(stats[k, 0] + stats[k, 2] for k in ids)
        y1 = max(stats[k, 1] + stats[k, 3] for k in ids)
        masks[pos] = ((x0, y0, x1, y1), np.isin(labels[y0:y1, x0:x1], ids))
    caps = [m[1].shape[0] for pos, m in masks.items()
            if pos in by_pos and table.members[by_pos[pos]["member"]].height == "cap"]
    if len(caps) < 3:
        raise SheetError("too few symbols found to measure the capital height")
    scale = float(np.median(caps))
    saved, empty = 0, []
    for pos, cl in sorted(by_pos.items()):
        if pos not in masks:
            empty.append(cl["member"])
            continue
        crop = normalize_crop(masks[pos][1], scale)
        save_crop_png(out_dir / cl["class"] / f"{cl['member']}__{path.stem}__r{pos[0]}c{pos[1]}.png", crop)
        saved += 1
    if debug_dir is not None:
        _debug_image(gray, to_grid, masks, pts, cols, rows, px, py, debug_dir / f"{path.stem}.png")
    return saved, empty, scale, hollow_seen


def _debug_image(gray, to_grid, masks, pts, cols, rows, px, py, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    to_img = np.linalg.inv(to_grid)

    def seg(a, b):
        p = cv2.perspectiveTransform(np.array([[a], [b]], np.float32), to_img).reshape(-1, 2)
        cv2.line(vis, tuple(int(v) for v in p[0]), tuple(int(v) for v in p[1]), (0, 200, 255), 1)

    for r in range(rows + 1):
        seg((0, r * py), (cols * px, r * py))
    for c in range(cols + 1):
        seg((c * px, 0), (c * px, rows * py))
    for (x0, y0, x1, y1), _ in masks.values():
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), (0, 255, 0), 1)
    for q in pts:
        cv2.circle(vis, (int(q[0]), int(q[1])), 4, (0, 0, 255), 1)
    cv2.imwrite(str(path), cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--photos", type=Path, default=Path(config.SHEETS_DIR))
    ap.add_argument("--out", type=Path, default=Path(config.CROPS_REAL_DIR))
    ap.add_argument("--layout", type=Path, default=LAYOUT)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    layout = json.loads(args.layout.read_text())
    blocks = {b["id"]: b for b in layout["blocks"]}
    table = load_symbols()
    photos = sorted(p for p in args.photos.glob("*.png") if p.stem.split("_")[0] in blocks)
    if not photos:
        print(f"no sheet photos (<block>_<k>.png) in {args.photos}")
        return 1
    total, failed, upright = 0, [], []
    for p in photos:
        block = blocks[p.stem.split("_")[0]]
        try:
            saved, empty, scale, hollow_seen = crop_photo(p, block, layout, table, args.out,
                                                          args.photos / "debug" if args.debug else None)
        except SheetError as exc:
            failed.append(p.name)
            print(f"{p.name}: skipped ({exc})")
            continue
        total += saved
        note = f"; empty cells: {', '.join(empty)}" if empty else ""
        if not hollow_seen:
            upright.append(p.name)
            note += "; read as an upright page"
        print(f"{p.name}: {saved} crops, capitals {scale:.0f} px{note}")
    print(f"\n{total} crops from {len(photos) - len(failed)} photos -> {args.out}")
    if upright:
        print(f"{len(upright)} photos had no clearly hollow corner square and were read as upright pages "
              "(hollow square at the top left of the photo). Retake any that were turned.")
    if failed:
        print(f"retake: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
