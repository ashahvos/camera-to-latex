"""Cut photos of the data sheets into labelled training crops.

  python -m tools.crop_sheet                  # every data/sheets/<block>_<k>.png
  python -m tools.crop_sheet --debug          # also save data/sheets/debug/<photo>.png

For each photo: find the four corner squares (the hollow one is top left, so
a page photographed upside down still works), map the grid with a
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


def find_marks(labels: np.ndarray, stats: np.ndarray, cents: np.ndarray, scale: float):
    """Component ids of the four corner squares in grid order (TL, TR, BR, BL) and their centres.

    A mark is a square blob about half a capital tall whose outline fills its
    box; the top-left one is hollow. Any other count than four means a mark
    is out of view (or something square-ish is lying on the page): retake.
    """
    cands = []
    for i in range(1, len(stats)):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (0.75 <= w / max(h, 1) <= 1.33) or not (0.3 * scale <= max(w, h) <= 0.8 * scale):
            continue
        mask = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 1, thickness=-1)
        outer = int(filled.sum())
        if outer < 0.8 * w * h:
            continue
        cands.append((i, cents[i], area / max(outer, 1), w * h))
    if len(cands) != 4:
        raise SheetError(f"found {len(cands)} corner squares, need exactly 4 (all corners in view?)")
    sizes = [c[3] for c in cands]
    if max(sizes) > 2.25 * min(sizes):
        raise SheetError("corner squares have very different sizes")
    hollow = min(range(4), key=lambda k: cands[k][2])
    if cands[hollow][2] > 0.85 or sorted(c[2] for c in cands)[1] < 0.85:
        raise SheetError("cannot tell which corner square is the hollow (top-left) one")
    centre = np.mean([c[1] for c in cands], axis=0)
    angle = lambda c: np.arctan2(c[1][1] - centre[1], c[1][0] - centre[0])
    ring = sorted(cands, key=angle)                         # clockwise on screen (y points down)
    k = [c[0] for c in ring].index(cands[hollow][0])
    order = ring[k:] + ring[:k]
    return [c[0] for c in order], np.array([c[1] for c in order], np.float32)


def crop_photo(path: Path, block: dict, layout: dict, table, out_dir: Path, debug_dir: Path | None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SheetError("cannot read photo")
    gray = orient(to_gray(img))
    for _ in range(2):
        binary = binarize(gray)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
        rough = estimate_scale(find_blobs(binary)[1])
        mark_ids, pts = find_marks(labels, stats, cents, rough)
        # page turned by 90/180/270 degrees: turn the photo back so the symbols are upright
        v = pts[1] - pts[0]
        quarter = int(round(np.degrees(np.arctan2(v[1], v[0])) / 90.0)) % 4
        if quarter == 0:
            break
        gray = cv2.rotate(gray, {1: cv2.ROTATE_90_COUNTERCLOCKWISE, 2: cv2.ROTATE_180,
                                 3: cv2.ROTATE_90_CLOCKWISE}[quarter])
    cols, rows = layout["cols"], layout["rows"]
    px, py = layout["pitch_x_cm"], layout["pitch_y_cm"]
    o = layout["mark_gap_cm"] + layout["mark_cm"] / 2
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
    return saved, empty, scale


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
    total, failed = 0, []
    for p in photos:
        block = blocks[p.stem.split("_")[0]]
        try:
            saved, empty, scale = crop_photo(p, block, layout, table, args.out,
                                             args.photos / "debug" if args.debug else None)
        except SheetError as exc:
            failed.append(p.name)
            print(f"{p.name}: skipped ({exc})")
            continue
        total += saved
        note = f"; empty cells: {', '.join(empty)}" if empty else ""
        print(f"{p.name}: {saved} crops, capitals {scale:.0f} px{note}")
    print(f"\n{total} crops from {len(photos) - len(failed)} photos -> {args.out}")
    if failed:
        print(f"retake: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
