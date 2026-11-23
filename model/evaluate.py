"""Score a trained classifier on crops (not whole lines; for lines use tools/eval_pipeline.py).

  python -m model.evaluate                         # held-out real crops (data/crops_real)
  python -m model.evaluate --all-real              # every real crop
  python -m model.evaluate --npz data/synthetic/test.npz

Prints accuracy, the worst classes and the most common confusions, and writes
results/crops_<model>.csv (one row per class).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from model.data import holdout_mask, load_npz, load_real_crops
from model.train import per_class_report
from pipeline.classify import Classifier
from pipeline.symbols import load_symbols


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=config.MODEL_NAME)
    ap.add_argument("--npz", type=Path, help="evaluate on this .npz instead of real crops")
    ap.add_argument("--real", type=Path, default=Path(config.CROPS_REAL_DIR))
    ap.add_argument("--all-real", action="store_true", help="use every real crop, not just held-out photos")
    args = ap.parse_args(argv)

    table = load_symbols()
    clf = Classifier.load(args.model)
    if args.npz:
        X, y = load_npz(args.npz, table)
        source = str(args.npz)
    else:
        X, y, groups = load_real_crops(args.real, table)
        if not args.all_real and len(X):
            keep = holdout_mask(groups)
            X, y = X[keep], y[keep]
        source = f"{args.real} ({'all' if args.all_real else 'held-out photos'})"
    if len(X) == 0:
        print(f"no crops found in {source}")
        return 1
    probs = clf.predict(X.astype(np.float32) / 255.0)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    acc = float((pred == y).mean())
    flagged = conf < config.CONF_MIN
    print(f"{source}: {len(X)} crops, accuracy {acc:.1%}")
    print(f"flagged as uncertain: {flagged.mean():.1%} (accuracy among the rest {(pred == y)[~flagged].mean():.1%})")
    print("most common confusions:")
    print(per_class_report(y, pred, table.classes))

    out = Path(config.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"crops_{args.model}.csv"
    rows = []
    for ci, name in enumerate(table.classes):
        m = y == ci
        if m.any():
            rows.append((name, int(m.sum()), float((pred[m] == ci).mean())))
    rows.sort(key=lambda r: r[2])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "crops", "accuracy"])
        w.writerows([(n, c, f"{a:.3f}") for n, c, a in rows])
    print("worst classes: " + ", ".join(f"{n} {a:.0%}" for n, _, a in rows[:8]))
    print(f"per-class table: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
