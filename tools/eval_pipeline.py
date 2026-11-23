"""Score the whole pipeline on the test lines (photo -> LaTeX), per category.

  python -m tools.eval_pipeline                          # test split, test_set/images/<id>.png
  python -m tools.eval_pipeline --split dev              # tune on these, report on test
  python -m tools.eval_pipeline --images test_set/synthetic --split all
  python -m tools.eval_pipeline --no-compile             # faster: skip the pdflatex check

Metrics (per category and overall):
  exact     share of lines whose LaTeX matches after normalization
            (spacing, braces and synonyms such as \\le / \\leq don't count)
  TER       token error rate: edit distance between token lists / expected tokens
  compiles  share of predicted lines that compile on their own
  time      seconds per line for photo -> LaTeX (not counting pdflatex)
Each line is read in the mode given in lines.tsv (as the user would set it).
Writes results/eval_<split>_<time>.csv with one row per line.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.classify import Classifier
from pipeline.latexdoc import check_line
from pipeline.latexnorm import same_line, token_error_rate
from pipeline.recognize import recognize
from tools.make_test_lines import read_lines

CATEGORIES = ("plain", "scripts", "fractions", "mixed")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path(config.TEST_SET_DIR) / "images")
    ap.add_argument("--split", default="test", choices=("test", "dev", "all"))
    ap.add_argument("--model", default=config.MODEL_NAME)
    ap.add_argument("--auto-mode", action="store_true", help="let the app guess text/math")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="only the summary")
    args = ap.parse_args(argv)

    clf = Classifier.load(args.model)
    rows = read_lines(split=args.split)
    results = []
    for r in rows:
        path = args.images / f"{r['id']}.png"
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        t0 = time.perf_counter()
        res = recognize(img, clf, mode="auto" if args.auto_mode else r["mode"])
        dt = time.perf_counter() - t0
        exact = same_line(res.latex, r["latex"], r["mode"]) and res.mode == r["mode"]
        ter = token_error_rate(res.latex, r["latex"], r["mode"])
        compiles = None
        if not args.no_compile:
            compiles = check_line(res.latex, res.mode)[0] if res.latex else False
        results.append({**r, "predicted": res.latex, "pred_mode": res.mode, "exact": exact,
                        "ter": ter, "compiles": compiles, "seconds": dt,
                        "flagged": sum(s.flagged for s in res.symbols)})
        if not args.quiet:
            mark = "ok " if exact else "-- "
            print(f"{mark}{r['id']} {r['category']:9s} expected {r['latex']:28s} got {res.latex}")
    if not results:
        print(f"no photos found in {args.images} for split {args.split}")
        return 1

    def summary(items):
        n = len(items)
        comp = [x["compiles"] for x in items if x["compiles"] is not None]
        return (n, sum(x["exact"] for x in items) / n, sum(x["ter"] for x in items) / n,
                (sum(comp) / len(comp)) if comp else float("nan"), sum(x["seconds"] for x in items) / n)

    print(f"\n{'category':10s} {'lines':>5s} {'exact':>7s} {'TER':>7s} {'compiles':>9s} {'s/line':>7s}")
    for cat in CATEGORIES + ("all",):
        items = results if cat == "all" else [x for x in results if x["category"] == cat]
        if items:
            n, ex, ter, comp, sec = summary(items)
            print(f"{cat:10s} {n:5d} {ex:7.1%} {ter:7.1%} {comp:9.1%} {sec:7.2f}")
    out = Path(config.RESULTS_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{args.split}_{time.strftime('%Y%m%d_%H%M')}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nper-line results: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
