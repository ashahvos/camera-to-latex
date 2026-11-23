"""The app: point the board at a printed line, press SPACE, get LaTeX, build a PDF.

  python -m pipeline.main                      live, with the board
  python -m pipeline.main --image a.png ...    saved photos instead of the board
  python -m pipeline.main --pdf                rebuild out/document.pdf from the session
  python -m pipeline.main --new                start a new session (old one is kept)

Live keys (click the window first):
  SPACE  read the line in view          a / ENTER  accept the line
  e      edit the LaTeX, then accept    r          reject, back to live view
  m      mode: auto -> text -> math     u          undo the last accepted line
  p      build and open the PDF         s          save this frame to data/raw
  q / ESC  quit (builds the PDF)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from pipeline.classify import Classifier
from pipeline.latexdoc import Session, check_line, open_file, preflight
from pipeline.preprocess import orient, to_gray
from pipeline.recognize import draw_result, recognize
from pipeline.symbols import load_symbols

MODES = ("auto", "text", "math")


def edit_text(text: str) -> str:
    """Ask for a corrected line in the terminal, pre-filled where possible."""
    try:
        import readline
        readline.set_startup_hook(lambda: readline.insert_text(text))
        try:
            return input("edit> ").strip() or text
        finally:
            readline.set_startup_hook()
    except (ImportError, AttributeError):
        print(f"current: {text}")
        return input("new (empty = keep): ").strip() or text


def accept(session: Session, latex: str, mode: str, frame=None) -> bool:
    if not latex:
        print("nothing to accept")
        return False
    ok, msg = check_line(latex, mode)
    if not ok:
        print("this line does not compile; press e to edit it\n" + msg)
        return False
    image = None
    if frame is not None:
        import cv2
        frames = Path(config.OUT_DIR) / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        image = str(frames / f"{len(session.lines) + 1:04d}.png")
        cv2.imwrite(image, frame)
    session.add(latex, mode, image)
    print(f"accepted line {len(session.lines)}: [{mode}] {latex}")
    return True


def build(session: Session, show: bool = True) -> None:
    if not session.lines:
        print("no lines yet")
        return
    ok, pdf, msg = session.build_pdf()
    if ok:
        print(f"PDF: {pdf}")
        if show:
            open_file(pdf)
    else:
        print("PDF build failed:\n" + msg)


def describe(result) -> str:
    flagged = [s.cls for s in result.symbols if s.flagged]
    extra = f"  check: {', '.join(flagged)}" if flagged else ""
    notes = f"  ({'; '.join(result.notes)})" if result.notes else ""
    return f"[{result.mode}] {result.latex}{extra}{notes}"


def run_images(paths, clf, session, mode, do_accept, show):
    import cv2
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{p}: cannot read")
            continue
        res = recognize(img, clf, mode=mode)
        print(f"{p}: {describe(res)}")
        if show:
            cv2.imshow("result", draw_result(orient(img), res))
            cv2.waitKey(0)
        if do_accept:
            accept(session, res.latex, res.mode, img)
    if show:
        cv2.destroyAllWindows()


def _panel(view: np.ndarray, lines: list[str]) -> np.ndarray:
    import cv2
    h = 22 * len(lines) + 10
    bar = np.full((h, view.shape[1], 3), 30, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(bar, t[:95], (8, 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1,
                    cv2.LINE_AA)
    return np.vstack([view, bar])


def run_live(clf, session, mode, port):
    import cv2
    from pipeline.capture import Camera, FrameGrabber
    cam = Camera(port).open()
    print("board answers: %dx%d, %d byte(s) per pixel" % cam.ping())
    grabber = FrameGrabber(cam)
    grabber.start()
    result, frozen, accepted_now = None, None, 0
    help_line = "SPACE read  a accept  e edit  r reject  m mode  u undo  p pdf  s save  q quit"
    try:
        while True:
            if frozen is None:
                frame, err = grabber.latest()
                if frame is None:
                    view = np.full((480, 640, 3), 40, np.uint8)
                else:
                    g = orient(frame)
                    view = cv2.cvtColor(cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                                        cv2.COLOR_GRAY2BGR)
                status = err or f"live  frames {grabber.count}"
                text = [f"mode {mode}   lines {len(session.lines)}   {status}", help_line]
            else:
                view = draw_result(orient(frozen), result)
                text = [f"mode {result.mode}: {result.latex}",
                        "; ".join(result.notes)[:95], help_line]
            cv2.imshow("camera to latex", _panel(view, text))
            k = cv2.waitKey(30) & 0xFF
            if k == 255:
                continue
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                frame, _ = grabber.latest()
                if frame is not None:
                    frozen = frame.copy()
                    grabber.paused = True
                    result = recognize(frozen, clf, mode=mode)
                    print(describe(result))
            elif k in (ord("a"), 13, 10) and frozen is not None:
                if accept(session, result.latex, result.mode, frozen):
                    accepted_now += 1
                    frozen, grabber.paused = None, False
            elif k == ord("e") and frozen is not None:
                new = edit_text(result.latex)
                if accept(session, new, result.mode, frozen):
                    accepted_now += 1
                    frozen, grabber.paused = None, False
            elif k == ord("r"):
                frozen, grabber.paused = None, False
            elif k == ord("m"):
                mode = MODES[(MODES.index(mode) + 1) % len(MODES)]
                print(f"mode: {mode}")
                if frozen is not None:
                    result = recognize(frozen, clf, mode=mode)
                    print(describe(result))
            elif k == ord("u"):
                ln = session.undo()
                print(f"removed: {ln['latex']}" if ln else "nothing to undo")
            elif k == ord("p"):
                build(session)
            elif k == ord("s"):
                frame = frozen if frozen is not None else grabber.latest()[0]
                if frame is not None:
                    raw = Path(config.RAW_DIR)
                    raw.mkdir(parents=True, exist_ok=True)
                    path = raw / f"frame_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"saved {path}")
    finally:
        grabber.stop()
        cv2.destroyAllWindows()
        cam.close()
    if accepted_now:
        build(session)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", nargs="+", type=Path, help="read saved photos instead of the board")
    ap.add_argument("--mode", choices=MODES, default="auto")
    ap.add_argument("--model", default=config.MODEL_NAME)
    ap.add_argument("--port")
    ap.add_argument("--accept", action="store_true", help="with --image: add every line to the session")
    ap.add_argument("--show", action="store_true", help="with --image: show each result")
    ap.add_argument("--pdf", action="store_true", help="build the PDF from the session")
    ap.add_argument("--new", action="store_true", help="archive the current session and start a new one")
    args = ap.parse_args(argv)

    session = Session.load()
    if args.new:
        old = session.archive()
        print(f"old session saved as {old}" if old else "starting a new session")
    for problem in preflight():
        print("warning:", problem)
    if args.pdf and not args.image:
        build(session)
        return 0
    try:
        clf = Classifier.load(args.model)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    for p in clf.check_table(load_symbols()):
        print("warning:", p)
    print(f"model {args.model} ({clf.backend}, {len(clf.classes)} classes)")
    if args.image:
        run_images(args.image, clf, session, args.mode, args.accept, args.show)
        if args.pdf:
            build(session)
        return 0
    from pipeline.capture import CaptureError
    try:
        run_live(clf, session, args.mode, args.port)
    except CaptureError as exc:
        print(f"camera problem: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
