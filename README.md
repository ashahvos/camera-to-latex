# Camera to LaTeX

Point the EE 105 board (Arduino Nano 33 BLE Sense Lite + OV7675 camera) at a printed
line of text or math, press SPACE, and get the line as LaTeX; accepted lines are
collected into a PDF. The board only takes pictures. Recognition runs on the laptop:

```
board --USB--> capture --> segment --> classify --> layout --> LaTeX --> PDF
(320x240 gray)  (FRM1)    (blobs)     (CNN, 159    (scripts,   (check)   (pdflatex)
                                       classes)    fractions)
```

Step-by-step instructions, written for a reader who starts with an empty Windows laptop:

- **docs/setup_guide.pdf**: GitHub account, PowerShell, installation, quick start (no board needed).
- **docs/tasks_guide.pdf**: the board and the camera stand, the symbol sheets and retraining,
  tuning and the freeze, the final measurement, the demo video, the report and the rehearsal.

`docs/instructions.pdf` is the earlier, short version and is superseded by these two.
Design choices: the slide deck.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Python 3.10-3.13
pip install -r requirements.txt
python -m pytest -q                                      # 48 tests, no board needed

python -m model.make_synthetic                           # synthetic crops (about 1 min)
python -m model.train                                    # CNN (TensorFlow); or --backend sklearn
python -m tools.make_test_lines --render                 # fake photos of the test lines
python -m tools.eval_pipeline --images test_set/synthetic --split all
```

With the board (flash `firmware/capture/capture.ino` first; it must contain the line
`#define CAMERA_MODEL  OV7675`, without which the library sets the sensor up as an OV7670):

```bash
python -m pipeline.capture --ping                        # "board answers: 320x240"
python -m pipeline.capture --soak 100                    # gate G1: 100/100
python -m pipeline.main                                  # the app (keys below)
```

App keys: `SPACE` read the line, `a`/`ENTER` accept, `e` edit, `r` reject, `m` mode
(auto/text/math), `u` undo, `p` build and open the PDF, `s` save the frame, `q` quit.
Without the board: `python -m pipeline.main --image photo.png --accept --pdf`.

## Data

| Source | Command | Notes |
|---|---|---|
| Synthetic | `python -m model.make_synthetic` | every symbol in about 30 fonts + 5 mathtext sets, fake camera |
| Our sheets | `python -m tools.make_sheets`, photograph, `python -m tools.crop_sheet` | 18 blocks x 3 photos, about 9 real crops per symbol |
| Public sets | `python -m tools.download_datasets --list`, `python -m tools.import_datasets ...` | optional: Chars74K, TMNIST-94, HASYv2, ... |
| Test lines | `python -m tools.make_test_lines --pdf --names`, photograph | 60 test + 20 dev lines in `test_set/lines.tsv`; `--font-pt 48` prints them larger |

Photograph a list with `python -m pipeline.capture --series DIR --names-file FILE`
(sheets: `data/sheets` + `sheets/photo_names.txt`; test lines: `test_set/images` +
`test_set/photo_names.txt`).

## Repository

```
config.py               every tunable number (serial port, thresholds, layout)
symbols.csv             159 classes / 182 members: class, height type, LaTeX, how to draw it
firmware/capture/       Arduino sketch: grayscale QVGA frames over USB
pipeline/               capture, preprocess, segment, classify, layout, latexnorm, latexdoc, recognize, main
model/                  make_synthetic, train, evaluate, export_tflite (optional, revision 2)
tools/                  make_sheets, crop_sheet, make_test_lines, eval_pipeline, download/import_datasets
sheets/                 printable data sheets + layout.json (generated, committed)
test_set/               lines.tsv, test_lines.pdf, images/ (your photos)
template/document.tex   preamble of the output PDF
tests/                  pytest suite (runs in CI without TensorFlow or a board)
docs/                   setup_guide.pdf, tasks_guide.pdf and their LaTeX sources; instructions.pdf (superseded)
```

Generated and ignored by git: `data/`, `models/`, `out/`, `results/`, `test_set/synthetic/`.
To share a trained model, `git add -f models/symbols_v1.keras models/symbols_v1.json`.

## Results

One row per measurement; counts are lines read exactly. Add a row by copying the last one.

| Date | Version | Photographs | Capitals | Classifier | Plain | Scripts | Fractions | Mixed | All | Symbols, held-out real crops |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-23 | before v1.0-rc1 | 80 simulated (test + dev lines) | 24 px | CNN, synthetic data only | 27/38 | 14/23 | 10/12 | 4/7 | 55/80 | not measured |

Targets on the 60 real test lines: plain 20/25, scripts 14/20, fractions 6/10, and 95 % of
held-out real crops. The test lines are evaluated only at the gate and for the final
measurement (see docs/tasks_guide.pdf, Tasks 3 and 4).

## Status

Tested without the board: the unit tests pass, and on fake photos of the 60 test lines
(24 px capitals) the sklearn fallback model trained on synthetic data alone reads 70 % of
lines exactly (80 % at 30 px capitals). The Keras CNN trained on synthetic data reads 55 of
the 80 fake photos exactly (row above). Changes of 24 Nov 2026, not yet run on the board:
the sketch selects the OV7675, `tools/crop_sheet.py` tolerates a hollow corner square whose
hole has filled in and rejects photos with a second block in view, and
`tools/make_test_lines.py` has the option `--font-pt`.
