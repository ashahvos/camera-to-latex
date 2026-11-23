# Camera to LaTeX

Point the EE 105 board (Arduino Nano 33 BLE Sense Lite + OV7675 camera) at a printed
line of text or math, press SPACE, and get the line as LaTeX; accepted lines are
collected into a PDF. The board only takes pictures. Recognition runs on the laptop:

```
board --USB--> capture --> segment --> classify --> layout --> LaTeX --> PDF
(320x240 gray)  (FRM1)    (blobs)     (CNN, 159    (scripts,   (check)   (pdflatex)
                                       classes)    fractions)
```

Full step-by-step instructions: **docs/instructions.pdf**. Design choices: the slide deck.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Python 3.10-3.13
pip install -r requirements.txt
python -m pytest -q                                      # 45 tests, no board needed

python -m model.make_synthetic                           # synthetic crops (about 1 min)
python -m model.train                                    # CNN (TensorFlow); or --backend sklearn
python -m tools.make_test_lines --render                 # fake photos of the test lines
python -m tools.eval_pipeline --images test_set/synthetic --split all
```

With the board (flash `firmware/capture/capture.ino` first):

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
| Test lines | `python -m tools.make_test_lines --pdf --names`, photograph | 60 test + 20 dev lines in `test_set/lines.tsv` |

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
docs/                   instructions.pdf
```

Generated and ignored by git: `data/`, `models/`, `out/`, `results/`, `test_set/synthetic/`.
To share a trained model, `git add -f models/symbols_v1.keras models/symbols_v1.json`.

## Status

Tested here without the board and without TensorFlow: unit tests pass, and on fake
photos of the 60 test lines (24 px capitals) the sklearn fallback model trained on
synthetic data alone reads 70 % of lines exactly (80 % at 30 px capitals). The Keras CNN
and the real-crop data are expected to raise this; measure with `tools/eval_pipeline.py`.
