"""Project-wide settings.

Every script reads its tunable numbers from here, so a change made after the
ruler test or on the dev lines only has to be made once.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Serial link and camera ---------------------------------------------------
SERIAL_PORT = None          # e.g. "/dev/cu.usbmodem14101", "/dev/ttyACM0", "COM3"; None = auto-detect
SERIAL_BAUD = 115200        # the Nano's native USB ignores this number
SERIAL_TIMEOUT_S = 5.0      # per frame
SERIAL_RETRIES = 3
FRAME_W, FRAME_H = 320, 240

# Fix the photo's orientation once the first live frame is in (bring-up step 3).
FLIP_H = False
FLIP_V = False
ROTATE_DEG = 0              # 0, 90, 180 or 270, applied after the flips
ROI = None                  # (x, y, w, h) inside the oriented photo, or None for all of it

# --- Segmentation (tune on the dev lines) --------------------------------------
BLUR_KSIZE = 3              # Gaussian blur before thresholding; 0 or 1 = off
THRESH_BLOCK = 31           # adaptive-threshold block size, odd, close to one symbol's size
THRESH_C = 10               # raise if paper texture shows up as ink
THRESH_WEAK_C = 5           # hysteresis: fainter ink kept where it touches ink (None = off)
OPEN_KSIZE = 0              # morphological opening (removes speckle); 0 = off
CLOSE_KSIZE = 0             # morphological closing (joins broken strokes); 0 = off
MIN_BLOB_AREA = 4           # pixels; a 3-4 px period has about 9-12
MAX_BLOB_FRACTION = 0.6     # blobs bigger than this share of the photo are dropped
MERGE_OVERLAP = 0.5         # horizontal overlap, as a share of the narrower blob
MERGE_MAX_HEIGHT = 1.5      # merged box height limit, in line scales (j with its dot: about 1.3)
FRACTION_BAR_ASPECT = 3.0   # a fraction bar is at least this many times wider than tall
FRACTION_NEIGHBOR_MIN = 0.4 # blobs above and below a bar must be this tall, in line scales
FRACTION_MAX_GAP = 1.0      # and no farther from the bar than this, in line scales
DEFAULT_SCALE = 24.0        # line scale used when a photo has too few symbols to measure

# --- Crops and classifier ---------------------------------------------------------
CROP_SIZE = 32
WINDOW_SCALE = 1.5          # crop window side = max(1.5 line scales, symbol size + 4 px)
CONF_MIN = 0.6              # flag a symbol below this top probability...
MARGIN_MIN = 0.2            # ...or when the top-2 gap is below this
REPAIR = True               # retry uncertain symbols: join broken pieces, cut touching letters
REPAIR_CONF = 0.8           # a repair is kept only if the classifier is at least this sure

# --- Layout -----------------------------------------------------------------------------
# Lengths are in capital heights ("line scales"). Each symbol is tested as normal,
# superscript and subscript, and the reading that fits its size and position best wins.
SPACE_GAP = 0.4             # a gap wider than this is a space between words (text mode)
SCRIPT_SIZE = 0.7           # scripts are printed this much smaller (LaTeX: 0.7)
SUP_SHIFT = 0.53            # a superscript's baseline sits this far above the line's baseline
SUB_SHIFT = 0.22            # a subscript's baseline sits this far below it
SCRIPT_PENALTY = 1.0        # how much better a script reading must fit before it is used
POS_TOL = 0.15              # position error that costs 1 point
SIZE_TOL = 0.2              # size error (log ratio) that costs 1 point
BIG_OPERATOR = 1.3          # Sigma/Pi taller than this become \sum/\prod
LAYOUT_LEVEL = 3            # 1 = plain line, 2 = + scripts, 3 = + fractions and roots

# --- Paths ---------------------------------------------------------------------------------
SYMBOLS_CSV = ROOT / "symbols.csv"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SHEETS_DIR = DATA_DIR / "sheets"
CROPS_REAL_DIR = DATA_DIR / "crops_real"
SYNTH_DIR = DATA_DIR / "synthetic"
EXTERNAL_DIR = DATA_DIR / "external"
DOWNLOADS_DIR = DATA_DIR / "downloads"
FONTS_DIR = DATA_DIR / "fonts"          # drop extra .ttf/.otf files here for synthetic data
MODELS_DIR = ROOT / "models"
MODEL_NAME = "symbols_v1"
OUT_DIR = ROOT / "out"
TEMPLATE_TEX = ROOT / "template" / "document.tex"
TEST_SET_DIR = ROOT / "test_set"
RESULTS_DIR = ROOT / "results"

# --- LaTeX ------------------------------------------------------------------------------------
PDFLATEX = "pdflatex"
COMPILE_TIMEOUT_S = 20
