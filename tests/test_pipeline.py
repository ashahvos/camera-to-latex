"""Photo -> LaTeX plumbing with a stub classifier (no trained model needed)."""
import numpy as np

from pipeline.recognize import recognize

from helpers import StubClassifier


def test_recognize_draws_boxes_and_lays_out():
    img = np.full((240, 320), 210, np.uint8)
    img[110:130, 100:112] = 30      # "1"
    img[110:130, 130:142] = 30      # "2"
    img[100:110, 150:157] = 30      # superscript "3"
    clf = StubClassifier(["digit_1", "digit_2", "digit_3"])
    res = recognize(img, clf, mode="math", oriented=True)
    assert res.latex == "12^{3}"
    assert res.mode == "math"
    assert len(res.symbols) == 3
