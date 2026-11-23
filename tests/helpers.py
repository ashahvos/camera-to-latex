"""Shared helpers for the tests: hand-built symbols and a stub classifier."""
import numpy as np

from pipeline.classify import apply_predictions
from pipeline.symbols import load_symbols
from pipeline.types import Box, Symbol


def sym(cls, x0, bottom, w, h, conf=0.99, alt=None):
    """A classified symbol whose box ends at `bottom` (image y grows downward)."""
    table = load_symbols()
    box = Box(int(x0), int(bottom - h), int(x0 + w), int(bottom))
    s = Symbol(box=box, mask=np.ones((box.h, box.w), bool))
    p = np.zeros(table.num_classes, np.float32)
    p[table.class_index[cls]] = conf
    if alt:
        p[table.class_index[alt[0]]] = alt[1]
    apply_predictions([s], p[None], table.classes)
    return s


def bar(x0, y, w, h=2):
    box = Box(int(x0), int(y), int(x0 + w), int(y + h))
    return Symbol(box=box, mask=np.ones((box.h, box.w), bool), kind="fracbar")


class StubClassifier:
    """Answers with a fixed class list, left to right (for plumbing tests)."""

    def __init__(self, answers):
        self.classes = load_symbols().classes
        self.answers = list(answers)
        self.calls = 0

    def predict(self, crops):
        n = len(crops)
        out = np.zeros((n, len(self.classes)), np.float32)
        for i in range(n):
            name = self.answers[min(i, len(self.answers) - 1)]
            out[i, self.classes.index(name)] = 1.0
        self.calls += 1
        return out
