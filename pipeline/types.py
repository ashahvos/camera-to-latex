"""Small data types shared by the pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Box:
    """Pixel box; x1 and y1 are exclusive. Image y grows downward."""
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    def union(self, other: "Box") -> "Box":
        return Box(min(self.x0, other.x0), min(self.y0, other.y0),
                   max(self.x1, other.x1), max(self.y1, other.y1))

    def x_overlap(self, other: "Box") -> int:
        return max(0, min(self.x1, other.x1) - max(self.x0, other.x0))

    def y_overlap(self, other: "Box") -> int:
        return max(0, min(self.y1, other.y1) - max(self.y0, other.y0))

    def y_gap(self, other: "Box") -> int:
        """Vertical empty space between the boxes (0 if they overlap)."""
        return max(0, max(self.y0, other.y0) - min(self.y1, other.y1))

    def x_gap(self, other: "Box") -> int:
        return max(0, max(self.x0, other.x0) - min(self.x1, other.x1))

    @staticmethod
    def union_all(boxes) -> "Box":
        boxes = list(boxes)
        out = boxes[0]
        for b in boxes[1:]:
            out = out.union(b)
        return out


@dataclass
class Blob:
    """One connected component of ink (or several, once merged)."""
    box: Box
    area: int
    labels: list = field(default_factory=list)   # component ids in the label image
    kind: str = "symbol"                          # "symbol" or "fracbar"


@dataclass
class Symbol:
    """A segmented symbol and, after classification, what it is."""
    box: Box
    mask: np.ndarray                 # bool, shape (box.h, box.w): the symbol's own ink
    kind: str = "symbol"             # "symbol" or "fracbar"
    crop: np.ndarray | None = None   # CROP_SIZE x CROP_SIZE float32 in [0, 1]
    probs: np.ndarray | None = None  # class probabilities
    cls: str | None = None           # top class
    conf: float = 0.0                # top probability
    margin: float = 0.0              # top-1 minus top-2 probability
    alternatives: list = field(default_factory=list)  # [(class, prob), ...] top 3
    member: str | None = None        # set by layout
    flagged: bool = False            # low confidence: shown in red, worth checking


@dataclass
class LineResult:
    latex: str
    mode: str                        # "text" or "math"
    symbols: list
    scale: float                     # line scale (capital height) in pixels
    cut_off: bool = False            # ink touched the photo's edge and was dropped
    notes: list = field(default_factory=list)

    @property
    def flagged(self) -> list:
        return [i for i, s in enumerate(self.symbols) if s.flagged]
