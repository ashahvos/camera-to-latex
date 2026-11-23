"""The symbol table (symbols.csv): every symbol the project knows.

Each row is one *member* (a symbol as it is written in LaTeX). Members that
look alike on paper (o, O and 0; l, I and |; c and C; ...) share one *class*:
the classifier only ever picks a class, and pipeline/layout.py picks the
member from size, position and neighbours.

Columns
  member      unique name, also used for sheet cells and crop file names
  class       what the classifier predicts (folder name under data/crops_real)
  height      where the symbol sits on the line:
                cap    capital height, on the baseline (A, 7, b, d)
                x      x-height, on the baseline (a, x, alpha)
                desc   x-height with a descender (g, p, y, eta)
                tall   ascender and descender (parentheses, j, beta, |)
                op     centred on the math axis (+, =, <, arrows)
                raised high on the line (' " ^ degree)
                dot    small, on the baseline (. ,)
                line   thin horizontal stroke (- _)
                big    big operators (sum, prod, int)
  latex_text  how to write it in a text-mode line ('' = math only)
  latex_math  how to write it in a math-mode line
  render      JSON list of ways to draw it for synthetic data:
                "t:<chars>"    text with ordinary font files (PIL)
                "m:<mathtext>" matplotlib mathtext, rendered as $<mathtext>$
                "r:<mathtext>" mathtext, keeping only the biggest piece of ink
                               (a radical drawn over an argument, argument removed)
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import config

HEIGHT_TYPES = ("cap", "x", "desc", "tall", "op", "raised", "dot", "line", "big")


@dataclass(frozen=True)
class Member:
    name: str
    cls: str
    height: str
    latex_text: str
    latex_math: str
    render: tuple

    @property
    def math_only(self) -> bool:
        return self.latex_text == ""

    def latex(self, mode: str) -> str:
        """LaTeX for this member in a 'text' or 'math' line."""
        if mode == "math":
            return self.latex_math
        if self.latex_text:
            return self.latex_text
        return "\\ensuremath{" + self.latex_math + "}"


class SymbolTable:
    def __init__(self, members: list[Member]):
        self.member_list = list(members)
        self.members = {m.name: m for m in members}
        self.classes: list[str] = []
        self.class_members: dict[str, list[Member]] = {}
        for m in members:
            if m.cls not in self.class_members:
                self.classes.append(m.cls)
                self.class_members[m.cls] = []
            self.class_members[m.cls].append(m)
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        self.member_index = {m.name: i for i, m in enumerate(self.member_list)}

    def __len__(self):
        return len(self.member_list)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def members_of(self, cls: str) -> list[Member]:
        return self.class_members[cls]

    def default_member(self, cls: str) -> Member:
        return self.class_members[cls][0]

    def class_heights(self, cls: str) -> set[str]:
        return {m.height for m in self.class_members[cls]}

    def char_map(self) -> dict[str, Member]:
        """Single character -> member, from each member's first text render.

        Used to map character datasets (Chars74K, TMNIST) onto the table.
        """
        out: dict[str, Member] = {}
        for m in self.member_list:
            for spec in m.render:
                if spec.startswith("t:") and len(spec) == 3:
                    out.setdefault(spec[2:], m)
                    break
        return out

    def latex_map(self) -> dict[str, Member]:
        """latex_math string -> member (first member wins)."""
        out: dict[str, Member] = {}
        for m in self.member_list:
            out.setdefault(m.latex_math, m)
        return out


def _parse_rows(rows) -> list[Member]:
    members = []
    for r in rows:
        render = tuple(json.loads(r["render"]))
        if r["height"] not in HEIGHT_TYPES:
            raise ValueError(f"{r['member']}: unknown height type {r['height']!r}")
        members.append(Member(r["member"], r["class"], r["height"],
                              r["latex_text"], r["latex_math"], render))
    return members


@lru_cache(maxsize=4)
def load_symbols(path: str | Path | None = None) -> SymbolTable:
    path = Path(path or config.SYMBOLS_CSV)
    with open(path, newline="", encoding="utf-8") as f:
        members = _parse_rows(csv.DictReader(f))
    names = [m.name for m in members]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate member names in {path}: {dup}")
    return SymbolTable(members)
