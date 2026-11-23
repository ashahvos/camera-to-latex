"""Turn classified symbols into one line of LaTeX.

The classifier only says which *class* each symbol is. This stage decides:
  * structure: fractions (from the bars segmentation marked), square roots,
    limits above/below big operators, superscripts and subscripts;
  * which member of a look-alike class was printed (o / O / 0, l / I / |,
    c / C, . / cdot, - / _, ...), from size, position and neighbours;
  * whether the line is text or math (mode "auto"), and the final string.

Positions are measured in capital heights ("cap") above a fitted baseline.
Every symbol is tried as each height type its class allows, at normal,
superscript and subscript level; the reading with the lowest cost wins.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

import config
from .classify import refine_scale
from .symbols import SymbolTable, load_symbols
from .types import Box, Symbol


# ------------------------------------------------------------- templates ---
def type_template(t: str, rx: float) -> tuple[float, float]:
    """(centre above baseline, height) of a normal-size symbol, in caps."""
    if t == "cap":
        return 0.5, 1.0
    if t == "x":
        return rx / 2.0, rx
    if t == "desc":
        return (rx - 0.3) / 2.0, rx + 0.3
    if t == "tall":
        return 0.37, 1.45
    if t == "op":
        return 0.37, 0.55
    if t == "raised":
        return 0.82, 0.35
    if t == "dot":
        return 0.03, 0.18
    if t == "line":
        return 0.37, 0.07
    if t == "big":
        return 0.37, 1.5
    raise ValueError(t)


_SIZE_TOL = {"op": 0.6, "line": 50.0, "dot": 0.6, "raised": 0.5}
_NO_SCRIPTS = {"raised", "dot"}
FUNCTIONS = {"sin", "cos", "tan", "log", "ln", "exp", "lim", "max", "min", "det",
             "sec", "csc", "cot", "sinh", "cosh", "tanh", "arcsin", "arccos", "arctan"}


# -------------------------------------------------------------- structures ---
@dataclass
class Metrics:
    cap: float          # capital height in pixels
    rx: float           # x-height / capital height
    base_a: float       # baseline y at x = 0 (image pixels)
    base_b: float = 0.0  # baseline slope

    def base(self, x: float) -> float:
        return self.base_a + self.base_b * x

    def pos(self, box: Box) -> tuple[float, float]:
        """(centre above the baseline, height), in capital heights."""
        return (self.base(box.cx) - box.cy) / self.cap, box.h / self.cap


@dataclass
class Group:
    els: list
    m: Metrics
    level: int = 0      # 0 normal, +1 superscript, -1 subscript (relative to m)


@dataclass
class El:
    kind: str                       # "sym", "frac" or "sqrt"
    box: Box
    sym: Symbol | None = None
    num: Group | None = None
    den: Group | None = None
    arg: Group | None = None
    sup: Group | None = None
    sub: Group | None = None
    htype: str | None = None        # chosen height type
    member: str | None = None       # chosen member name
    sup_raw: list = field(default_factory=list)
    sub_raw: list = field(default_factory=list)

    @property
    def cls(self):
        return self.sym.cls if self.sym is not None else None

    @property
    def full_box(self) -> Box:
        b = self.box
        for g in (self.sup, self.sub):
            if g is not None and g.els:
                b = b.union(Box.union_all(e.full_box for e in g.els))
        return b


# ---------------------------------------------------------------- metrics ---
def _cap_hint(symbols, fallback: float) -> float:
    c = refine_scale(symbols) if symbols else None
    return float(c) if c else float(fallback)


def _metrics(els, cap: float, table: SymbolTable) -> Metrics:
    """Fit the baseline (and refine cap height and x-height ratio).

    Symbols whose class has one height type each 'vote' for where the
    baseline would be if they sat on it. Every vote is tried: the winner is
    the baseline under which the whole group is explained most cheaply, each
    symbol read as normal, superscript or subscript (scripts cost extra, the
    leftmost symbol cannot be one). So a subscript run such as the 'max' of
    lambda_max cannot outvote its base. A line is then fitted through the
    agreeing votes to follow a slightly tilted page.
    """
    voters = []
    for e in els:
        if e.kind != "sym" or e.cls is None:
            continue
        hs = table.class_heights(e.cls)
        if len(hs) == 1:
            t = next(iter(hs))
            if t not in ("dot", "raised"):
                voters.append((e, t))
    rx = 0.66
    if not voters:
        ys = [e.box.y1 for e in els] or [0]
        return Metrics(cap, rx, float(np.median(ys)))
    syms = [e for e in els if e.kind == "sym" and e.cls is not None]
    first = min(els, key=lambda e: e.box.x0)
    a, b = 0.0, 0.0
    for _ in range(2):
        cx = np.array([e.box.cx for e, _ in voters])
        votes = np.array([e.box.cy + type_template(t, rx)[0] * cap for e, t in voters])
        tol = 0.15 * cap
        best, best_cost = None, float("inf")
        for v in np.unique(np.round(votes, 1)):
            m = Metrics(cap, rx, float(v))
            total = sum(min(_best_reading(e, m, table, allow_scripts=e is not first)[0], 6.0) for e in syms)
            if total < best_cost - 1e-9:
                best, best_cost = float(v), total
        inl = np.abs(votes - best) <= tol
        a, b = float(np.median(votes[inl])), 0.0
        if inl.sum() >= 3 and np.ptp(cx[inl]) >= 2 * cap:
            b, a = np.polyfit(cx[inl], votes[inl], 1)
            if abs(b) > 0.1:
                a, b = float(np.median(votes[inl])), 0.0
        xs = [e.box.h for (e, t), ok in zip(voters, inl) if ok and t == "x"]
        caps = [e.box.h for (e, t), ok in zip(voters, inl) if ok and t == "cap"]
        if caps:
            c2 = float(np.median(caps))
            if 0.75 * cap <= c2 <= 1.35 * cap:
                cap = c2
        if xs:
            rx = float(np.clip(np.median(xs) / cap, 0.55, 0.8))
    return Metrics(cap, rx, float(a), float(b))


# --------------------------------------------------------- hypotheses ---
def _local_pos(e: El, g: Group) -> tuple[float, float]:
    """Position in the frame of the group's own level (scripts scaled back up)."""
    c, h = g.m.pos(e.box)
    if g.level > 0:
        c = (c - config.SUP_SHIFT) / config.SCRIPT_SIZE
        h = h / config.SCRIPT_SIZE
    elif g.level < 0:
        c = (c + config.SUB_SHIFT) / config.SCRIPT_SIZE
        h = h / config.SCRIPT_SIZE
    return c, h


def _best_reading(e: El, m: Metrics, table: SymbolTable, allow_scripts: bool):
    """(cost, height type, level) of the best-fitting reading of one symbol."""
    c, h = m.pos(e.box)
    best = None
    for t in sorted(table.class_heights(e.cls)):
        c0, h0 = type_template(t, m.rx)
        size_tol = _SIZE_TOL.get(t, config.SIZE_TOL)
        levels = (0, 1, -1) if allow_scripts and t not in _NO_SCRIPTS else (0,)
        for lv in levels:
            if lv == 0:
                cp, hp, prior = c0, h0, 0.0
            elif lv > 0:
                cp, hp = config.SUP_SHIFT + config.SCRIPT_SIZE * c0, config.SCRIPT_SIZE * h0
                prior = config.SCRIPT_PENALTY
            else:
                cp, hp = -config.SUB_SHIFT + config.SCRIPT_SIZE * c0, config.SCRIPT_SIZE * h0
                prior = config.SCRIPT_PENALTY
            cost = (abs(c - cp) / config.POS_TOL
                    + abs(math.log(max(h, 1e-3) / hp)) / size_tol + prior)
            if best is None or cost < best[0]:
                best = (cost, t, lv)
    return best


# ---------------------------------------------------------------- structure ---
def _fraction_parts(els, bar: Symbol, cap: float):
    tol = 0.15 * cap
    reach = 4.0 * cap
    x0, x1 = bar.box.x0 - tol, bar.box.x1 + tol
    above, below = [], []
    for e in els:
        if not (x0 <= e.box.cx <= x1):
            continue
        if e.box.y1 <= bar.box.y0 + 2 and bar.box.y0 - e.box.y1 <= reach:
            above.append(e)
        elif e.box.y0 >= bar.box.y1 - 2 and e.box.y0 - bar.box.y1 <= reach:
            below.append(e)
    return above, below


def _as_minus(bar: Symbol) -> Symbol:
    bar.kind = "symbol"
    bar.cls, bar.conf, bar.margin = "bar", 1.0, 1.0
    return bar


def _build(symbols, bars, cap: float, table: SymbolTable, features: int) -> Group:
    """Symbols (and fraction bars) of one group -> laid-out Group."""
    els = [El("sym", s.box, sym=s) for s in symbols if s.cls is not None]
    bars = sorted(bars, key=lambda b: -b.box.w)
    if features < 3:
        els += [El("sym", b.box, sym=_as_minus(b)) for b in bars]
        bars = []
    used = set()
    for bar in bars:
        if id(bar) in used:
            continue
        others = [b for b in bars if b is not bar and id(b) not in used]
        cand = els + [El("bar", b.box, sym=b) for b in others]
        above, below = _fraction_parts(cand, bar, cap)
        if not above or not below:
            continue
        used.add(id(bar))
        parts = above + below
        for e in parts:
            if e.kind == "bar":
                used.add(id(e.sym))
        taken = {id(e) for e in parts if e.kind != "bar"}
        els = [e for e in els if id(e) not in taken]

        def sub(part):
            syms = [e.sym for e in part if e.kind == "sym"]
            sub_bars = [e.sym for e in part if e.kind == "bar"]
            return _build(syms, sub_bars, _cap_hint(syms, cap), table, features)

        box = Box.union_all([bar.box] + [e.box for e in parts])
        els.append(El("frac", box, sym=bar, num=sub(above), den=sub(below)))
    for bar in bars:
        if id(bar) not in used:
            els.append(El("sym", bar.box, sym=_as_minus(bar)))
    if features >= 3:
        els = _group_roots(els, cap, table)
    return _finish(els, cap, table, features)


def _group_roots(els, cap: float, table: SymbolTable):
    """A radical with ink under its bar becomes \\sqrt{...}."""
    roots = sorted([e for e in els if e.kind == "sym" and e.cls == "sqrt"], key=lambda e: -e.box.w)
    taken = set()
    for r in roots:
        if id(r) in taken:
            continue
        hook = r.box.x0 + 0.45 * r.box.h
        inside = [o for o in els if o is not r and id(o) not in taken
                  and hook < o.box.cx < r.box.x1 + 2 and r.box.y0 < o.box.cy < r.box.y1]
        if not inside:
            continue
        for o in inside:
            taken.add(id(o))
        arg = _finish(inside, cap, table, 3)
        r.kind, r.arg = "sqrt", arg
    return [e for e in els if id(e) not in taken]


def _attach_limits(els, m: Metrics, table: SymbolTable, features: int):
    """Limits printed above/below a big operator (display style)."""
    if features < 3:
        return els
    taken = set()
    for e in els:
        if e.kind != "sym" or e.cls not in ("upper_Sigma", "upper_Pi", "int", "oint"):
            continue
        if e.box.h < config.BIG_OPERATOR * m.cap:
            continue
        tol = 0.2 * m.cap
        above = [o for o in els if o is not e and id(o) not in taken and o.box.y1 <= e.box.y0 + 2
                 and e.box.x0 - tol <= o.box.cx <= e.box.x1 + tol]
        below = [o for o in els if o is not e and id(o) not in taken and o.box.y0 >= e.box.y1 - 2
                 and e.box.x0 - tol <= o.box.cx <= e.box.x1 + tol]
        for grp, attr in ((above, "sup"), (below, "sub")):
            if grp:
                for o in grp:
                    taken.add(id(o))
                syms = [o.sym for o in grp if o.kind == "sym"]
                sub_m = _metrics(grp, _cap_hint(syms, 0.7 * m.cap), table)
                setattr(e, attr, _finish_with(grp, sub_m, table, features))
    return [e for e in els if id(e) not in taken]


def _finish(els, cap: float, table: SymbolTable, features: int) -> Group:
    m = _metrics(els, cap, table)
    els = _attach_limits(els, m, table, features)
    return _finish_with(els, m, table, features)


def _finish_with(els, m: Metrics, table: SymbolTable, features: int) -> Group:
    """Pick each symbol's reading and hang scripts on their bases."""
    els = sorted(els, key=lambda e: e.box.x0)
    out, base = [], None
    for e in els:
        lv = 0
        if e.kind == "sym":
            _, e.htype, lv = _best_reading(e, m, table, allow_scripts=(base is not None and features >= 2))
        if lv != 0 and base is not None:
            (base.sup_raw if lv > 0 else base.sub_raw).append(e)
        else:
            out.append(e)
            base = e
    for e in out:
        if e.sup_raw:
            e.sup = Group(e.sup_raw, m, +1)
        if e.sub_raw:
            e.sub = Group(e.sub_raw, m, -1)
        e.sup_raw, e.sub_raw = [], []
    return Group(out, m, 0)


# ------------------------------------------------------------------ mode ---
def _walk(group: Group):
    for e in group.els:
        yield e
        for g in (e.num, e.den, e.arg, e.sup, e.sub):
            if g is not None:
                yield from _walk(g)


def decide_mode(group: Group, table: SymbolTable) -> str:
    for e in _walk(group):
        if e.kind in ("frac", "sqrt") or e.sup is not None or e.sub is not None:
            return "math"
        if e.kind == "sym":
            members = table.members_of(e.cls)
            if all(mm.math_only for mm in members):
                return "math"
            if e.cls in ("equal", "plus", "less", "greater"):
                return "math"
    return "text"


# --------------------------------------------------------------- members ---
_LETTER_CLASSES = None


def _letter_classes(table: SymbolTable) -> set:
    global _LETTER_CLASSES
    if _LETTER_CLASSES is None:
        _LETTER_CLASSES = {m.cls for m in table.member_list
                           if len(m.latex_text) == 1 and m.latex_text.isalpha()}
    return _LETTER_CLASSES


def _looks_digit(e: El | None) -> bool:
    return (e is not None and e.kind == "sym"
            and (e.cls.startswith("digit_") or (e.cls == "o0" and e.htype == "cap")))


def _looks_letter(e: El | None, table: SymbolTable) -> bool:
    if e is None or e.kind != "sym":
        return False
    if e.cls == "o0":
        return e.htype == "x"
    if e.cls == "xX":
        return e.htype != "op"
    if e.cls == "lI":
        return e.htype == "cap"
    return e.cls in _letter_classes(table)


def _looks_upper(e: El | None) -> bool:
    if e is None or e.kind != "sym":
        return False
    if e.cls.startswith("upper_") or e.cls.startswith("bb"):
        return True
    return e.cls in ("cC", "sS", "vV", "wW", "xX", "zZ") and e.htype == "cap"


def _looks_lower(e: El | None) -> bool:
    if e is None or e.kind != "sym":
        return False
    if e.cls.startswith("lower_"):
        return True
    return e.cls in ("cC", "sS", "vV", "wW", "xX", "zZ", "o0") and e.htype == "x"


def _words(group: Group) -> list[list[int]]:
    """Indexes of group.els split where the gap is wider than SPACE_GAP."""
    words, cur, prev = [], [], None
    for i, e in enumerate(group.els):
        if prev is not None and e.full_box.x0 - prev.full_box.x1 > config.SPACE_GAP * group.m.cap:
            words.append(cur)
            cur = []
        cur.append(i)
        prev = e
    if cur:
        words.append(cur)
    return words


def _pick_member(e: El, i: int, group: Group, word: list[int], mode: str, table: SymbolTable) -> str:
    members = table.members_of(e.cls)
    if len(members) == 1:
        return members[0].name
    c, h = _local_pos(e, group)
    els = group.els
    prev = els[i - 1] if i > 0 else None
    nxt = els[i + 1] if i + 1 < len(els) else None
    others = [els[k] for k in word if k != i]
    cls, t = e.cls, e.htype

    if cls in ("cC", "sS", "vV", "wW", "zZ"):
        lower, upper = members[0].name, members[1].name
        return lower if t == "x" else upper
    if cls == "xX":
        gap_l = (e.box.x0 - prev.full_box.x1) / group.m.cap if prev else 0
        gap_r = (nxt.full_box.x0 - e.box.x1) / group.m.cap if nxt else 0
        prev_ok = _looks_digit(prev) or (prev is not None and prev.cls == "rparen")
        next_ok = _looks_digit(nxt) or (nxt is not None and nxt.cls == "lparen")
        if prev_ok and next_ok and gap_l > 0.1 and gap_r > 0.1:
            return "times"
        return "lower_x" if h < (group.m.rx + 1.0) / 2.0 else "upper_X"
    if cls == "o0":
        if t == "raised":
            return "degree"
        if t == "x":
            return "lower_o"
        if mode == "math":
            if any(_looks_upper(o) for o in (prev, nxt)) and not any(_looks_digit(o) for o in (prev, nxt)):
                return "upper_O"
            return "digit_0"
        if any(_looks_digit(o) for o in others) and not any(_looks_letter(o, table) for o in others):
            return "digit_0"
        if any(_looks_letter(o, table) for o in others):
            return "upper_O"
        return "digit_0" if e.box.w < 0.72 * e.box.h else "upper_O"
    if cls == "lI":
        if t == "tall" or c < 0.37 - 0.1:
            return "vert"
        if mode == "math":
            return "upper_I" if any(_looks_upper(o) for o in (prev, nxt)) else "lower_l"
        if any(_looks_lower(o) for o in others):
            return "lower_l"
        return "upper_I"
    if cls == "dot":
        return "cdot" if t == "op" or c > 0.2 else "period"
    if cls == "comma":
        if t == "raised":
            return "prime" if mode == "math" else "apostrophe"
        return "comma"
    if cls == "bar":
        return "underscore" if c < 0.08 else "minus"
    if cls == "star":
        return "star"
    if cls == "tilde":
        return "sim" if mode == "math" else "tilde"
    if cls == "lower_f":
        return "f_math" if mode == "math" and t == "tall" else "lower_f"
    if cls in ("upper_Sigma", "upper_Pi"):
        big = e.box.h >= config.BIG_OPERATOR * group.m.cap * (config.SCRIPT_SIZE if group.level else 1.0)
        names = {m.height: m.name for m in members}
        return names["big"] if big else names["cap"]
    same = [m for m in members if m.height == t]
    return (same or members)[0].name


PAIRS = {("less", "less"): "ll", ("greater", "greater"): "gg", ("vert", "vert"): "parallel"}


def _combine_pairs(group: Group) -> None:
    """<< , >> and the parallel sign print as two pieces: join them back."""
    out = []
    for e in group.els:
        prev = out[-1] if out else None
        if (prev is not None and prev.kind == "sym" and e.kind == "sym" and prev.sup is None
                and prev.sub is None and (prev.member, e.member) in PAIRS
                and e.box.x0 - prev.box.x1 <= 0.15 * group.m.cap
                and prev.box.y_overlap(e.box) >= 0.7 * min(prev.box.h, e.box.h)):
            prev.member = PAIRS[(prev.member, e.member)]
            prev.box = prev.box.union(e.box)
            prev.sup, prev.sub = e.sup, e.sub
            prev.sym.member = e.sym.member = prev.member
            continue
        out.append(e)
    group.els = out


def _type_for(e: El, group: Group, table: SymbolTable) -> str:
    """Best height type for e's class at its current level (after a class change)."""
    c, h = _local_pos(e, group)
    best = None
    for t in table.class_heights(e.cls):
        c0, h0 = type_template(t, group.m.rx)
        cost = abs(c - c0) / config.POS_TOL + abs(math.log(max(h, 1e-3) / h0)) / _SIZE_TOL.get(t, config.SIZE_TOL)
        if best is None or cost < best[0]:
            best = (cost, t)
    return best[1]


def _is_digit_class(cls: str) -> bool:
    return cls.startswith("digit_") or cls == "o0"


def _rerank(group: Group, mode: str, slot: bool, table: SymbolTable) -> None:
    """Use neighbours to settle letter/digit look-alikes the classifier hesitated on.

    A letter-like symbol whose second guess is a digit (>= 15 %) becomes that
    digit next to other digits, or alone in a fraction part or script (>= 25 %):
    l -> 1, theta -> 0. In text mode a digit inside a word of letters becomes
    its letter alternative the same way.
    """
    els = group.els
    for i, e in enumerate(els):
        if e.kind != "sym" or not e.sym.alternatives:
            continue
        prev = els[i - 1] if i > 0 else None
        nxt = els[i + 1] if i + 1 < len(els) else None
        alts = [(c, p) for c, p in e.sym.alternatives[1:]]
        new = None
        if not _is_digit_class(e.cls):
            digits = [(c, p) for c, p in alts if _is_digit_class(c)]
            if digits:
                c, p = digits[0]
                near_digit = _looks_digit(prev) or _looks_digit(nxt)
                if (near_digit and p >= 0.15) or (slot and len(els) == 1 and p >= 0.25):
                    new = c
        elif mode == "text":
            letters = [(c, p) for c, p in alts if c in _letter_classes(table)]
            if letters and _looks_letter(prev, table) and _looks_letter(nxt, table) and letters[0][1] >= 0.15:
                new = letters[0][0]
        if new is not None:
            e.sym.cls = new
            e.htype = _type_for(e, group, table)


def _resolve(group: Group, mode: str, table: SymbolTable, slot: bool = False) -> None:
    _rerank(group, mode, slot, table)
    words = _words(group)
    word_of = {i: w for w in words for i in w}
    for i, e in enumerate(group.els):
        if e.kind == "sym":
            e.member = _pick_member(e, i, group, word_of.get(i, [i]), mode, table)
            e.sym.member = e.member
        for g in (e.num, e.den, e.arg, e.sup, e.sub):
            if g is not None:
                _resolve(g, mode, table, slot=g is not e.arg)
    _combine_pairs(group)


# ------------------------------------------------------------- serializing ---
_CMD_END = re.compile(r"\\[A-Za-z]+$")


def _join_math(parts: list[str]) -> str:
    out = ""
    for p in parts:
        if out and p and _CMD_END.search(out) and p[0].isalpha():
            out += " "
        out += p
    return out


def _sym_math(e: El, table: SymbolTable) -> str:
    if e.member == "dquote":
        return "''"
    return table.members[e.member].latex_math


def _degree_only(g: Group | None) -> bool:
    return g is not None and len(g.els) == 1 and g.els[0].kind == "sym" and g.els[0].cls == "o0"


def _el_math(e: El, table: SymbolTable) -> str:
    if e.kind == "frac":
        s = "\\frac{" + _group_math(e.num, table) + "}{" + _group_math(e.den, table) + "}"
    elif e.kind == "sqrt":
        s = "\\sqrt{" + _group_math(e.arg, table) + "}"
    else:
        s = _sym_math(e, table)
    if e.sub is not None and e.sub.els:
        s += "_{" + _group_math(e.sub, table) + "}"
    if e.sup is not None and e.sup.els:
        if _degree_only(e.sup) and _looks_digit(e):
            s += "^{\\circ}"
        else:
            s += "^{" + _group_math(e.sup, table) + "}"
    return s


def _plain_lower(e: El, table: SymbolTable) -> str | None:
    if e.kind != "sym" or e.sup is not None or e.sub is not None:
        return None
    lm = table.members[e.member].latex_math
    return lm if len(lm) == 1 and lm.islower() else None


def _group_math(group: Group, table: SymbolTable, functions: bool = False) -> str:
    parts = []
    els = group.els
    i = 0
    while i < len(els):
        if functions and _plain_lower(els[i], table):
            j = i
            word = ""
            while j < len(els) and _plain_lower(els[j], table):
                if j > i and els[j].box.x0 - els[j - 1].box.x1 > 0.3 * group.m.cap:
                    break
                word += _plain_lower(els[j], table)
                j += 1
            if word in FUNCTIONS:
                parts.append("\\" + word)
                i = j
                continue
        parts.append(_el_math(els[i], table))
        i += 1
    return _join_math(parts)


def _group_text(group: Group, table: SymbolTable) -> str:
    out, prev = "", None
    for e in group.els:
        if prev is not None and e.full_box.x0 - prev.full_box.x1 > config.SPACE_GAP * group.m.cap:
            out += " "
        if e.kind == "sym" and e.sup is None and e.sub is None:
            out += table.members[e.member].latex("text")
        else:
            out += "\\ensuremath{" + _el_math(e, table) + "}"
        prev = e
    return out


# ------------------------------------------------------------------ entry ---
@dataclass
class Layout:
    latex: str
    mode: str
    group: Group


def layout_line(symbols, mode: str = "auto", scale: float | None = None,
                table: SymbolTable | None = None, features: int | None = None) -> Layout:
    """Classified symbols of one line -> LaTeX (without $ or \\[ \\])."""
    table = table or load_symbols()
    features = config.LAYOUT_LEVEL if features is None else features
    syms = [s for s in symbols if s.kind == "symbol" and s.cls is not None]
    bars = [s for s in symbols if s.kind == "fracbar"]
    if not syms and not bars:
        return Layout("", "text" if mode == "auto" else mode, Group([], Metrics(1, 0.66, 0)))
    cap = float(scale) if scale else _cap_hint(syms, float(np.median([s.box.h for s in syms or bars])))
    if mode == "text":
        features = 1            # text lines: no scripts or fractions
    group = _build(syms, bars, cap, table, features)
    if mode == "auto":
        mode = decide_mode(group, table)
    _resolve(group, mode, table)
    if mode == "math":
        latex = _group_math(group, table, functions=True)
    else:
        latex = _group_text(group, table)
    return Layout(latex, mode, group)
