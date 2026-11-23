"""Compare LaTeX strings fairly: tokenize, normalize, edit distance.

Two strings count as the same line when they typeset the same symbols in the
same structure, whatever the spacing, brace style or synonym used:
  x^2+y^2  ==  x^{2} + y^{2}        \\le  ==  \\leq        \\frac12 == \\frac{1}{2}
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|[{}^_]|\s+|.", re.S)

ALIASES = {
    "\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq", "\\to": "\\rightarrow",
    "\\gets": "\\leftarrow", "\\lbrace": "\\{", "\\rbrace": "\\}", "\\vert": "|",
    "\\mid": "|", "\\lvert": "|", "\\rvert": "|", "\\lnot": "\\neg", "\\land": "\\wedge",
    "\\lor": "\\vee", "\\dfrac": "\\frac", "\\tfrac": "\\frac", "\\varnothing": "\\emptyset",
    "\\textasciitilde": "~", "\\textbackslash": "\\backslash", "\\textless": "<",
    "\\textgreater": ">", "\\textbar": "|", "\\textperiodcentered": "\\cdot",
    "\\texttimes": "\\times", "\\textdegree": "\\circ", "\\textquotedbl": '"',
    "\\textasciicircum": "^", "\\ast": "*", "\\prime": "'",
}
DROP = {"\\,", "\\;", "\\:", "\\!", "\\ ", "\\quad", "\\qquad", "\\left", "\\right",
        "\\displaystyle", "\\textstyle", "\\ensuremath", "\\text", "\\mathrm",
        "\\bigl", "\\bigr", "\\Bigl", "\\Bigr", "$", "\\(", "\\)", "\\[", "\\]"}
TWO_ARGS = {"\\frac"}
ONE_ARG = {"\\sqrt", "\\mathbb"}


def tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s)


def _group_end(toks, i):
    """Index after the brace group starting at toks[i] == '{'."""
    depth = 0
    for j in range(i, len(toks)):
        if toks[j] == "{":
            depth += 1
        elif toks[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(toks)


def _arg(toks, i):
    """(argument tokens without outer braces, next index) for one argument at i."""
    if i >= len(toks):
        return [], i
    if toks[i] == "{":
        j = _group_end(toks, i)
        return toks[i + 1:j - 1], j
    return [toks[i]], i + 1


def _canon(toks: list[str]) -> list[str]:
    """Rebuild a token list with every script/argument braced exactly once."""
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t in ("^", "_"):
            arg, i = _arg(toks, i + 1)
            arg = _canon(arg)
            out += [t, "{"] + arg + ["}"]
            continue
        if t in TWO_ARGS or t in ONE_ARG:
            n = 2 if t in TWO_ARGS else 1
            out.append(t)
            i += 1
            for _ in range(n):
                arg, i = _arg(toks, i)
                out += ["{"] + _canon(arg) + ["}"]
            continue
        if t == "{":
            j = _group_end(toks, i)
            inner = _canon(toks[i + 1:j - 1])
            out += inner          # plain grouping braces carry no meaning here
            i = j
            continue
        out.append(t)
        i += 1
    return out


def normalize_tokens(s: str, mode: str = "math") -> list[str]:
    toks = tokenize(s)
    out = []
    for t in toks:
        if t.isspace():
            if mode == "text" and out and out[-1] != " ":
                out.append(" ")
            continue
        t = ALIASES.get(t, t)
        if t in DROP:
            continue
        out.append(t)
    while out and out[-1] == " ":
        out.pop()
    while out and out[0] == " ":
        out.pop(0)
    # '^{\circ}' and '\circ' (text degree) compare equal; f^{'} == f'
    out = _canon(out)
    res = []
    i = 0
    while i < len(out):
        if out[i] == "^" and out[i + 1:i + 4] in (["{", "\\circ", "}"], ["{", "'", "}"]):
            res.append(out[i + 2])
            i += 4
            continue
        res.append(out[i])
        i += 1
    return [t for t in res if t != " " or mode == "text"]


def normalize(s: str, mode: str = "math") -> str:
    return "".join(normalize_tokens(s, mode))


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def token_error_rate(pred: str, expected: str, mode: str = "math") -> float:
    p = normalize_tokens(pred, mode)
    e = normalize_tokens(expected, mode)
    return edit_distance(p, e) / max(1, len(e))


def same_line(pred: str, expected: str, mode: str = "math") -> bool:
    return normalize_tokens(pred, mode) == normalize_tokens(expected, mode)
