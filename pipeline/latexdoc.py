"""The output document: accepted lines -> out/lines.tex -> out/document.pdf.

A session (out/session.json) keeps every accepted line, so the PDF can be
rebuilt at any time and a crash never loses work. Each line is compile-checked
on its own before it is accepted, so one bad line never breaks the document.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config

DEFAULT_TEMPLATE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\begin{document}
\input{lines}
\end{document}
"""


def line_to_tex(latex: str, mode: str) -> str:
    latex = latex.strip()
    if mode == "math":
        return "\\[ " + latex + " \\]"
    return "\\noindent " + latex + "\\par"


def lines_to_tex(lines) -> str:
    body = ["% Written by pipeline/latexdoc.py; edit out/session.json, not this file."]
    for ln in lines:
        body.append(line_to_tex(ln["latex"], ln["mode"]))
        body.append("")
    return "\n".join(body) + "\n"


def template_text() -> str:
    p = Path(config.TEMPLATE_TEX)
    return p.read_text() if p.exists() else DEFAULT_TEMPLATE


def _run_pdflatex(tex_name: str, cwd: Path, draft: bool = False):
    cmd = [config.PDFLATEX, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error"]
    if draft:
        cmd.append("-draftmode")
    cmd.append(tex_name)
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=config.COMPILE_TIMEOUT_S, errors="replace")
    except FileNotFoundError:
        return False, f"'{config.PDFLATEX}' not found: install TeX Live / MacTeX / MiKTeX"
    except subprocess.TimeoutExpired:
        return False, f"pdflatex took longer than {config.COMPILE_TIMEOUT_S} s"
    if r.returncode != 0:
        return False, _error_lines(r.stdout)
    return True, ""


def _error_lines(log: str) -> str:
    keep = [ln for ln in log.splitlines() if ln.startswith("!") or ":" in ln and ".tex:" in ln]
    return "\n".join(keep[:6]) or log[-800:]


def check_line(latex: str, mode: str, work_dir: Path | None = None):
    """(ok, message): does this one line compile with the document's preamble?"""
    work = Path(work_dir or Path(config.OUT_DIR) / ".check")
    work.mkdir(parents=True, exist_ok=True)
    (work / "lines.tex").write_text(line_to_tex(latex, mode) + "\n")
    (work / "check.tex").write_text(template_text())
    return _run_pdflatex("check.tex", work, draft=True)


def compile_lines(lines, out_dir: Path | None = None):
    """Write lines.tex + document.tex in out_dir and run pdflatex.

    Returns (ok, pdf path or None, message).
    """
    out = Path(out_dir or config.OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lines.tex").write_text(lines_to_tex(lines))
    (out / "document.tex").write_text(template_text())
    ok, msg = _run_pdflatex("document.tex", out)
    pdf = out / "document.pdf"
    return ok, (pdf if ok and pdf.exists() else None), msg


def preflight() -> list[str]:
    """Problems that would stop PDF output (empty list = ready)."""
    problems = []
    if shutil.which(config.PDFLATEX) is None:
        problems.append(f"'{config.PDFLATEX}' is not on PATH: install a TeX distribution")
        return problems
    ok, msg = check_line(r"x^{2}+\alpha\leq\mathbb{R}", "math")
    if not ok:
        problems.append("test document failed to compile (lmodern / amsmath / amssymb missing?):\n" + msg)
    ok, msg = check_line(r"Text \# \$ \% \& \_ \{ \} \textbackslash{} \textasciitilde{}", "text")
    if not ok:
        problems.append("text-mode test failed to compile:\n" + msg)
    return problems


def open_file(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # opening is a convenience only
        print(f"(could not open {path}: {exc})")


@dataclass
class Session:
    path: Path
    title: str = "Camera to LaTeX"
    lines: list = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "Session":
        path = Path(path or Path(config.OUT_DIR) / "session.json")
        if path.exists():
            data = json.loads(path.read_text())
            return cls(path, data.get("title", "Camera to LaTeX"), data.get("lines", []))
        return cls(path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"title": self.title, "lines": self.lines}, indent=2))
        tmp.replace(self.path)

    def add(self, latex: str, mode: str, image: str | None = None) -> None:
        self.lines.append({"latex": latex, "mode": mode, "image": image,
                           "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.save()

    def undo(self):
        if not self.lines:
            return None
        ln = self.lines.pop()
        self.save()
        return ln

    def build_pdf(self):
        return compile_lines(self.lines, self.path.parent)

    def archive(self) -> Path | None:
        """Move the current session aside so a new one can start."""
        if not self.path.exists():
            return None
        dest = self.path.with_name(f"session_{time.strftime('%Y%m%d_%H%M%S')}.json")
        self.path.replace(dest)
        self.lines = []
        return dest
