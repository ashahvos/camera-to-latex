import shutil

import pytest

from pipeline.latexdoc import Session, check_line, compile_lines, lines_to_tex


def test_lines_to_tex():
    tex = lines_to_tex([{"latex": "Hello", "mode": "text"}, {"latex": "x^{2}", "mode": "math"}])
    assert "\\noindent Hello\\par" in tex
    assert "\\[ x^{2} \\]" in tex


def test_session_add_undo(tmp_path):
    s = Session.load(tmp_path / "session.json")
    s.add("a+b", "math")
    s.add("Hi", "text")
    s2 = Session.load(tmp_path / "session.json")
    assert [ln["latex"] for ln in s2.lines] == ["a+b", "Hi"]
    assert s2.undo()["latex"] == "Hi"
    assert len(Session.load(tmp_path / "session.json").lines) == 1


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no pdflatex")
def test_compile_and_check(tmp_path):
    ok, pdf, msg = compile_lines([{"latex": "x^{2}+\\alpha", "mode": "math"},
                                  {"latex": "Score: 50\\%", "mode": "text"}], tmp_path)
    assert ok and pdf.exists(), msg
    assert check_line("\\frac{1}{2}", "math", tmp_path / "c")[0]
    assert not check_line("\\frac{1}{", "math", tmp_path / "c")[0]
