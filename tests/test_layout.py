"""Layout on hand-placed symbols. Capital height 20 px, baseline at y = 100."""
from pipeline.latexnorm import same_line
from pipeline.layout import layout_line

from helpers import bar, sym

S = 20


def run(symbols, mode="math"):
    return layout_line(symbols, mode=mode, scale=S)


def test_superscript():
    out = run([sym("xX", 10, 100, 12, 13), sym("digit_2", 23, 89, 9, 14),
               sym("plus", 36, 98, 12, 11), sym("digit_1", 52, 100, 8, 20)])
    assert out.latex == "x^{2}+1"


def test_subscript():
    out = run([sym("lower_a", 10, 100, 11, 13), sym("digit_1", 22, 104, 7, 14)])
    assert out.latex == "a_{1}"


def test_text_words_and_spaces():
    out = run([sym("lower_a", 10, 100, 11, 13), sym("lower_b", 22, 100, 11, 20),
               sym("cC", 44, 100, 10, 13), sym("lower_d", 55, 100, 11, 20)], mode="text")
    assert out.latex == "ab cd"


def test_fraction():
    out = run([sym("digit_1", 20, 89, 8, 16), bar(10, 92, 30), sym("digit_2", 20, 113, 9, 16)])
    assert out.latex == "\\frac{1}{2}"


def test_zero_and_o_from_context():
    digits = run([sym("digit_1", 10, 100, 8, 20), sym("o0", 22, 100, 11, 20),
                  sym("digit_5", 36, 100, 10, 20)], mode="text")
    assert digits.latex == "105"
    word = run([sym("upper_H", 10, 100, 14, 20), sym("lower_e", 26, 100, 10, 13),
                sym("lI", 38, 100, 4, 20), sym("lI", 44, 100, 4, 20), sym("o0", 50, 100, 11, 13)],
               mode="text")
    assert word.latex == "Hello"


def test_degree_sign():
    out = run([sym("digit_9", 10, 100, 11, 20), sym("o0", 22, 100, 12, 20), sym("o0", 36, 82, 7, 7)])
    assert out.latex == "90^{\\circ}"


def test_prime():
    out = run([sym("lower_f", 10, 104, 9, 26), sym("comma", 20, 84, 3, 6)])
    assert out.latex == "f'"


def test_times_between_digits():
    out = run([sym("digit_2", 10, 100, 9, 20), sym("xX", 26, 97, 11, 11), sym("digit_3", 42, 100, 9, 20)])
    assert same_line(out.latex, "2\\times 3")


def test_two_less_signs_make_ll():
    out = run([sym("lower_a", 0, 100, 11, 13), sym("less", 16, 98, 9, 11), sym("less", 24, 98, 9, 11),
               sym("lower_b", 40, 100, 11, 20)])
    assert out.latex == "a\\ll b"


def test_auto_mode():
    assert run([sym("alpha", 10, 100, 12, 13)], mode="auto").mode == "math"
    assert run([sym("lower_a", 10, 100, 11, 13), sym("lower_b", 22, 100, 11, 20)], mode="auto").mode == "text"


def test_digit_context_fixes_l_versus_1():
    out = run([sym("digit_2", 10, 100, 9, 20), sym("lI", 22, 100, 6, 20, conf=0.6, alt=("digit_1", 0.35)),
               sym("digit_3", 32, 100, 9, 20)])
    assert out.latex == "213"


def test_command_followed_by_letter_gets_space():
    out = run([sym("upper_Delta", 10, 100, 14, 20), sym("lower_t", 26, 100, 7, 18)])
    assert out.latex == "\\Delta t"
