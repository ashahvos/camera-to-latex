from pipeline.latexnorm import normalize_tokens, same_line, token_error_rate


def test_spacing_and_braces_do_not_matter():
    assert same_line("x^2 + y^2", "x^{2}+y^{2}")
    assert same_line("\\frac12", "\\frac{1}{2}")
    assert same_line("a\\le b", "a\\leq b")


def test_degree_and_prime_forms():
    assert same_line("90^\\circ", "90^{\\circ}")
    assert same_line("f^{\\prime}(x)", "f'(x)")


def test_text_mode_spaces_matter():
    assert not same_line("ab cd", "abcd", "text")
    assert same_line("ab  cd", "ab cd", "text")


def test_token_error_rate():
    assert token_error_rate("x^{2}", "x^{2}") == 0
    assert abs(token_error_rate("x^{3}", "x^{2}") - 0.2) < 1e-9
    assert normalize_tokens("\\{1\\}") == ["\\{", "1", "\\}"]
