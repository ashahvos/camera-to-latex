import json

from pipeline.symbols import HEIGHT_TYPES, load_symbols


def test_table_loads():
    t = load_symbols()
    assert len(t) >= 180
    assert t.num_classes == 159


def test_class_names_are_safe_folder_names():
    t = load_symbols()
    lowered = [c.lower() for c in t.classes]
    assert len(set(lowered)) == len(lowered), "two classes differ only by case (breaks macOS folders)"
    for c in t.classes:
        assert c.replace("_", "").isalnum(), c


def test_every_member_has_math_latex_and_renders():
    t = load_symbols()
    for m in t.member_list:
        assert m.latex_math, m.name
        assert m.height in HEIGHT_TYPES
        assert m.render, m.name
        for spec in m.render:
            assert spec[:2] in ("t:", "m:", "r:"), (m.name, spec)


def test_lookalikes_share_a_class():
    t = load_symbols()
    assert t.members["lower_o"].cls == t.members["digit_0"].cls == t.members["upper_O"].cls
    assert t.members["lower_l"].cls == t.members["upper_I"].cls == t.members["vert"].cls
    assert t.members["period"].cls == t.members["cdot"].cls


def test_char_map_covers_printable_ascii():
    t = load_symbols()
    cmap = t.char_map()
    missing = [chr(c) for c in range(33, 127) if chr(c) not in cmap]
    assert not missing, missing


def test_latex_member_text_and_math():
    t = load_symbols()
    assert t.members["hash"].latex("text") == "\\#"
    assert t.members["alpha"].latex("text") == "\\ensuremath{\\alpha}"
    assert t.members["alpha"].latex("math") == "\\alpha"
