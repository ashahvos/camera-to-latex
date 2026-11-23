import numpy as np

from pipeline.segment import find_blobs, mark_fraction_bars, merge_parts, segment, select_line


def canvas(h=120, w=200):
    return np.zeros((h, w), np.uint8)


def test_i_dot_and_stem_are_one_symbol():
    b = canvas()
    b[40:43, 50:53] = 1          # dot
    b[46:60, 50:53] = 1          # stem
    b[40:60, 80:92] = 1          # a capital-sized block next to it
    b[40:60, 100:112] = 1
    syms, scale, _ = segment(b)
    assert len(syms) == 3
    assert syms[0].box.h == 20


def test_equals_sign_is_one_symbol():
    b = canvas()
    b[40:60, 20:32] = 1
    b[47:49, 40:56] = 1
    b[52:54, 40:56] = 1
    b[40:60, 64:76] = 1
    syms, _, _ = segment(b)
    assert len(syms) == 3


def test_fraction_bar_is_marked_and_not_merged():
    b = canvas()
    b[30:46, 60:64] = 1          # numerator
    b[50:52, 52:72] = 1          # bar
    b[56:72, 58:68] = 1          # denominator
    b[42:58, 90:102] = 1         # something on the main line
    syms, scale, _ = segment(b)
    kinds = sorted(s.kind for s in syms)
    assert kinds.count("fracbar") == 1
    assert len(syms) == 4


def test_two_lines_keep_the_middle_one():
    b = canvas(240, 200)
    for x in (20, 40, 60):
        b[10:30, x:x + 10] = 1      # line near the top
        b[110:130, x:x + 10] = 1    # line in the middle
    labels, blobs, _ = find_blobs(b)
    kept = select_line(blobs, 20.0, b.shape)
    assert len(kept) == 3
    assert all(k.box.y0 >= 100 for k in kept)


def test_display_fraction_rows_are_joined():
    b = canvas(240, 200)
    b[80:100, 90:100] = 1       # numerator, its own row
    b[110:112, 80:112] = 1      # bar, 10 px below
    b[122:142, 90:102] = 1      # denominator, 10 px below the bar
    labels, blobs, _ = find_blobs(b)
    kept = select_line(blobs, 20.0, b.shape)
    assert len(kept) == 3


def test_edge_ink_sets_cut_off():
    b = canvas()
    b[40:60, 0:8] = 1           # touches the left edge
    b[40:60, 30:42] = 1
    b[40:60, 50:62] = 1
    syms, _, cut = segment(b)
    assert cut and len(syms) == 2


def test_ticks_of_double_quote_join():
    b = canvas()
    for x in (20, 50, 90):
        b[40:60, x:x + 12] = 1
    b[38:45, 70:72] = 1
    b[38:45, 74:76] = 1
    labels, blobs, _ = find_blobs(b)
    groups = merge_parts(blobs, 20.0)
    assert len(groups) == 4
