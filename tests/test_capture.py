import io

import numpy as np
import pytest

from pipeline.capture import CaptureError, encode_frame, read_frame


def test_round_trip_with_junk_before_header():
    img = (np.arange(320 * 240) % 251).astype(np.uint8).reshape(240, 320)
    stream = io.BytesIO(b"hello from the board\r\n" + encode_frame(img))
    out = read_frame(stream, timeout=1)
    assert out.shape == (240, 320)
    assert np.array_equal(out, img)


def test_bad_checksum_is_rejected():
    img = np.full((4, 5), 7, np.uint8)
    data = bytearray(encode_frame(img))
    data[-1] ^= 0xFF
    with pytest.raises(CaptureError):
        read_frame(io.BytesIO(bytes(data)), timeout=0.5)


def test_board_error_line_is_reported():
    with pytest.raises(CaptureError, match="ERR camera"):
        read_frame(io.BytesIO(b"ERR camera\r\n"), timeout=0.5)


def test_truncated_frame_times_out():
    img = np.zeros((10, 10), np.uint8)
    data = encode_frame(img)[:-20]
    with pytest.raises(CaptureError):
        read_frame(io.BytesIO(data), timeout=0.2)
