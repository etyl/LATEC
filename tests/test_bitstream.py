import numpy as np
import pytest

from latec.bitstream import Header, pack_stage, read_stream, unpack_stage, write_stream


def make_header(**kw):
    base = dict(
        channels=3,
        src_dtype=0,
        spatial=(100, 130),
        eb=2.0,
        levels=3,
        anchor_stride=8,
        anchor_block=4,
        radius=1 << 15,
        max_radius=64,
        agg_level=2,
        vmin=0.0,
        vmax=255.0,
        ckpt_hash=bytes(range(16)),
        flags=1,
    )
    base.update(kw)
    return Header(**base)


def test_header_roundtrip():
    h = make_header()
    h2 = Header.unpack(h.pack())
    assert h == h2


def test_stream_roundtrip():
    h = make_header()
    payload = b"payload" * 100
    stream = write_stream(h, payload)
    h2, p2 = read_stream(stream)
    assert h2 == h
    assert p2 == payload


def test_stream_roundtrip_nd():
    h = make_header(spatial=(12, 20, 24, 28))
    stream = write_stream(h, b"x" * 33)
    h2, p2 = read_stream(stream)
    assert h2.spatial == (12, 20, 24, 28)
    assert p2 == b"x" * 33


def test_bad_magic():
    with pytest.raises(ValueError):
        Header.unpack(b"NOTDEEP!" + b"\0" * 100)


def test_empty_spatial_shape_rejected():
    with pytest.raises(ValueError, match="spatial shape"):
        make_header(spatial=()).pack()


def test_stage_roundtrip():
    rng = np.random.RandomState(0)
    codes = rng.randint(0, 1000, 500).astype(np.uint32)
    codes[::17] = 0
    outliers = rng.randn(int((codes == 0).sum())).astype(np.float32)
    buf = pack_stage(codes, outliers) + pack_stage(
        np.zeros(0, np.uint32), np.zeros(0, np.float32)
    )
    c2, o2, off = unpack_stage(buf, 0)
    assert np.array_equal(c2, codes)
    assert np.array_equal(o2, outliers)
    c3, o3, off = unpack_stage(buf, off)
    assert len(c3) == 0 and len(o3) == 0 and off == len(buf)
