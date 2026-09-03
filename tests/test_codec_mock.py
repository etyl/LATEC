"""Full-pipeline integration tests with a private test predictor."""

import numpy as np
import pytest

from latec.codec import _gather_stage, _scatter_stage, compress, decompress
from tests.helpers import NearestPredictor


def _decode(stream):
    return decompress(stream, lambda _header: NearestPredictor())


def smooth_image(h, w, c, seed=0):
    """Smooth-ish synthetic image so prediction has something to work with."""
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [
            128
            + 100 * np.sin(xx / (10 + 5 * k)) * np.cos(yy / (13 + 3 * k))
            + rng.randn(h, w) * 2
            for k in range(c)
        ],
        axis=-1,
    )
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize(
    "shape,eb",
    [
        ((130, 70, 3), 0.5),
        ((130, 70, 3), 2.0),
        ((100, 100, 1), 8.0),
        ((64, 64, 3), 2.0),
    ],
)
def test_roundtrip_bound(shape, eb):
    h, w, c = shape
    img = smooth_image(h, w, c)
    if c == 1:
        img = img[..., 0]
    stream, stats = compress(img, eb, NearestPredictor())
    rec = _decode(stream)
    assert rec.shape == img.shape and rec.dtype == img.dtype
    assert np.abs(img.astype(np.int64) - rec.astype(np.int64)).max() <= eb
    assert stats["ratio"] > 1.0


def test_deterministic_streams_and_output():
    img = smooth_image(100, 130, 3)
    s1, _ = compress(img, 2.0, NearestPredictor())
    s2, _ = compress(img, 2.0, NearestPredictor())
    assert s1 == s2
    r1 = _decode(s1)
    r2 = _decode(s1)
    assert np.array_equal(r1, r2)


def test_encoder_recon_matches_decoder_output():
    img = smooth_image(64, 64, 3, seed=5)
    stream, stats = compress(img, 1.0, NearestPredictor())
    rec = _decode(stream)
    assert np.array_equal(stats["recon"], rec)


def test_float_input():
    rng = np.random.RandomState(7)
    img = (rng.rand(70, 90).astype(np.float32) * 4).round(2)
    stream, _ = compress(img, 0.01, NearestPredictor())
    rec = _decode(stream)
    assert rec.dtype == np.float32
    assert np.abs(img - rec).max() <= 0.01


def test_unidentified_predictor_stream_requires_factory():
    img = smooth_image(64, 64, 3)
    stream, _ = compress(img, 2.0, NearestPredictor())
    with pytest.raises(ValueError, match="predictor_factory"):
        decompress(stream)


@pytest.mark.parametrize("shape", [(16, 12), (7, 5, 9), (5, 4, 6, 3)])
def test_stage_gather_scatter_match_boolean_indexing(shape):
    """The codec walks a stage by flat index rather than with the boolean mask
    (NumPy expands an n-D boolean index into one int64 array per axis, which at
    rank 5 was the codec's host peak). Both directions must be identical."""
    rng = np.random.RandomState(0)
    pos = rng.rand(*shape) < 0.4
    field = rng.rand(2, *shape).astype(np.float32)
    n = int(pos.sum())

    values = _gather_stage(field, pos, n)
    np.testing.assert_array_equal(values, field[:, pos])

    recon = np.zeros_like(field)
    _scatter_stage(recon, pos, values)
    expected = np.zeros_like(field)
    expected[:, pos] = values
    np.testing.assert_array_equal(recon, expected)
