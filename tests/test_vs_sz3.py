"""Verify the interp codec against the original SZ3 (pysz / sz3 CLI).

Skipped when no SZ3 implementation is installed. Both are ABS error-bounded, so
the real check is that our rate is competitive: on smooth data our cubic-interp
pipeline matches or beats SZ3, so we assert we stay within 1.3x of its size.
"""

import numpy as np
import pytest

from latec.baselines import sz3_roundtrip
from latec.codec import compress, decompress
from latec.predictor import InterpPredictor
from tests.test_codec_mock import smooth_image


@pytest.mark.parametrize("shape,eb", [((128, 128, 3), 4.0), ((160, 120, 1), 2.0)])
def test_matches_sz3(shape, eb):
    h, w, c = shape
    img = smooth_image(h, w, c)
    if c == 1:
        img = img[..., 0]

    sz3 = sz3_roundtrip(img, eb)
    if sz3 is None:
        pytest.skip("no SZ3 implementation available")
    sz3_bytes, sz3_rec = sz3

    predictor = InterpPredictor("cubic", levels=4, anchor_stride=16, anchor_block=1)
    stream, _ = compress(img, eb, predictor)
    rec = decompress(stream)

    err = lambda r: np.abs(img.astype(np.int64) - r.astype(np.int64)).max()
    assert err(rec) <= eb  # our bound holds
    assert err(sz3_rec) <= eb  # SZ3's bound holds (sanity)
    assert len(stream) <= 1.3 * sz3_bytes  # rate competitive with SZ3
