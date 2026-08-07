import numpy as np

from latec.codec import compress, decompress
from latec.predictor import InterpPredictor


def _normalized_smooth(h=72, w=80):
    yy, xx = np.mgrid[0:h, 0:w]
    img = 0.5 + 0.35 * np.sin(xx / 9.0) * np.cos(yy / 11.0)
    img += 0.05 * np.sin((xx + yy) / 5.0)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _psnr_unit(img, rec):
    mse = float(np.mean((img.astype(np.float64) - rec.astype(np.float64)) ** 2))
    return 10 * np.log10(1.0 / mse) if mse else float("inf")


def test_stage_diagnostics_account_for_total_float_distortion():
    img = _normalized_smooth()
    eb = 2.0 / 255.0
    kw = dict(levels=4, anchor_stride=8, anchor_block=4)

    stream, stats = compress(img, eb, InterpPredictor("cubic", **kw), **kw)
    rec = decompress(stream)

    assert rec.dtype == np.float32
    assert np.array_equal(stats["recon"], rec)
    assert np.abs(img - rec).max() <= eb
    assert sum(stats["stage_codes"]) == img.size
    assert sum(stats["stage_outliers"]) == stats["outliers"]
    assert sum(stats["stage_payload_bytes"]) > 0

    total_sse = float(np.square(img.astype(np.float64) - rec.astype(np.float64)).sum())
    assert np.isclose(sum(stats["stage_recon_sse"]), total_sse)
    assert max(stats["stage_recon_max"]) <= eb


def test_uint8_style_eb_is_too_coarse_for_normalized_float_images():
    img = _normalized_smooth()
    kw = dict(levels=4, anchor_stride=8, anchor_block=4)

    coarse_stream, _ = compress(img, 1.0, InterpPredictor("cubic", **kw), **kw)
    coarse = decompress(coarse_stream)
    fine_stream, _ = compress(img, 1.0 / 255.0, InterpPredictor("cubic", **kw), **kw)
    fine = decompress(fine_stream)

    assert _psnr_unit(coarse, img) < 15.0
    assert _psnr_unit(fine, img) > 45.0
    assert len(coarse_stream) < len(fine_stream)


def test_tighter_coarse_level_eb_reduces_propagated_interp_distortion():
    """Interp reuses coarse reconstructions as predictors for finer levels.
    Tightening the coarse-level error budget spends a few more bits, but it
    should reduce the final reconstruction MSE on smooth data."""
    img = _normalized_smooth(96, 96)
    eb = 4.0 / 255.0
    kw = dict(levels=4, anchor_stride=16, anchor_block=1)

    flat_stream, _ = compress(
        img, eb, InterpPredictor("cubic", **kw), **kw, eb_ratio=1.0
    )
    flat = decompress(flat_stream)
    tight_stream, _ = compress(
        img, eb, InterpPredictor("cubic", **kw), **kw, eb_ratio=0.7
    )
    tight = decompress(tight_stream)

    flat_mse = float(np.mean((img.astype(np.float64) - flat.astype(np.float64)) ** 2))
    tight_mse = float(np.mean((img.astype(np.float64) - tight.astype(np.float64)) ** 2))
    assert tight_mse < flat_mse
    assert len(tight_stream) >= len(flat_stream)
