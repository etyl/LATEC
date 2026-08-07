import numpy as np
import pytest

from latec.quantizer import dequantize, quantize


@pytest.mark.parametrize(
    "scale,eb", [(1.0, 0.01), (255.0, 2.0), (1e6, 1e-3), (1e-3, 1e-7)]
)
def test_bound_holds_random(scale, eb):
    rng = np.random.RandomState(42)
    x = (rng.randn(10000) * scale).astype(np.float32)
    pred = x + (rng.randn(10000) * eb * 10).astype(np.float32)
    codes, outliers = quantize(x, pred, eb)
    rec = dequantize(pred, codes, outliers, eb)
    assert np.abs(x - rec).max() <= eb


def test_bound_at_tie_edges():
    eb = 0.1
    pred = np.zeros(9, np.float32)
    # residuals exactly at odd multiples of eb (rounding ties)
    x = (np.arange(-4, 5) * 2 + 1).astype(np.float32) * eb
    codes, outliers = quantize(x, pred, eb)
    rec = dequantize(pred, codes, outliers, eb)
    assert np.abs(x - rec).max() <= eb


def test_large_pred_absorption_demotes_to_outlier():
    # |pred| >> eb: pred + 2*eb*q loses the correction in float32
    eb = 1e-3
    x = np.full(100, 1e7, np.float32) + np.linspace(0, 1, 100, dtype=np.float32)
    pred = np.full(100, 1e7, np.float32)
    codes, outliers = quantize(x, pred, eb)
    rec = dequantize(pred, codes, outliers, eb)
    assert np.abs(x - rec).max() <= eb


def test_radius_overflow_becomes_outlier():
    eb = 0.5
    x = np.array([0.0, 1e6], np.float32)  # residual 1e6 -> q = 1e6 >> radius
    pred = np.zeros(2, np.float32)
    codes, outliers = quantize(x, pred, eb, radius=1 << 15)
    assert codes[1] == 0 and len(outliers) == 1
    rec = dequantize(pred, codes, outliers, eb, radius=1 << 15)
    assert np.abs(x - rec).max() <= eb


def test_round_output_integer_domain():
    # eb=1.5: float error can be 1.5, which rounds to integer error 2
    eb = 1.5
    x = np.arange(0, 256, dtype=np.float32)
    pred = x + 1.5
    codes, outliers = quantize(x, pred, eb, round_output=True)
    rec = np.rint(dequantize(pred, codes, outliers, eb))
    assert np.abs(x - rec).max() <= eb


def test_encoder_recon_matches_decoder():
    rng = np.random.RandomState(0)
    x = rng.rand(1000).astype(np.float32) * 100
    pred = x + rng.randn(1000).astype(np.float32)
    eb = 0.05
    codes, outliers = quantize(x, pred, eb)
    r1 = dequantize(pred, codes, outliers, eb)
    r2 = dequantize(pred.copy(), codes.copy(), outliers.copy(), eb)
    assert np.array_equal(r1.view(np.uint32), r2.view(np.uint32))


def test_invalid_eb():
    with pytest.raises(ValueError):
        quantize(np.zeros(1, np.float32), np.zeros(1, np.float32), 0.0)
