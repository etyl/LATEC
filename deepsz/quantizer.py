"""Linear-scaling quantization with an absolute error bound (classic SZ stage 2).

Codes are non-negative integers in [0, 2*radius):
  - code 0 is reserved for "unpredictable" values (outliers), whose exact
    float32 values are stored in a side array in scan order,
  - otherwise code = round((x - pred) / (2*eb)) + radius.

The error bound |x - dequantize(...)| <= eb holds unconditionally: after
computing the candidate reconstruction with the exact same arithmetic the
decoder uses, any value whose reconstruction violates the bound (float
rounding at bound edges, |pred| >> eb absorption) is demoted to an outlier.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RADIUS = 1 << 15


def quantize(
    x: np.ndarray,
    pred: np.ndarray,
    eb: float,
    radius: int = DEFAULT_RADIUS,
    round_output: bool | tuple[float, float] = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize values ``x`` against predictions ``pred``.

    Returns ``(codes, outlier_vals)`` where ``codes`` is uint32 with the same
    flattened length as ``x`` and ``outlier_vals`` holds the exact float32
    values for positions with code 0, in scan order.

    ``round_output=True`` verifies the bound against the rounded-to-integer
    reconstruction — required for integer sources, where the final cast can
    otherwise push the error past eb (e.g. eb=1.5, float error 1.5 rounds to
    integer error 2). Pass ``(span, offset)`` instead of ``True`` when ``x``
    is itself normalized (e.g. to [0, 1]) but the final output is rounded to
    an integer in the *raw* ``x * span + offset`` units -- the reconstruction
    is verified as ``(round(recon * span + offset) - offset) / span``, i.e.
    against the actual value the caller will round downstream.
    """
    if eb <= 0:
        raise ValueError(f"error bound must be > 0, got {eb}")
    x = np.asarray(x, dtype=np.float32).ravel()
    pred = np.asarray(pred, dtype=np.float32).ravel()
    if x.shape != pred.shape:
        raise ValueError(f"shape mismatch: x {x.shape} vs pred {pred.shape}")

    w = np.float64(2.0 * eb)
    q = np.rint((x.astype(np.float64) - pred.astype(np.float64)) / w).astype(np.int64)

    in_range = np.abs(q) < radius
    codes = np.where(in_range, q + radius, 0).astype(np.uint32)

    # Verify with the decoder's own arithmetic; demote violators to outliers.
    recon = _recon_from_codes(pred, codes, eb, radius)
    if round_output is True:
        recon = np.rint(recon)
    elif round_output:
        span, offset = round_output
        recon = (
            (np.rint(recon.astype(np.float64) * span + offset) - offset) / span
        ).astype(np.float32)
    ok = in_range & (np.abs(x - recon) <= np.float32(eb))
    codes[~ok] = 0
    outlier_vals = x[codes == 0].copy()
    return codes, outlier_vals


def dequantize(
    pred: np.ndarray,
    codes: np.ndarray,
    outlier_vals: np.ndarray,
    eb: float,
    radius: int = DEFAULT_RADIUS,
) -> np.ndarray:
    """Reconstruct float32 values from codes (+ exact outliers)."""
    pred = np.asarray(pred, dtype=np.float32).ravel()
    codes = np.asarray(codes, dtype=np.uint32).ravel()
    recon = _recon_from_codes(pred, codes, eb, radius)
    is_outlier = codes == 0
    n_out = int(is_outlier.sum())
    if n_out != len(outlier_vals):
        raise ValueError(
            f"outlier count mismatch: {n_out} zeros vs {len(outlier_vals)} values"
        )
    recon[is_outlier] = np.asarray(outlier_vals, dtype=np.float32)
    return recon


def _recon_from_codes(
    pred: np.ndarray, codes: np.ndarray, eb: float, radius: int
) -> np.ndarray:
    """Shared reconstruction arithmetic — the single source of truth for both
    the encoder's verification pass and the decoder."""
    q = codes.astype(np.int64) - radius
    return (pred.astype(np.float64) + np.float64(2.0 * eb) * q).astype(np.float32)
