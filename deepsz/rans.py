"""Context-adaptive ANS coding for quantization bins.

Symbols are grouped by their predicted scale level and coded with compiled
``constriction`` categorical models. The groups are pushed in reverse order so
the ANS stack decodes them in ascending scale-level order.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class RansTables:
    cdfs: np.ndarray
    precision: int
    eb: float
    radius: int
    scale_grid: np.ndarray

    @property
    def total(self) -> int:
        return 1 << self.precision

    @property
    def alphabet(self) -> int:
        return 2 * self.radius

    def cdfs_for(self, kexp: int) -> np.ndarray:
        """CDFs over the reduced alphabet of window exponent ``kexp``."""
        return _build_laplace_cdfs_cached(int(kexp), self.cdfs.shape[0], self.precision)


# Shared eb-relative scale-grid span [eb/SCALE_LO_DIV, eb*SCALE_HI_MULT].
# HI was 64 until bitstream v6 / gnn-stream v4-v5: at eb<=1e-5 the prediction
# error stays orders of magnitude above eb, so ~half the points needed a scale
# above 64*eb and pinned at the grid ceiling (bench_levels sat+%). 64 levels
# over 16 octaves is still 0.25 octave/level -> negligible mismatch cost.
SCALE_LO_DIV = 16.0
SCALE_HI_MULT = 4096.0


# ---- per-stage alphabet sizing ----
#
# ``_quantize_pmf`` gives every symbol at least one count of ``1 << precision``,
# so a full 2*radius alphabet reserves 2*radius/2**precision of the mass for
# bins that cannot occur (0.39% at radius 2**15, precision 24). That caps p(0)
# at 0.99609 and floors EVERY symbol at 0.00565 bits however confident the model
# is -- which is the entire bill in a flat stage. Coding a stage over just the
# window it actually uses removes the floor.
#
# The window is a power of two, ``|q| <= K = 2**kexp - 1``, so the reduced
# alphabet ``2*K + 2 = 2**(kexp + 1)`` stays a power of two and only ~16 tables
# can ever be built. ``kexp = full_kexp(radius)`` reproduces the full-width
# alphabet exactly (symbol 0 is the outlier marker in both), so the narrowing is
# the identity there and the shipped tables are unchanged at that setting.


def full_kexp(radius: int) -> int:
    """Window exponent covering every in-range bin index for ``radius``."""
    return max(1, int(int(radius) - 1).bit_length())


def choose_kexp(codes: np.ndarray, radius: int) -> int:
    """Smallest window exponent that contains every in-range code in ``codes``.

    Outlier codes (0) always fit: they map to reduced symbol 0.
    """
    codes = np.asarray(codes, np.uint32).ravel()
    inr = codes[codes != 0]
    if len(inr) == 0:
        return 1
    peak = int(np.abs(inr.astype(np.int64) - int(radius)).max())
    return max(1, int(peak).bit_length())


def _narrow(codes: np.ndarray, radius: int, kexp: int) -> np.ndarray:
    """Full-alphabet codes -> reduced alphabet ``[0, 2**(kexp + 1))``."""
    span = (1 << int(kexp)) - 1
    q = codes.astype(np.int64) - int(radius)
    syms = np.where(codes == 0, 0, q + span + 1)
    if len(syms) and (int(syms.min()) < 0 or int(syms.max()) > 2 * span + 1):
        raise ValueError("code outside the narrowed rANS alphabet")
    return syms


def _widen(syms: np.ndarray, radius: int, kexp: int) -> np.ndarray:
    """Inverse of :func:`_narrow`."""
    span = (1 << int(kexp)) - 1
    codes = syms.astype(np.int64) - span - 1 + int(radius)
    return np.where(syms == 0, 0, codes).astype(np.uint32)


def scale_to_level(scale: np.ndarray, eb: float, n_levels: int = 64) -> np.ndarray:
    """Quantize Laplacian scale values to the shared 64-level log grid."""
    scale = np.asarray(scale, np.float32)
    lo = float(eb) / SCALE_LO_DIV
    hi = float(eb) * SCALE_HI_MULT
    if eb <= 0:
        raise ValueError("eb must be > 0")
    t = (np.log(np.clip(scale, lo, hi)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return np.rint(t * (n_levels - 1)).clip(0, n_levels - 1).astype(np.uint8)


def build_laplace_tables(
    eb: float,
    radius: int,
    *,
    n_levels: int = 64,
    precision: int = 24,
) -> RansTables:
    """Precompute integer CDFs for the scale grid at one stage eb/radius."""
    eb = float(eb)
    radius = int(radius)
    n_levels = int(n_levels)
    precision = int(precision)
    if eb <= 0:
        raise ValueError("eb must be > 0")
    cdfs = _build_laplace_cdfs_cached(full_kexp(radius), n_levels, precision)
    lo = eb / SCALE_LO_DIV
    hi = eb * SCALE_HI_MULT
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_levels)).astype(np.float32)
    return RansTables(
        cdfs=cdfs, precision=precision, eb=eb, radius=radius, scale_grid=grid
    )


@lru_cache(maxsize=128)
def _build_laplace_cdfs_cached(
    kexp: int,
    n_levels: int,
    precision: int,
) -> np.ndarray:
    """CDFs in normalized units; invariant to the absolute error bound.

    Symbol 0 is the outlier marker; symbol ``i > 0`` is bin index
    ``q = i - 1 - (2**kexp - 1)``.
    """
    span = (1 << kexp) - 1
    alphabet = 2 * span + 2
    total = 1 << precision
    if alphabet > total:
        raise ValueError("precision is too small for the quantizer alphabet")

    eb = 1.0
    lo = eb / SCALE_LO_DIV
    hi = eb * SCALE_HI_MULT
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_levels)).astype(np.float64)
    q = np.arange(-span, span + 1, dtype=np.float64)
    left = q * (2.0 * eb) - eb
    right = q * (2.0 * eb) + eb

    cdfs = np.empty((n_levels, alphabet + 1), np.uint32)
    for i, b in enumerate(grid):
        weights = np.empty(alphabet, np.float64)
        weights[1:] = _laplace_cdf_np(right, b) - _laplace_cdf_np(left, b)
        weights[0] = weights[1:].max() * 1e-9  # outlier marker
        pmf = _quantize_pmf(weights, total)
        cdf = np.empty(alphabet + 1, np.uint32)
        cdf[0] = 0
        cdf[1:] = np.cumsum(pmf, dtype=np.uint64)
        cdfs[i] = cdf
    return cdfs


def model_bits(
    codes: np.ndarray,
    levels64: np.ndarray,
    tables: RansTables,
    *,
    kexp: int | None = None,
) -> float:
    codes = np.asarray(codes, np.uint32).ravel()
    levels = np.asarray(levels64, np.uint8).ravel()
    if codes.shape != levels.shape:
        raise ValueError("codes and levels must have the same flattened length")
    if len(codes) == 0:
        return 0.0
    if kexp is None:
        kexp = full_kexp(tables.radius)
    cdfs = tables.cdfs_for(kexp)
    all_syms = _narrow(codes, tables.radius, kexp)
    probs = np.empty(len(codes), np.float64)
    for level in range(cdfs.shape[0]):
        idx = np.flatnonzero(levels == level)
        if len(idx):
            syms = all_syms[idx]
            cdf = cdfs[level]
            probs[idx] = cdf[syms + 1].astype(np.int64) - cdf[syms].astype(np.int64)
    return float((-np.log2(probs / tables.total)).sum())


def rans_encode(
    codes: np.ndarray,
    levels64: np.ndarray,
    tables: RansTables,
    *,
    kexp: int | None = None,
) -> bytes:
    import constriction

    codes = np.asarray(codes, np.uint32).ravel()
    levels = np.asarray(levels64, np.uint8).ravel()
    if codes.shape != levels.shape:
        raise ValueError("codes and levels must have the same flattened length")
    if len(codes) == 0:
        return b""
    if int(codes.max(initial=0)) >= tables.alphabet:
        raise ValueError("code exceeds table alphabet")
    n_levels = tables.cdfs.shape[0]
    if int(levels.max(initial=0)) >= n_levels:
        raise ValueError("scale level exceeds table count")
    if kexp is None:
        kexp = full_kexp(tables.radius)

    models = _ans_models(kexp, n_levels, tables.precision)
    enc = constriction.stream.stack.AnsCoder()
    syms_i32 = _narrow(codes, tables.radius, kexp).astype(np.int32)
    for level in range(n_levels - 1, -1, -1):
        idx = np.flatnonzero(levels == level)
        if len(idx):
            enc.encode_reverse(syms_i32[idx], models[level])
    words = enc.get_compressed()
    return words.astype("<u4", copy=False).tobytes()


def rans_decode(
    blob: bytes,
    levels64: np.ndarray,
    tables: RansTables,
    *,
    kexp: int | None = None,
) -> np.ndarray:
    import constriction

    levels = np.asarray(levels64, np.uint8).ravel()
    out = np.empty(len(levels), np.uint32)
    if len(levels) == 0:
        if blob:
            raise ValueError("non-empty rANS blob for an empty stage")
        return out
    if len(blob) % 4:
        raise ValueError("rANS payload is not aligned to 32-bit words")
    n_levels = tables.cdfs.shape[0]
    if int(levels.max(initial=0)) >= n_levels:
        raise ValueError("scale level exceeds table count")

    if kexp is None:
        kexp = full_kexp(tables.radius)
    words = np.frombuffer(blob, dtype="<u4").astype(np.uint32, copy=False)
    dec = constriction.stream.stack.AnsCoder(words)
    models = _ans_models(kexp, n_levels, tables.precision)
    syms = np.empty(len(levels), np.int64)
    for level in range(n_levels):
        idx = np.flatnonzero(levels == level)
        if len(idx):
            syms[idx] = dec.decode(models[level], len(idx))
    if not dec.is_empty():
        raise ValueError("trailing data in rANS payload")
    out[:] = _widen(syms, tables.radius, kexp)
    return out


@lru_cache(maxsize=64)
def _ans_models(kexp: int, n_levels: int, precision: int):
    """Build reusable compiled models.

    The discretized Laplace PMFs are invariant to ``eb`` because both bin
    boundaries and the scale grid grow linearly with it.
    """
    import constriction

    cdfs = _build_laplace_cdfs_cached(int(kexp), int(n_levels), int(precision))
    return tuple(
        constriction.stream.model.Categorical(
            np.diff(cdf.astype(np.int64)).astype(np.float32), perfect=False
        )
        for cdf in cdfs
    )


def _laplace_cdf_np(x, b):
    z = np.clip(x / b, -700.0, 700.0)
    return np.where(x < 0, 0.5 * np.exp(z), 1.0 - 0.5 * np.exp(-z))


def _quantize_pmf(weights: np.ndarray, total: int) -> np.ndarray:
    w = np.asarray(weights, np.float64)
    scaled = w / w.sum() * total
    pmf = np.floor(scaled).astype(np.int64)
    pmf = np.maximum(pmf, 1)
    diff = int(total - pmf.sum())
    frac = scaled - np.floor(scaled)
    if diff > 0:
        order = np.argsort(-frac)
        bulk, rem = divmod(diff, len(pmf))
        if bulk:
            pmf += bulk
        if rem:
            pmf[order[:rem]] += 1
    elif diff < 0:
        order = np.argsort(frac)
        avail = pmf[order] - 1
        cum = np.cumsum(avail)
        take = np.clip(-diff - (cum - avail), 0, avail)
        pmf[order] -= take
        if take.sum() < -diff:
            raise ValueError("could not normalize PMF at requested precision")
    return pmf.astype(np.uint32)
