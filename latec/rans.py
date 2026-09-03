"""Context-adaptive ANS coding for quantization bins.

Symbols are grouped by their predicted scale level and coded with compiled
``constriction`` categorical models. The groups are pushed in reverse order so
the ANS stack decodes them in ascending scale-level order.
"""

from __future__ import annotations

from collections import OrderedDict
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

    def cdfs_for(self, kexp: int, shape: int = 0, owx: int = 0) -> np.ndarray:
        """CDFs over the reduced alphabet of window ``kexp``, shaped by ``shape``."""
        return _build_cdfs_cached(
            int(kexp), int(shape), self.cdfs.shape[0], self.precision, int(owx)
        )


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


# ---- per-stage shape selection ----
#
# One Laplace scale cannot be both peakier at zero and heavier in the tail than
# a Laplace, but that is what the residuals in a single scale level measure like
# (s3d 1e-5, level 20: empirical p(0) 0.586 vs Laplace 0.377) -- the signature of
# pooling heterogeneous true scales into one grid level. Peak and tail are
# rigidly coupled, so the coder is forced to be wrong at both ends.
#
# The fix needs no new table machinery: mix two rows of the existing per-level
# weights, a peaked core plus a broader tail,
#
#     p(level) = (1 - w) * p[level] + w * p[min(level + D, n_levels - 1)]
#
# and let the encoder pick from a fixed dictionary, sending the entry id. A
# dictionary rather than a fitted table because fitted shapes do not generalise
# across fields -- the same pattern as the swept interp end_mode and the
# per-chunk gate descriptor. Entry 0 is pure Laplace, so no stage can come out
# worse than it would have, and the id rides free in the framing word.

_MIX_DELTAS = (4, 8, 16, 24)
_MIX_WEIGHTS = (0.02, 0.05, 0.12, 0.25, 0.45)

# (level offset, tail weight); entry 0 is pure Laplace.
SHAPES: tuple[tuple[int, float], ...] = ((0, 0.0),) + tuple(
    (d, w) for d in _MIX_DELTAS for w in _MIX_WEIGHTS
)
N_SHAPES = len(SHAPES)
SHAPE_BITS = 5
assert N_SHAPES <= 1 << SHAPE_BITS

# Fraction of a stage's points scored when picking the shape. The candidates are
# coarsely different, so this is a choice between shapes rather than an estimate
# of a parameter: a 5% stride picked the same entry on every measured stage at a
# third of the cost.
_SELECT_SAMPLE = 0.05
_SELECT_MIN = 1024

# ---- outlier-marker weight ----
#
# Symbol 0 marks an unpredictable value (the float itself rides in the stage's
# outlier array). The Laplace dictionary has no room for it -- every entry
# prices it at ~1e-9 -- which is right when outliers are freak events and very
# wrong when they are systematic: an integer source at a coarse bound demotes
# whole stages (a sixth of the points on the 160x120 uint8 image at eb=2), and
# at 24-bit precision each one then costs ~24 bits, tripling the stage. The
# marker's weight is therefore its own signalled axis, snapped to this
# dictionary; entry 0 keeps the original tables exactly, so the per-point path
# is unchanged.
OUTLIER_WEIGHTS: tuple[float, ...] = (
    0.0,
    2.0**-13,
    2.0**-10,
    2.0**-8,
    2.0**-6,
    2.0**-4,
    2.0**-2,
    0.5,
)
OUTLIER_BITS = 3
assert len(OUTLIER_WEIGHTS) <= 1 << OUTLIER_BITS


def choose_outlier_weight(n_outliers: int, n_codes: int) -> int:
    """Dictionary entry closest (in log odds) to the observed marker rate."""
    if n_codes <= 0 or n_outliers <= 0:
        return 0
    rate = n_outliers / n_codes
    return int(
        np.argmin([abs(np.log(max(w, 1e-12)) - np.log(rate)) for w in OUTLIER_WEIGHTS])
    )


def _with_outlier_weight(weights: np.ndarray, owx: int) -> np.ndarray:
    """Re-weight the marker symbol of a per-level weight table."""
    ow = OUTLIER_WEIGHTS[int(owx)]
    if not ow:
        return weights
    out = weights.copy()
    out[:, 1:] *= (1.0 - ow) / out[:, 1:].sum(1, keepdims=True)
    out[:, 0] = ow
    return out

# A model set costs n_levels * 2**(kexp+1) * 4 bytes, so at the widest windows
# each dictionary entry is 16 MB and takes ~1 s to build. Offering all 21 there
# pushes the working set past any sane cache budget and the rebuilds dominate
# (measured 4.5x encode on rti at 1e-5). Above _WIDE_KEXP the encoder therefore
# only considers the light-tail entries, which is where the wide-window picks
# concentrate anyway: it holds the working set near 120 MB and costs nothing on
# s3d, 0.08pp on cloud and 0.27pp on rti. This is an encoder-side restriction --
# the id space and the decoder are unchanged.
_WIDE_KEXP = 13
_WIDE_WEIGHT = 0.02


@lru_cache(maxsize=None)
def _shape_ids(kexp: int) -> tuple[int, ...]:
    """Dictionary entries the encoder may select for window ``kexp``."""
    if kexp < _WIDE_KEXP:
        return tuple(range(N_SHAPES))
    return (0,) + tuple(
        i for i, (_d, w) in enumerate(SHAPES) if w == _WIDE_WEIGHT
    )


def choose_shape(
    codes: np.ndarray,
    levels64: np.ndarray,
    tables: RansTables,
    kexp: int,
    *,
    sample: float = _SELECT_SAMPLE,
) -> int:
    """Dictionary entry minimizing this stage's coded length.

    Scores against the continuous per-level weights, so all candidates share one
    cached array and quantized tables are only ever built for the winner.
    """
    codes = np.asarray(codes, np.uint32).ravel()
    if len(codes) == 0:
        return 0
    n_levels = tables.cdfs.shape[0]
    syms = _narrow(codes, tables.radius, kexp)
    levels = np.asarray(levels64, np.uint8).ravel().astype(np.int64)
    if syms.shape != levels.shape:
        raise ValueError("codes and levels must have the same flattened length")
    if sample < 1.0:
        step = max(1, len(syms) // max(_SELECT_MIN, int(len(syms) * sample)))
        if step > 1:
            syms, levels = syms[::step], levels[::step]

    weights = _level_weights(int(kexp), n_levels)
    floor = 1.0 / tables.total  # _quantize_pmf never leaves a symbol below this
    idx = np.arange(n_levels)
    best_id, best_cost = 0, np.inf
    for shape_id in _shape_ids(int(kexp)):
        d, w = SHAPES[shape_id]
        p = weights[levels, syms]
        if w:
            broad = weights[np.minimum(idx + d, n_levels - 1)[levels], syms]
            p = (1.0 - w) * p + w * broad
        cost = -np.log2(np.maximum(p, floor)).sum()
        if cost < best_cost:
            best_id, best_cost = shape_id, float(cost)
    return best_id


@lru_cache(maxsize=32)
def _level_weights(kexp: int, n_levels: int) -> np.ndarray:
    """Row-normalized continuous per-level bin probabilities for one window.

    The shared basis for every dictionary entry, and the only full-alphabet array
    the selection pass needs.
    """
    span = (1 << kexp) - 1
    alphabet = 2 * span + 2
    eb = 1.0
    lo = eb / SCALE_LO_DIV
    hi = eb * SCALE_HI_MULT
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_levels)).astype(np.float64)
    q = np.arange(-span, span + 1, dtype=np.float64)
    left = q * (2.0 * eb) - eb
    right = q * (2.0 * eb) + eb
    w = np.empty((n_levels, alphabet), np.float64)
    for i, b in enumerate(grid):
        w[i, 1:] = _laplace_cdf_np(right, b) - _laplace_cdf_np(left, b)
        w[i, 0] = w[i, 1:].max() * 1e-9  # outlier marker
    w /= w.sum(1, keepdims=True)
    return w


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
    cdfs = _build_cdfs_cached(full_kexp(radius), 0, n_levels, precision)
    lo = eb / SCALE_LO_DIV
    hi = eb * SCALE_HI_MULT
    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_levels)).astype(np.float32)
    return RansTables(
        cdfs=cdfs, precision=precision, eb=eb, radius=radius, scale_grid=grid
    )


@lru_cache(maxsize=16)
def _build_cdfs_cached(
    kexp: int,
    shape: int,
    n_levels: int,
    precision: int,
    owx: int = 0,
) -> np.ndarray:
    """CDFs in normalized units; invariant to the absolute error bound.

    Symbol 0 is the outlier marker; symbol ``i > 0`` is bin index
    ``q = i - 1 - (2**kexp - 1)``. ``shape`` indexes :data:`SHAPES`; entry 0 is
    the pure Laplace.
    """
    span = (1 << kexp) - 1
    alphabet = 2 * span + 2
    total = 1 << precision
    if alphabet > total:
        raise ValueError("precision is too small for the quantizer alphabet")

    weights = _with_outlier_weight(_shaped_weights(kexp, shape, n_levels), owx)

    cdfs = np.empty((n_levels, alphabet + 1), np.uint32)
    cdfs[:, 0] = 0
    cdfs[:, 1:] = np.cumsum(_quantize_pmf_rows(weights, total), axis=1, dtype=np.uint64)
    return cdfs


def model_bits(
    codes: np.ndarray,
    levels64: np.ndarray,
    tables: RansTables,
    *,
    kexp: int | None = None,
    shape: int = 0,
) -> float:
    codes = np.asarray(codes, np.uint32).ravel()
    levels = np.asarray(levels64, np.uint8).ravel()
    if codes.shape != levels.shape:
        raise ValueError("codes and levels must have the same flattened length")
    if len(codes) == 0:
        return 0.0
    if kexp is None:
        kexp = full_kexp(tables.radius)
    cdfs = tables.cdfs_for(kexp, shape)
    all_syms = _narrow(codes, tables.radius, kexp)
    probs = np.empty(len(codes), np.float64)
    for level in range(cdfs.shape[0]):
        idx = np.flatnonzero(levels == level)
        if len(idx):
            syms = all_syms[idx]
            cdf = cdfs[level]
            probs[idx] = cdf[syms + 1].astype(np.int64) - cdf[syms].astype(np.int64)
    return float((-np.log2(probs / tables.total)).sum())


# ---- stage-constant scale level ----
#
# A predictor with no scale head (interpolation) still wants the compiled rANS
# back end rather than a Huffman fallback. It has no per-point scale, so the
# encoder fits ONE level of the shared grid to the stage's own code histogram
# and signals it in the stage frame, exactly as ``kexp`` and the shape id are
# signalled. The model machinery, tables and dictionary are the per-point path's,
# with the per-level partition collapsed to a single bulk call.


# Symbols per pass of the stage-constant coder. ``_narrow`` widens to int64, so
# an untiled pass over the finest stage of a large field allocates several times
# the stage; the coder is a stack, so tiles just have to be pushed in reverse.
_CONST_TILE = 1 << 22


@lru_cache(maxsize=32)
def _shaped_weights(kexp: int, shape: int, n_levels: int) -> np.ndarray:
    """Per-level weights of one dictionary entry (entry 0 is pure Laplace)."""
    weights = _level_weights(kexp, n_levels)
    d, w = SHAPES[shape]
    if not w:
        return weights
    broad = weights[np.minimum(np.arange(n_levels) + d, n_levels - 1)]
    return (1.0 - w) * weights + w * broad


@lru_cache(maxsize=16)
def _neg_log2_weights(kexp: int, shape: int, n_levels: int, precision: int, owx: int):
    """``-log2`` of the per-level model weights, for histogram scoring."""
    weights = _with_outlier_weight(_shaped_weights(kexp, shape, n_levels), owx)
    floor = 1.0 / (1 << precision)  # _quantize_pmf never leaves a symbol below this
    return -np.log2(np.maximum(weights, floor)).astype(np.float32)


def choose_const_level(
    codes: np.ndarray, tables: RansTables, kexp: int
) -> tuple[int, int, int, float]:
    """(level, shape, outlier-weight entry, coded bits) for the shortest coding
    of ``codes`` under one scale level.

    Scored on the stage's exact symbol histogram: the candidate cost is then a
    matrix-vector product over the reduced alphabet, so all 64 levels of every
    dictionary entry are evaluated for the price of one pass over the codes.
    The marker weight follows the observed outlier rate, which is optimal
    independently of the level and shape (it only moves mass between symbol 0
    and the rest). The returned bit count is the model's own, which the coder
    tracks to well under a percent -- accurate enough for ``pack_stage`` to
    compare back ends without encoding.
    """
    codes = np.asarray(codes, np.uint32).ravel()
    n_levels = tables.cdfs.shape[0]
    if len(codes) == 0:
        return 0, 0, 0, 0.0
    alphabet = 2 * ((1 << int(kexp)) - 1) + 2
    hist = np.zeros(alphabet, np.float32)
    for a in range(0, len(codes), _CONST_TILE):
        hist += np.bincount(
            _narrow(codes[a : a + _CONST_TILE], tables.radius, int(kexp)),
            minlength=alphabet,
        )
    owx = choose_outlier_weight(int(hist[0]), len(codes))
    best = (np.inf, 0, 0)
    for shape_id in _shape_ids(int(kexp)):
        cost = _neg_log2_weights(
            int(kexp), int(shape_id), n_levels, tables.precision, owx
        ) @ hist
        level = int(cost.argmin())
        if cost[level] < best[0]:
            best = (float(cost[level]), level, int(shape_id))
    return best[1], best[2], owx, best[0]


def literal_dtype(kexp: int) -> np.dtype:
    """Symbol width for the literal back end: the narrowed alphabet is
    ``2**(kexp + 1)``, so windows up to 7 fit a byte."""
    return np.dtype(np.uint8 if int(kexp) <= 7 else np.uint16)


def narrow_literal(codes: np.ndarray, radius: int, kexp: int) -> np.ndarray:
    """The stage's narrowed symbols as a compact literal array.

    The rANS model is memoryless, which is the right call when a stage's codes
    are close to independent and costs a third of the stream when they are not:
    a smooth 5-D field's finest stages measure H(sym | previous) ~ 0.82 bit
    against H(sym) ~ 1.19, and no scale level can price that. Handing zstd the
    raw symbols instead lets its LZ pass take the correlation. ``pack_stage``
    picks per stage.
    """
    out = np.empty(len(codes), literal_dtype(kexp))
    for a in range(0, len(codes), _CONST_TILE):  # _narrow widens to int64
        out[a : a + _CONST_TILE] = _narrow(
            codes[a : a + _CONST_TILE], radius, int(kexp)
        )
    return out


def widen_literal(buf, n_codes: int, radius: int, kexp: int) -> np.ndarray:
    """Inverse of :func:`narrow_literal`."""
    syms = np.frombuffer(buf, literal_dtype(kexp), count=int(n_codes))
    out = np.empty(int(n_codes), np.uint32)
    for a in range(0, int(n_codes), _CONST_TILE):
        out[a : a + _CONST_TILE] = _widen(syms[a : a + _CONST_TILE], radius, int(kexp))
    return out


def rans_encode_const(
    codes: np.ndarray,
    level: int,
    tables: RansTables,
    *,
    kexp: int | None = None,
    shape: int = 0,
    owx: int = 0,
) -> bytes:
    """Encode a whole stage against a single scale level (see above)."""
    import constriction

    codes = np.asarray(codes, np.uint32).ravel()
    if len(codes) == 0:
        return b""
    if int(codes.max(initial=0)) >= tables.alphabet:
        raise ValueError("code exceeds table alphabet")
    n_levels = tables.cdfs.shape[0]
    if not 0 <= int(level) < n_levels:
        raise ValueError("scale level exceeds table count")
    if kexp is None:
        kexp = full_kexp(tables.radius)
    models = _ans_models(int(kexp), int(shape), n_levels, tables.precision, int(owx))
    model = models[int(level)]
    enc = constriction.stream.stack.AnsCoder()
    for a in reversed(range(0, len(codes), _CONST_TILE)):  # stack: last in, first out
        enc.encode_reverse(
            _narrow(codes[a : a + _CONST_TILE], tables.radius, int(kexp)).astype(
                np.int32
            ),
            model,
        )
    return enc.get_compressed().astype("<u4", copy=False).tobytes()


def rans_decode_const(
    blob: bytes,
    n_codes: int,
    level: int,
    tables: RansTables,
    *,
    kexp: int | None = None,
    shape: int = 0,
    owx: int = 0,
) -> np.ndarray:
    """Inverse of :func:`rans_encode_const`."""
    import constriction

    n_codes = int(n_codes)
    if n_codes == 0:
        if blob:
            raise ValueError("non-empty rANS blob for an empty stage")
        return np.empty(0, np.uint32)
    if len(blob) % 4:
        raise ValueError("rANS payload is not aligned to 32-bit words")
    n_levels = tables.cdfs.shape[0]
    if not 0 <= int(level) < n_levels:
        raise ValueError("scale level exceeds table count")
    if kexp is None:
        kexp = full_kexp(tables.radius)
    models = _ans_models(int(kexp), int(shape), n_levels, tables.precision, int(owx))
    dec = constriction.stream.stack.AnsCoder(np.frombuffer(blob, dtype="<u4"))
    model = models[int(level)]
    out = np.empty(n_codes, np.uint32)
    for a in range(0, n_codes, _CONST_TILE):
        b = min(a + _CONST_TILE, n_codes)
        out[a:b] = _widen(dec.decode(model, b - a), tables.radius, int(kexp))
    if not dec.is_empty():
        raise ValueError("trailing data in rANS payload")
    return out


@lru_cache(maxsize=4)
def const_tables(radius: int) -> RansTables:
    """Tables for the stage-constant path. The CDFs are eb-invariant and the
    scale grid is unused there (the level is signalled, not derived), so one
    table set per radius serves every stage."""
    return build_laplace_tables(1.0, int(radius))


def rans_encode(
    codes: np.ndarray,
    levels64: np.ndarray,
    tables: RansTables,
    *,
    kexp: int | None = None,
    shape: int = 0,
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

    models = _ans_models(kexp, shape, n_levels, tables.precision)
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
    shape: int = 0,
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
    models = _ans_models(kexp, shape, n_levels, tables.precision)
    syms = np.empty(len(levels), np.int64)
    for level in range(n_levels):
        idx = np.flatnonzero(levels == level)
        if len(idx):
            syms[idx] = dec.decode(models[level], len(idx))
    if not dec.is_empty():
        raise ValueError("trailing data in rANS payload")
    out[:] = _widen(syms, tables.radius, kexp)
    return out


# A model set costs n_levels * 2**(kexp+1) * 4 bytes, so counting entries is the
# wrong bound: one field can want a dozen shapes at kexp 15 (160 MB) and dozens
# more at kexp <= 8 (under 2 MB combined). Budget by bytes and evict least
# recently used, which keeps the small windows resident for free and only ever
# rebuilds the wide ones.
_MODEL_BUDGET_BYTES = 192 << 20
_model_cache: "OrderedDict[tuple[int, int, int, int, int], tuple]" = OrderedDict()
_model_bytes = 0


def _ans_models(kexp: int, shape: int, n_levels: int, precision: int, owx: int = 0):
    """Build reusable compiled models, cached under a byte budget.

    The discretized PMFs are invariant to ``eb`` because both bin boundaries and
    the scale grid grow linearly with it.
    """
    global _model_bytes
    import constriction

    key = (int(kexp), int(shape), int(n_levels), int(precision), int(owx))
    hit = _model_cache.get(key)
    if hit is not None:
        _model_cache.move_to_end(key)
        return hit

    cdfs = _build_cdfs_cached(*key)
    models = tuple(
        constriction.stream.model.Categorical(
            np.diff(cdf.astype(np.int64)).astype(np.float32), perfect=False
        )
        for cdf in cdfs
    )
    size = int(cdfs.shape[0]) * int(cdfs.shape[1]) * 4
    _model_cache[key] = models
    _model_bytes += size
    while len(_model_cache) > 1 and _model_bytes > _MODEL_BUDGET_BYTES:
        old, _ = _model_cache.popitem(last=False)
        _model_bytes -= int(old[2]) * ((1 << (old[0] + 1)) + 1) * 4
    return models


def _clear_model_cache() -> None:
    """Drop cached models; for tests that measure construction cost."""
    global _model_bytes
    _model_cache.clear()
    _model_bytes = 0


def _laplace_cdf_np(x, b):
    z = np.clip(x / b, -700.0, 700.0)
    return np.where(x < 0, 0.5 * np.exp(z), 1.0 - 0.5 * np.exp(-z))


def _quantize_pmf(weights: np.ndarray, total: int) -> np.ndarray:
    return _quantize_pmf_rows(np.asarray(weights, np.float64)[None], total)[0]


def _quantize_pmf_rows(weights: np.ndarray, total: int) -> np.ndarray:
    """Quantize each row of ``weights`` to counts summing to ``total``.

    Batched over levels because building a full-width table one level at a time
    costs ~250 ms, which a per-stage shape dictionary would pay many times over.
    """
    w = np.asarray(weights, np.float64)
    if w.ndim != 2:
        raise ValueError("weights must be 2-D")
    n_rows, alphabet = w.shape
    scaled = w / w.sum(1, keepdims=True) * total
    floor = np.floor(scaled)
    pmf = np.maximum(floor.astype(np.int64), 1)
    frac = scaled - floor
    diff = total - pmf.sum(1)

    # Surplus rows: spread whole passes, then one extra count to the largest
    # remainders. Deficit rows: reclaim from the smallest remainders, never
    # below the floor of 1 that keeps every symbol decodable.
    up = np.flatnonzero(diff > 0)
    if len(up):
        d = diff[up]
        bulk, rem = np.divmod(d, alphabet)
        pmf[up] += bulk[:, None]
        order = np.argsort(-frac[up], axis=1, kind="stable")
        rank = np.empty_like(order)
        np.put_along_axis(
            rank, order, np.broadcast_to(np.arange(alphabet), order.shape), axis=1
        )
        pmf[up] += rank < rem[:, None]

    down = np.flatnonzero(diff < 0)
    if len(down):
        need = -diff[down]
        order = np.argsort(frac[down], axis=1, kind="stable")
        avail = np.take_along_axis(pmf[down], order, axis=1) - 1
        cum = np.cumsum(avail, axis=1)
        take = np.clip(need[:, None] - (cum - avail), 0, avail)
        if np.any(take.sum(1) < need):
            raise ValueError("could not normalize PMF at requested precision")
        rows = pmf[down]
        np.put_along_axis(
            rows, order, np.take_along_axis(rows, order, axis=1) - take, axis=1
        )
        pmf[down] = rows

    return pmf.astype(np.uint32)
