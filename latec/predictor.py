"""Predictors: the pluggable stage that replaces SZ's Lorenzo/spline prediction.

Both predictors share the interface
    predict(recon, known, pos=None) -> pred
with recon float32 (C, T, T) in original data units, known bool (T, T), and
pos an optional bool (T, T) mask of the points actually being predicted this
stage. With pos given, pred is the compact float32 (C, pos.sum()) prediction
at just those points; with pos=None, pred covers the whole tile. Predictions
must be a pure function of (recon * known, known) so the decoder can
reproduce them exactly.
"""

from __future__ import annotations

import numpy as np

from .bitstream import FLAG_CUBIC, FLAG_INTERP
from .levels import LazyStageMasks, mask_tiles, stage_counts, stage_strides


def default_interp_center(ndim: int) -> int:
    """Best fast-mode ``center`` for a spatial rank. Averaging the per-axis
    interpolation (``center=0``) helps in 2-D/3-D but degrades as rank grows —
    a 4-D cell-centre blends 4 already-quantized axis-estimates and loses to
    SZ3's single-direction scheme. Empirically ``center=1`` (interpolate along
    one axis, like SZ3) wins from 4-D up; ``center=0`` wins at/below 3-D."""
    return 1 if ndim >= 4 else 0


# How a 1-D interpolation line treats its ends, as an independent pair of bits
# (see ``_interp_axis_at``). Measured stream sizes, cubic, eb_ratio 0.9:
#
#   field / rel eb          legacy      QUAD   QUAD|EXTRAP
#   s3d 160^3     1e-2        7242      7005          4686   -35%
#   s3d 160^3     1e-3       33624     28945         21096   -37%
#   s3d 160^3     1e-4      133278    109992         96433   -28%
#   rti 32^4      1e-3      703949    700270        702095
#   CLOUDf48 3-D  1e-2      604217    607575        611385   +1.2%
#   CLOUDf48 3-D  1e-3     1836769   1858680       1869022   +1.8%
#   kodim23 2-D    eb 2     353710    353426        353699
#
# No rank rule separates these -- s3d and CLOUDf48 are both 3-D and want
# opposite ends of the table -- so this is a swept, stream-recorded choice.
# Both bits clear reproduces the pre-fix behaviour (linear fallback, then an
# edge copy) for the fields that prefer it.
END_QUAD = 1 << 0  # quadratic form where one far (+-3s) neighbour is missing
END_EXTRAP = 1 << 1  # extrapolate past a line end instead of copying
END_MODE_MAX = END_QUAD | END_EXTRAP


def default_interp_end_mode(ndim: int) -> int:
    """Fast-mode ``end_mode``: the quadratic ends, without extrapolation.

    Deliberately not rank-dependent (unlike ``center``) -- see the table above,
    where two 3-D fields disagree. ``END_QUAD`` is the best single default on the
    evidence: it wins on 3 of the 4 fields measured (up to -14% on s3d) and its
    worst case is +1.2% on CLOUDf48. ``END_EXTRAP`` is left off by default
    because it is bimodal -- another -24% on s3d, but the worst regressions in
    the table elsewhere -- so it has to be earned by the ``tune != "fast"``
    sweep, which tries all four combinations and keeps the smallest stream.

    The knob is field-dependent, not rank-dependent, which is the general lesson:
    SZ3 profiles each field on ~0.5% sampled blocks and picks its interpolation
    configuration per input, and that -- not any single stencil -- is where its
    remaining advantage over a fixed-default encoder comes from."""
    return END_QUAD


# Query points are predicted in tiles: every intermediate below is (C, tile)
# rather than (C, |stage|), so a stage's working set stays in cache and the peak
# host memory of a prediction no longer scales with the field.
_TILE = 1 << 20


def _flat_gatherer(flatW, flat, coords, shape, strides, axis):
    """``gather(off) -> (values (C, M) float64, valid (M,))`` for the neighbours
    ``off`` cells away along ``axis``, edge-clamped.

    Indexes the flattened reconstruction by arithmetic on the query points' own
    flat indices (one multiply-add per neighbour) instead of building an n-D
    index tuple, and upcasts only the gathered M values -- promoting the whole
    field to float64 first cost a full-size temporary per stage.
    """
    ca = coords[axis]
    n = shape[axis]
    stride = int(strides[axis])

    def gather(off):
        t = ca + off
        valid = (t >= 0) & (t < n)
        np.clip(t, 0, n - 1, out=t)
        return flatW[:, flat + (t - ca) * stride].astype(np.float64), valid

    return gather


def _interp_axis_at(gather, s, order, end_mode):
    """SZ3's 1D interpolation of one query tile along one axis: cubic weights
    [-1, 9, 9, -1]/16 over the four nearest same-line samples. ``gather(off)``
    supplies the neighbours ``off`` cells away and their validity (see
    ``_flat_gatherer``), so this is independent of how the reconstruction is
    laid out.

    Line ends follow SZ3's ``Interpolators.hpp`` rather than degrading to linear
    and then to an edge copy, which is what this used to do and what cost the
    interpolation baseline a third of its stream on s3d:

    * ``END_QUAD``: one far neighbour missing (``±3s`` outside, both ``±s``
      inside) -> the quadratic through the three available samples,
      ``interp_quad_1`` / ``interp_quad_2``, instead of dropping two orders to
      linear;
    * ``END_EXTRAP``: one *near* neighbour missing (past the last lattice point
      on this line) -> SZ3's ``interp_linear1`` extrapolation ``1.5b - 0.5a``
      from the two samples behind, instead of copying ``b``. A nearest copy has
      residual of order the local variation rather than of order eb, so those
      few points are expensive *and* their reconstruction error propagates into
      every finer level interpolated from them.

    ``end_mode`` 0 restores the original linear/edge-copy behaviour. See
    ``default_interp_end_mode`` for why the choice is rank-gated and swept."""
    Lm1, vm1 = gather(-s)
    Lp1, vp1 = gather(+s)
    lin = 0.5 * (Lm1 + Lp1)
    pred = lin
    far = None
    if order == "cubic":
        Lm3, vm3 = gather(-3 * s)
        Lp3, vp3 = gather(+3 * s)
        far = (Lm3, vm3, Lp3, vp3)
        cub = (-Lm3 + 9 * Lm1 + 9 * Lp1 - Lp3) / 16.0
        one_far = lin
        if end_mode & END_QUAD:
            one_far = np.where(
                (vp3 & ~vm3)[None],
                (3 * Lm1 + 6 * Lp1 - Lp3) / 8.0,  # interp_quad_1 (leading end)
                np.where(
                    (vm3 & ~vp3)[None],
                    (-Lm3 + 6 * Lm1 + 3 * Lp1) / 8.0,  # interp_quad_2 (trailing)
                    lin,  # both far neighbours missing: nothing better than linear
                ),
            )
        pred = np.where((vm3 & vp3)[None], cub, one_far)
    both = vm1 & vp1
    left, right = Lm1, Lp1
    if end_mode & END_EXTRAP and not both.all():
        if far is None:  # linear order never gathered the far samples
            Lm3, vm3 = gather(-3 * s)
            Lp3, vp3 = gather(+3 * s)
        else:
            Lm3, vm3, Lp3, vp3 = far
        # interp_linear1: continue the last known slope, falling back to the copy
        # when even the second sample behind is off the line.
        left = np.where(vm3[None], 1.5 * Lm1 - 0.5 * Lm3, Lm1)
        right = np.where(vp3[None], 1.5 * Lp1 - 0.5 * Lp3, Lp1)
    only_left = (vm1 & ~vp1)[None]
    return np.where(both[None], pred, np.where(only_left, left, right))  # (C, M)


class InterpPredictor:
    """SZ3-style interpolation baseline dropped into LATEC's closed loop, so
    GNN vs. classical interpolation is isolated to the predictor (identical
    quantizer + rANS/zstd stage, matching SZ3's own pipeline). Torch- and
    checkpoint-free, so streams decode without a model.

    Each dyadic level is split into codec sub-stages ordered by how many axes a
    point straddles as a midpoint (``levels.stage_plan``): the one-odd-axis
    edge-midpoints first (kept one sub-stage per axis), then every multi-axis
    weight fused into a single sub-stage (all 2-D cell centres, all 3-D face
    centres, ...). Because they are separate stages the codec quantizes each
    into ``recon`` before the next reads it — SZ3's interleaved quantize/predict
    order — so a point's ±stride neighbours along each of its odd axes are
    already reconstructed, and it is predicted by averaging the single-axis
    interpolation over exactly those axes (2 neighbours for an edge-midpoint, 4
    for a 2-D cell centre). A fused sub-stage mixes points with different odd
    axes, so ``predict`` re-derives each point's own set from its coordinates
    rather than trusting one tuple for the whole stage. The predictor supplies
    its own ``stage_masks``; the decoder rebuilds the identical schedule from
    the header dims alone.

    SZ3's interpolation has no fixed input size, so the codec runs it over the
    whole field as a single region without padding or prediction seams.
    """

    tunable = True  # encoder sweeps (eb_ratio, center) and keeps the smallest
    fast_eb_ratio = 0.9  # single-encode (tune=fast) default; see codec.encode

    def __init__(
        self,
        order: str = "cubic",
        levels: int = 4,
        anchor_stride: int = 16,
        anchor_block: int = 4,
        center: int | None = None,
        end_mode: int | None = None,
    ):
        if order not in ("linear", "cubic"):
            raise ValueError("order must be 'linear' or 'cubic'")
        if end_mode is not None and not 0 <= end_mode <= END_MODE_MAX:
            raise ValueError(f"end_mode must be in [0, {END_MODE_MAX}]")
        self.order = order
        self.levels = levels
        self.anchor_stride = anchor_stride
        self.anchor_block = anchor_block
        # multi-odd-axis ("centre") prediction: 0 avg all odd axes, 1/2
        # interpolate along the first/last odd axis only (SZ3's single-direction
        # centre). None = pick by spatial rank (``default_interp_center``): 0 for
        # <=3-D, 1 from 4-D up, where averaging over many axes starts to lose.
        # The codec resolves None to a concrete value once the region rank is
        # known; decode always overrides from the header.
        self.center = center
        # line-end handling (END_QUAD | END_EXTRAP). None = pick by spatial rank
        # (``default_interp_end_mode``), resolved by the codec and always
        # overridden from the stream flags on decode, exactly like ``center``.
        self.end_mode = end_mode
        self.stream_flag = FLAG_INTERP | (FLAG_CUBIC if order == "cubic" else 0)
        self.checkpoint_hash = b"\0" * 16
        self._cache: dict[tuple[int, ...], dict] = {}

    def _schedule(self, shape: tuple[int, ...]) -> dict:
        """``|known so far| -> (stride, axes)`` for a region of this shape, so
        ``predict`` knows which axes to interpolate along at each call.

        The stage order, strides and sizes are the shared schedule's
        (``levels.stage_strides`` / ``levels.stage_counts``, so the GNN
        codec/trainer stay on the identical plan), all closed form: keying the
        geometry off the stage sizes must not cost a pass over the field's
        stage masks. Cached per shape."""
        key = tuple(int(n) for n in shape)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ndim = len(key)
        counts = stage_counts(key, self.levels, self.anchor_stride, self.anchor_block)
        strides = stage_strides(ndim, self.levels, self.anchor_stride)
        # stage axes, in plan order: one weight-1 sub-stage per axis, then the
        # fused weight >= 2 sub-stages, whose points' own odd axes vary.
        fused = tuple(range(ndim))
        axes_by_stage = [()] + [
            (axis,) if axis < ndim else fused
            for _ in range(self.levels)
            for axis in range(2 * ndim - 1)
        ]
        schedule: dict[int, tuple[int, tuple[int, ...]]] = {}
        covered = 0
        for n, s, axes in zip(counts, strides, axes_by_stage):
            if axes and n:  # non-anchor sub-stage; predict() is keyed by prior |known|
                schedule[covered] = (s, axes)
            covered += n
        self._cache[key] = schedule
        return schedule

    def stage_masks(self, shape, levels, anchor_stride, anchor_block):
        return LazyStageMasks(shape, levels, anchor_stride, anchor_block)

    def predict(
        self, recon: np.ndarray, known: np.ndarray, pos: np.ndarray | None = None
    ) -> np.ndarray:
        entry = self._schedule(recon.shape[1:]).get(int(np.count_nonzero(known)))
        if entry is None:
            raise ValueError("known mask does not match the interp schedule")
        s, axes = entry
        shape = recon.shape[1:]
        # Interpolate each odd axis from its ±stride neighbours (already
        # reconstructed in an earlier, lower-weight sub-stage of the same level):
        # an edge-midpoint (one odd axis) reads 2 priors, a 2-D centre (two odd
        # axes) reads 4. ``center`` picks how a multi-odd-axis point combines
        # them: averaged (0) or single-direction (1/2).
        center = (
            self.center
            if self.center is not None
            else default_interp_center(len(shape))
        )
        end_mode = (
            self.end_mode
            if self.end_mode is not None
            else default_interp_end_mode(len(shape))
        )
        flatW = recon.reshape(recon.shape[0], -1)
        strides = np.cumprod((1,) + shape[:0:-1])[::-1].astype(np.int64)
        n = int(np.count_nonzero(pos)) if pos is not None else int(np.prod(shape))
        out = np.empty((recon.shape[0], n), np.float32)
        at = 0
        for flat in mask_tiles(pos, shape, _TILE):
            coords = np.unravel_index(flat, shape)

            def is_odd(a):  # this point's own midpoint axes, not the stage-wide
                # candidate set: a fused sub-stage (weight >= 2) mixes points whose
                # odd axes differ, e.g. all 3-D face centres share one sub-stage.
                # ``s`` is a power of two, so the odd-multiple test is one bit.
                return (coords[a] & s) != 0

            def axis_pred(a):
                return _interp_axis_at(
                    _flat_gatherer(flatW, flat, coords, shape, strides, a),
                    s,
                    self.order,
                    end_mode,
                )

            if center == 0 or len(axes) == 1:
                total = count = None
                for a in axes:
                    v = axis_pred(a)
                    m = is_odd(a)[None].astype(np.float64)
                    total = v * m if total is None else total + v * m
                    count = m if count is None else count + m
                tile = total / count
            else:
                tile = None
                order = reversed(axes) if center == 1 else axes  # first-/last-wins
                for a in order:
                    v = axis_pred(a)
                    take = is_odd(a)[None]
                    tile = v if tile is None else np.where(take, v, tile)
            out[:, at : at + len(flat)] = tile  # (C, tile)
            at += len(flat)
        return out if pos is not None else out.reshape(recon.shape)
