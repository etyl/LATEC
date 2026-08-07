"""SZ-style interpolation baseline: bound holds and streams self-decode."""

import numpy as np
import pytest

from latec.bitstream import END_MODE_SHIFT
from latec.codec import compress, decompress
from latec.levels import stage_plan
from latec.predictor import (
    END_EXTRAP,
    END_MODE_MAX,
    END_QUAD,
    InterpPredictor,
    default_interp_end_mode,
)
from tests.test_codec_mock import smooth_image

END_MODES = list(range(END_MODE_MAX + 1))


def _interp_axis_full(V, axis, s, order, end_mode=0):
    """Pre-refactor whole-grid 1-D interpolation (reference for the compact
    per-point gather now in InterpPredictor). ``end_mode`` 0 is the original
    linear-then-edge-copy line-end handling; the bits add SZ3's quadratic ends
    and its one-sided extrapolation."""
    n = V.shape[axis]

    def gather(off):
        j = np.arange(n) + off
        return np.take(V, np.clip(j, 0, n - 1), axis=axis), (j >= 0) & (j < n)

    def bc(m):
        sh = [1] * V.ndim
        sh[axis] = m.shape[0]
        return m.reshape(sh)

    Lm1, vm1 = gather(-s)
    Lp1, vp1 = gather(+s)
    lin = 0.5 * (Lm1 + Lp1)
    pred = lin
    Lm3, vm3 = gather(-3 * s)
    Lp3, vp3 = gather(+3 * s)
    if order == "cubic":
        cub = (-Lm3 + 9 * Lm1 + 9 * Lp1 - Lp3) / 16.0
        one_far = lin
        if end_mode & END_QUAD:
            one_far = np.where(
                bc(vp3 & ~vm3),
                (3 * Lm1 + 6 * Lp1 - Lp3) / 8.0,
                np.where(bc(vm3 & ~vp3), (-Lm3 + 6 * Lm1 + 3 * Lp1) / 8.0, lin),
            )
        pred = np.where(bc(vm3 & vp3), cub, one_far)
    left, right = Lm1, Lp1
    if end_mode & END_EXTRAP:
        left = np.where(bc(vm3), 1.5 * Lm1 - 0.5 * Lm3, Lm1)
        right = np.where(bc(vp3), 1.5 * Lp1 - 0.5 * Lp3, Lp1)
    return np.where(bc(vm1 & vp1), pred, np.where(bc(vm1 & ~vp1), left, right))


@pytest.mark.parametrize("order", ["linear", "cubic"])
@pytest.mark.parametrize("center", [0, 1, 2])
@pytest.mark.parametrize("end_mode", END_MODES)
def test_compact_gather_matches_full_grid(order, center, end_mode):
    """The compact per-point predict is bit-identical to evaluating the old
    whole-grid interpolation and slicing the stage's points out — for every
    sub-stage of a non-multiple-of-stride, multi-channel region, and for every
    line-end mode (the ends are exactly where the two forms could diverge)."""
    shape, levels, stride, block = (70, 90), 3, 8, 1
    recon = np.random.RandomState(0).rand(2, *shape).astype(np.float32) * 50
    pred = InterpPredictor(
        order, levels, stride, block, center=center, end_mode=end_mode
    )
    known = np.zeros(shape, bool)
    for mask, s, axes in stage_plan(shape, levels, stride, block):
        if axes and mask.any():
            got = pred.predict(recon, known, mask)
            W = recon.astype(np.float64)
            if center == 0 or len(axes) == 1:
                ref = sum(
                    _interp_axis_full(W, a + 1, s, order, end_mode) for a in axes
                ) / len(axes)
            else:
                a = axes[0] if center == 1 else axes[-1]
                ref = _interp_axis_full(W, a + 1, s, order, end_mode)
            assert np.array_equal(got, ref.astype(np.float32)[:, mask])
        known |= mask


@pytest.mark.parametrize("order", ["linear", "cubic"])
@pytest.mark.parametrize("center", [0, 1, 2])
def test_fused_stage_matches_per_point_reference_3d(order, center):
    """3-D fused weight>=2 sub-stages (e.g. all 3 face-centre axis sets sharing
    one sub-stage) mix points with different odd axes; predict() must re-derive
    each point's own odd set rather than use one axes tuple for the whole
    stage. Reference combines the whole-grid per-axis interpolations using
    that same per-point rule instead of trusting ``axes`` uniformly."""
    shape, levels, stride, block = (16, 16, 16), 2, 4, 1
    recon = np.random.RandomState(0).rand(2, *shape).astype(np.float32) * 50
    pred = InterpPredictor(order, levels, stride, block, center=center, end_mode=0)
    known = np.zeros(shape, bool)
    saw_fused = False
    for mask, s, axes in stage_plan(shape, levels, stride, block):
        if axes and mask.any():
            got = pred.predict(recon, known, mask)
            W = recon.astype(np.float64)

            def odd_full(a):  # this axis's midpoints, whole grid, (1,)*4 broadcast
                sel = (np.arange(shape[a]) // s) % 2 == 1
                sh = [1, 1, 1, 1]
                sh[a + 1] = shape[a]
                return sel.reshape(sh)

            if center == 0 or len(axes) == 1:
                total = count = None
                for a in axes:
                    v = _interp_axis_full(W, a + 1, s, order, 0)
                    m = odd_full(a).astype(np.float64)
                    total = v * m if total is None else total + v * m
                    count = m if count is None else count + m
                with np.errstate(divide="ignore", invalid="ignore"):
                    ref = np.broadcast_to(total / count, W.shape)  # count==0 off-stage; sliced away below
            else:
                ref = None
                for a in reversed(axes) if center == 1 else axes:
                    v = _interp_axis_full(W, a + 1, s, order, 0)
                    ref = v if ref is None else np.where(odd_full(a), v, ref)
                ref = np.broadcast_to(ref, W.shape)
            if len(axes) > 1:
                saw_fused = True
            assert np.array_equal(got, ref.astype(np.float32)[:, mask])
        known |= mask
    assert saw_fused  # confirms the test actually exercised a fused sub-stage


def test_end_mode_roundtrips_through_flags_and_holds_bound():
    """Every line-end mode decodes from the stream flags alone (no factory) and
    keeps the bound; the encoder's recon matches the decoder's output."""
    img = smooth_image(70, 90, 1)[..., 0]
    eb, kw = 3.0, dict(levels=4, anchor_stride=8, anchor_block=1)
    for end_mode in END_MODES:
        pred = InterpPredictor("cubic", end_mode=end_mode, **kw)
        stream, stats = compress(img, eb, pred, eb_ratio=0.9, **kw)
        from latec.bitstream import read_stream

        header, _ = read_stream(stream)
        assert (header.flags >> END_MODE_SHIFT) & END_MODE_MAX == end_mode
        rec = decompress(stream)
        assert np.abs(img.astype(np.int64) - rec.astype(np.int64)).max() <= eb
        assert np.array_equal(stats["recon"], rec)
        assert stats["interp_end_mode"] == end_mode


def test_end_mode_sweep_picks_no_worse_than_the_default():
    """``tune != "fast"`` tries all four end modes, so its pick can never be
    larger than the fast-mode default's stream."""
    img = smooth_image(70, 90, 1)[..., 0]
    eb, kw = 3.0, dict(levels=4, anchor_stride=8, anchor_block=1)
    fast, fast_stats = compress(
        img, eb, InterpPredictor("cubic", **kw), tune="fast", **kw
    )
    tuned, tuned_stats = compress(
        img, eb, InterpPredictor("cubic", **kw), tune="size", **kw
    )
    assert fast_stats["interp_end_mode"] == default_interp_end_mode(2) == END_QUAD
    assert len(tuned) <= len(fast)
    assert tuned_stats["interp_end_mode"] in END_MODES


@pytest.mark.parametrize("order", ["linear", "cubic"])
@pytest.mark.parametrize("shape,eb", [((70, 90, 3), 2.0), ((80, 80, 1), 4.0)])
def test_roundtrip_bound(order, shape, eb):
    h, w, c = shape
    img = smooth_image(h, w, c)
    if c == 1:
        img = img[..., 0]
    # predictor schedule must match compress()'s levels/anchor_stride/block
    predictor = InterpPredictor(order, levels=4, anchor_stride=8, anchor_block=4)
    stream, stats = compress(
        img, eb, predictor, levels=4, anchor_stride=8, anchor_block=4
    )
    rec = decompress(stream)  # no factory: torch-free, decodes from flags alone
    assert rec.shape == img.shape and rec.dtype == img.dtype
    assert np.abs(img.astype(np.int64) - rec.astype(np.int64)).max() <= eb
    assert np.array_equal(stats["recon"], rec)  # encoder recon == decoder output


@pytest.mark.parametrize("eb_ratio", [1.0, 0.8, 0.5])
@pytest.mark.parametrize("center", [0, 1, 2])
def test_per_level_eb_and_center(eb_ratio, center):
    """Fixed per-level eb decay + any centre mode: the finest level keeps the
    full eb, so the global bound holds, and the tuned params round-trip via the
    header (encoder recon == decoder output)."""
    img = smooth_image(80, 96, 1)[..., 0]
    eb = 3.0
    pred = InterpPredictor(
        "cubic", levels=4, anchor_stride=8, anchor_block=4, center=center
    )
    stream, stats = compress(
        img, eb, pred, levels=4, anchor_stride=8, anchor_block=4, eb_ratio=eb_ratio
    )
    rec = decompress(stream)
    assert np.abs(img.astype(np.int64) - rec.astype(np.int64)).max() <= eb
    assert np.array_equal(stats["recon"], rec)


def test_autotune_improves_distortion_with_bounded_size_slack():
    """Auto-tuning may spend a small bounded size slack to reduce propagated
    coarse-level quantization error."""
    img = smooth_image(96, 96, 1)[..., 0]
    eb = 3.0
    kw = dict(levels=4, anchor_stride=8, anchor_block=4)
    tuned, tuned_stats = compress(
        img, eb, InterpPredictor("cubic", **kw), tune="rd", **kw
    )
    flat, flat_stats = compress(
        img, eb, InterpPredictor("cubic", **kw), eb_ratio=1.0, **kw
    )
    assert len(tuned) <= 1.05 * len(flat)
    assert tuned_stats["recon_sse"] <= flat_stats["recon_sse"]


def test_fast_tune_is_single_candidate():
    """Fast mode does a single encode at the predictor's fixed defaults: no
    eb_ratio/center sweep. The 2-D interp default is center=0, eb_ratio=0.9."""
    img = smooth_image(96, 96, 1)[..., 0]
    kw = dict(levels=4, anchor_stride=8, anchor_block=4)
    _, stats = compress(img, 3.0, InterpPredictor("cubic", **kw), tune="fast", **kw)
    assert stats["tune_candidates"] == 1
    assert stats["eb_ratio"] == InterpPredictor.fast_eb_ratio  # 0.9
    assert stats["interp_center"] == 0  # 2-D -> averaging centre
