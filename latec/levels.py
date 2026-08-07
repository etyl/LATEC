"""Progressive stage schedule for the LATEC closed loop.

Stage 0 (anchors): ``anchor_block^ndim`` pixel blocks whose corners lie on the
``anchor_stride`` grid — coded by direct quantization. Each dyadic level
(stride = anchor_stride / 2^k) is then densified one axis at a time, but
*sequentially by how many axes a point is a midpoint on*: a point whose
coordinate is an odd multiple of the level stride on the axes in some set ``A``
(and on the coarse grid elsewhere) is filled only after every point with fewer
odd axes. So its ±stride neighbours along each axis in ``A`` have already been
decoded, and it is interpolated along each of those axes from them — the more
axes a point straddles, the more reconstructed neighbours it sees. In 2-D this
recovers the classic order: horizontal/vertical edge-midpoints (one odd axis,
two coarse neighbours) then the cell centre (two odd axes, four neighbours).

Grouping every distinct axis *set* into its own sub-stage costs ``2^ndim - 1``
of them per level — exponential in rank. Only the weight-1 sets (one sub-stage
per axis, e.g. the 3 edge-midpoint directions in 3-D) carry distinct enough
geometry to keep separate; every axis set of the same weight >= 2 (all 3 face
centres in 3-D, all 4 face-diagonals in 4-D, ...) is revealed together as one
fused sub-stage, since they differ only in *which* axes are odd and prediction
re-derives that per point from its own coordinates (``levels.py`` builds no
per-point axes metadata; ``predictor.py``/``gnn_codec.py`` recompute
``(coord // stride) % 2 == 1`` where needed). That yields ``2*ndim - 1``
sub-stages per level — linear in rank, identical to the exponential form at
ndim <= 2 — and the disjoint masks still union to the full grid. Encoder,
decoder, and the GNN trainer all derive it from the header parameters alone.
"""

from __future__ import annotations

import itertools

import numpy as np


def _combo_slices(axes: tuple[int, ...], s: int, ndim: int) -> tuple[slice, ...]:
    """Index tuple selecting the points that are midpoints (odd multiples of
    ``s``) on exactly ``axes`` and on the coarse (``2s``) grid elsewhere.

    Both per-axis conditions are plain arithmetic progressions -- odd multiples
    of ``s`` are ``s::2s``, multiples of ``2s`` are ``0::2s`` -- so the set is a
    cross product of strided slices and can be written straight into a zeroed
    mask. The equivalent broadcast form (``ones`` then one ``&=`` per axis)
    touched the whole dense grid ``ndim + 1`` times per sub-stage, and the fused
    weight >= 2 loop below built one such grid per axis set only to ``|=`` them
    together; at rank 4 that was the most expensive host operation in a warm
    encode."""
    return tuple(
        slice(s, None, 2 * s) if j in axes else slice(0, None, 2 * s)
        for j in range(ndim)
    )


def stage_plan(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> list[tuple[np.ndarray, int, tuple[int, ...]]]:
    """Ordered sub-stages as (mask, stride, axes) for a grid of arbitrary rank.

    ``axes`` is the *candidate* set of axes on which this sub-stage's points
    may be midpoints: a single axis for the weight-1 sub-stages (kept
    separate, so ``axes`` names it exactly), or ``range(ndim)`` for a fused
    weight >= 2 sub-stage, since its points' actual odd-axis sets vary (all
    same-weight combinations are revealed together). Callers that need a
    point's real odd axes re-derive them from its own coordinates —
    ``(coord // stride) % 2 == 1`` — which is exact for every point in this
    schedule. The anchor stage has ``axes == ()``. ``stage_masks`` drops the
    metadata."""
    shape = tuple(int(n) for n in shape)
    ndim = len(shape)
    if ndim < 1:
        raise ValueError("shape must have at least one axis")
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if anchor_stride < 2 or anchor_stride & (anchor_stride - 1):
        raise ValueError("anchor_stride must be a power of two >= 2")
    if not 1 <= anchor_block <= anchor_stride:
        raise ValueError("anchor_block must be in [1, anchor_stride]")
    # The dyadic levels must reach stride 1, i.e. levels >= log2(anchor_stride);
    # otherwise the finest pixels never become midpoints and get dumped into the
    # remainder clause below as one huge, badly-predicted stage (60-90% bloat).
    if (1 << levels) < anchor_stride:
        raise ValueError(
            f"levels={levels} too small for anchor_stride="
            f"{anchor_stride}: need levels >= log2(anchor_stride) = "
            f"{anchor_stride.bit_length() - 1} to densify to stride 1"
        )

    covered = np.zeros(shape, bool)

    anchor = np.zeros(shape, bool)
    for offs in itertools.product(range(anchor_block), repeat=ndim):
        anchor[tuple(slice(o, None, anchor_stride) for o in offs)] = True
    plan: list[tuple[np.ndarray, int, tuple[int, ...]]] = [(anchor, anchor_stride, ())]
    covered |= anchor

    for k in range(1, levels + 1):
        s = max(anchor_stride >> k, 1)
        # weight w = number of axes a point is a midpoint on; low -> high so a
        # point's ±s neighbours along its odd axes are already decoded.
        for axis in range(ndim):  # weight 1: kept separate, one sub-stage each
            mask = np.zeros(shape, bool)
            mask[_combo_slices((axis,), s, ndim)] = True
            mask &= ~covered
            if k == levels and ndim == 1:  # only sub-stage this level: absorbs remainder
                mask |= ~covered
            plan.append((mask, s, (axis,)))
            covered |= mask

        for w in range(2, ndim + 1):  # weight >= 2: fuse same-weight axis sets
            mask = np.zeros(shape, bool)
            for axes in itertools.combinations(range(ndim), w):
                mask[_combo_slices(axes, s, ndim)] = True
            mask &= ~covered
            if k == levels and w == ndim:  # final sub-stage absorbs remainder
                mask |= ~covered
            plan.append((mask, s, tuple(range(ndim))))
            covered |= mask

    return plan


def stage_masks(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> list[np.ndarray]:
    return [
        mask for mask, _, _ in stage_plan(shape, levels, anchor_stride, anchor_block)
    ]


def stage_strides(ndim: int, levels: int, anchor_stride: int) -> list[int]:
    """Per-stage lattice stride, aligned with ``stage_plan`` order, computed in
    closed form from ``(ndim, levels, anchor_stride)`` alone — no masks.

    The stride sequence is a pure function of the schedule shape: stage 0 (the
    anchors) has stride ``anchor_stride``; each dyadic level ``k`` contributes
    ``2*ndim - 1`` sub-stages (``ndim`` weight-1 + ``ndim - 1`` fused weight >= 2,
    see ``stage_plan``) all at stride ``max(anchor_stride >> k, 1)``. It is
    independent of the grid extent, so this reproduces ``[stride for _, stride,
    _ in stage_plan(shape, ...)]`` for any ``shape`` of rank ``ndim`` without
    materialising a single stage mask (the mask build is O(levels * n_points),
    catastrophic on the large representative grids ``stage_ebs`` is handed in
    high dimensions)."""
    if ndim < 1:
        raise ValueError("shape must have at least one axis")
    if levels < 1:
        raise ValueError("levels must be >= 1")
    if anchor_stride < 2 or anchor_stride & (anchor_stride - 1):
        raise ValueError("anchor_stride must be a power of two >= 2")
    if (1 << levels) < anchor_stride:
        raise ValueError(
            f"levels={levels} too small for anchor_stride="
            f"{anchor_stride}: need levels >= log2(anchor_stride) = "
            f"{anchor_stride.bit_length() - 1} to densify to stride 1"
        )
    per_level = 2 * ndim - 1
    strides = [anchor_stride]
    for k in range(1, levels + 1):
        strides += [max(anchor_stride >> k, 1)] * per_level
    return strides


def point_levels(
    coords: "list[np.ndarray] | tuple[np.ndarray, ...]",
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> np.ndarray:
    """Dyadic level at which each point is revealed, from coordinate residues.

    ``coords`` is one integer array per axis (equal shapes, broadcast not
    required). Level 0 = anchor pattern (every coordinate ``% anchor_stride <
    anchor_block``); otherwise the smallest ``k >= 1`` whose lattice
    ``stride >> k`` contains the point on every axis. Points off every dyadic
    lattice (the schedule's remainder clause) land on the finest level. This is
    exactly the level of the ``stage_plan`` stage that reveals the point, and
    the chunked codec uses it to pick a halo point's per-level coarse
    embedding without materialising any full-shape stage mask."""
    coords = [np.asarray(c, np.int64) for c in coords]
    out = np.full(coords[0].shape, levels, np.int8)
    anchor = np.ones(coords[0].shape, bool)
    for c in coords:
        anchor &= (c % anchor_stride) < anchor_block
    for k in range(levels - 1, 0, -1):  # coarse levels overwrite finer ones
        s = max(anchor_stride >> k, 1)
        on = np.ones(coords[0].shape, bool)
        for c in coords:
            on &= (c % s) == 0
        out[on] = k
    out[anchor] = 0
    return out


def stage_ebs(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    anchor_block: int,
    eb: float,
    eb_ratio: float,
) -> list[float]:
    """Per-stage absolute error bound, aligned with ``stage_plan`` order.

    Coarser (larger-stride) levels get a tighter bound ``eb * eb_ratio**depth``,
    ``depth`` = log2(stride / finest stride), so their quantization error
    propagates less into the finer levels interpolated from them (QoZ-style
    level-wise error budgeting). The finest level keeps the full ``eb``, so the
    global ``|x - recon| <= eb`` bound still holds unconditionally. ``eb_ratio``
    1.0 -> flat ``eb`` everywhere (classic SZ).

    Depends on ``shape`` only through its rank: the per-stage strides are
    closed-form (see ``stage_strides``), so no stage masks are built. This keeps
    the call cheap even on the large same-rank representative grids the chunked
    codec evaluates it on (a 4-D ``(2*stride)^4`` grid is ~17M points — building
    its masks cost seconds per compress)."""
    strides = stage_strides(len(shape), levels, anchor_stride)
    if not 1 <= anchor_block <= anchor_stride:
        raise ValueError("anchor_block must be in [1, anchor_stride]")
    finest = min(strides)
    return [eb * eb_ratio ** np.log2(stride / finest) for stride in strides]
