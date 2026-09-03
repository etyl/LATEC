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


def _check_plan(
    shape: tuple[int, ...], levels: int, anchor_stride: int, anchor_block: int
) -> tuple[int, ...]:
    """Validate schedule parameters and return the normalized shape."""
    shape = tuple(int(n) for n in shape)
    if len(shape) < 1:
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
    return shape


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
    metadata. ``iter_stage_plan`` is the same schedule without holding every
    mask alive at once."""
    return list(iter_stage_plan(shape, levels, anchor_stride, anchor_block))


def iter_stage_plan(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
):
    """``stage_plan`` as a generator: one mask is built per step.

    A codec pass reads each stage once, in order, so the list form's real cost
    is that all ``1 + levels * (2*ndim - 1)`` full-size boolean grids stay alive
    together -- 36 of them for a 1 GiB float32 field is 9 GiB of masks, by far
    the largest allocation in a whole-field encode. Streaming them keeps two
    (the current mask and the running ``covered`` grid)."""
    shape = _check_plan(shape, levels, anchor_stride, anchor_block)
    ndim = len(shape)
    covered = np.zeros(shape, bool)

    anchor = np.zeros(shape, bool)
    for offs in itertools.product(range(anchor_block), repeat=ndim):
        anchor[tuple(slice(o, None, anchor_stride) for o in offs)] = True
    covered |= anchor
    yield (anchor, anchor_stride, ())
    del anchor

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
            covered |= mask
            yield (mask, s, (axis,))
            del mask

        for w in range(2, ndim + 1):  # weight >= 2: fuse same-weight axis sets
            mask = np.zeros(shape, bool)
            for axes in itertools.combinations(range(ndim), w):
                mask[_combo_slices(axes, s, ndim)] = True
            mask &= ~covered
            if k == levels and w == ndim:  # final sub-stage absorbs remainder
                mask |= ~covered
            covered |= mask
            yield (mask, s, tuple(range(ndim)))
            del mask


class LazyStageMasks:
    """The ``stage_plan`` masks as a sized, re-iterable stream.

    Behaves like the ``stage_masks`` list for a codec pass -- ``len()`` and
    iteration in schedule order -- but builds each mask on demand instead of
    holding the whole schedule's grids alive (see ``iter_stage_plan``). ``len``
    is closed form, so it costs nothing to ask."""

    def __init__(
        self,
        shape: tuple[int, ...],
        levels: int,
        anchor_stride: int,
        anchor_block: int = 1,
    ):
        self.shape = _check_plan(shape, levels, anchor_stride, anchor_block)
        self.levels = int(levels)
        self.anchor_stride = int(anchor_stride)
        self.anchor_block = int(anchor_block)

    def __len__(self) -> int:
        return n_stages(len(self.shape), self.levels)

    def __iter__(self):
        for mask, _, _ in iter_stage_plan(
            self.shape, self.levels, self.anchor_stride, self.anchor_block
        ):
            yield mask


# Points per tile when a stage is walked by flat index (see ``mask_tiles``).
_MASK_TILE = 1 << 20


def mask_tiles(pos, shape, tile: int = _MASK_TILE):
    """Yield a stage mask's points as flat-index tiles, in ``recon[:, pos]``
    order (C-order over the mask); ``pos=None`` means every point.

    Both the codec and the interpolation predictor walk a stage this way instead
    of indexing with the boolean mask directly. NumPy expands an n-D boolean
    index into one int64 array per axis, so ``field[:, pos]`` on a 5-D field
    transiently allocates 40 bytes per selected point -- several hundred MiB at
    the finest stage of a large field, and the codec's actual host peak. The
    mask is scanned a slab at a time, so neither the tile nor the scan holds an
    index array proportional to the stage size."""
    if pos is None:
        n = int(np.prod(shape))
        for a in range(0, n, tile):
            yield np.arange(a, min(a + tile, n), dtype=np.int64)
        return
    flat = pos.reshape(-1)
    slab = 4 * tile
    for a in range(0, flat.size, slab):
        idx = np.flatnonzero(flat[a : a + slab])
        if not len(idx):
            continue
        idx += a
        for b in range(0, len(idx), tile):
            yield idx[b : b + tile]


def n_stages(ndim: int, levels: int) -> int:
    """Number of sub-stages in the schedule (see ``stage_strides``)."""
    return 1 + levels * (2 * ndim - 1)


def _slice_count(n: int, start: int, step: int) -> int:
    """Number of indices ``start::step`` selects from an axis of extent ``n``."""
    return 0 if n <= start else (n - start + step - 1) // step


def stage_counts(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> list[int]:
    """Point count per stage, aligned with ``stage_plan`` order.

    Closed form for ``anchor_block == 1``: every stage's set is a union of
    cross products of arithmetic progressions, and the schedule's sets are
    disjoint by construction (a point revealed at stride ``s`` is an odd
    multiple of ``s`` on at least one axis, so no coarser lattice contains it),
    which makes the ``&= ~covered`` step a no-op for counting. The final
    sub-stage additionally absorbs every point off all dyadic lattices, i.e.
    whatever the other stages leave over. Blocked anchors overlap the dyadic
    lattices, so those fall back to counting the masks.

    Lets a caller size the schedule -- the interpolation predictor keys its
    per-stage geometry on the number of points revealed so far -- without
    materializing a single stage mask."""
    shape = _check_plan(shape, levels, anchor_stride, anchor_block)
    ndim = len(shape)
    if anchor_block != 1:
        return [
            int(np.count_nonzero(mask))
            for mask, _, _ in iter_stage_plan(shape, levels, anchor_stride, anchor_block)
        ]
    counts = [int(np.prod([_slice_count(n, 0, anchor_stride) for n in shape]))]
    prev_s = None
    for k in range(1, levels + 1):
        s = max(anchor_stride >> k, 1)
        if s == prev_s:  # repeated finest level: every point is already covered
            counts += [0] * (2 * ndim - 1)
            continue
        prev_s = s
        for axis in range(ndim):
            counts.append(
                int(
                    np.prod(
                        [
                            _slice_count(n, s if j == axis else 0, 2 * s)
                            for j, n in enumerate(shape)
                        ]
                    )
                )
            )
        for w in range(2, ndim + 1):
            counts.append(
                sum(
                    int(
                        np.prod(
                            [
                                _slice_count(n, s if j in axes else 0, 2 * s)
                                for j, n in enumerate(shape)
                            ]
                        )
                    )
                    for axes in itertools.combinations(range(ndim), w)
                )
            )
    counts[-1] += int(np.prod(shape)) - sum(counts)  # remainder clause
    return counts


def stage_masks(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    anchor_block: int = 1,
) -> list[np.ndarray]:
    return [
        mask
        for mask, _, _ in iter_stage_plan(shape, levels, anchor_stride, anchor_block)
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


# --- closed-form stage geometry (no masks, no per-point plan) ---------------
# Every sub-stage's point set is a union of *parity classes*: a class fixes, per
# axis, whether the coordinate is an odd multiple of the sub-stage stride (a
# midpoint on that axis) or an even one, so the class is a cross product of
# arithmetic progressions -- a strided view of the grid, addressable with pure
# integer arithmetic the way SZ3/HPEZ address theirs.
#
# The one thing a view cannot give directly is the class's position inside the
# *stage* array, which is ordered by flat grid index and therefore interleaves
# the classes. That order is lexicographic in (m_0, p_0, m_1, p_1, ...) -- the
# per-axis lattice index split into its high part and its parity bit -- and
# counting the stage points that precede a given position collapses to
#
#     rank(m, p) = offset(p) + sum_j m_j * G_j(r_j)
#
# with r_j the odd-axis budget still unplaced at axis j. It is *affine in m*, so
# each class is also a strided view of the stage array. Both halves of the
# schedule are then plain strides, and the codec never materializes a per-point
# index, a stage mask, or a cached plan.


class ClassLayout:
    """One parity class of a sub-stage: where its points live, in both spaces.

    ``starts``/``step``/``sizes`` address the grid (``start::step`` per axis, a
    view); ``offset``/``strides`` address the stage's compact point array (an
    ``as_strided`` view of it). ``parity[j]`` is 1 on the axes the class's points
    are midpoints of, which is exactly the set of axes it is interpolated along.
    """

    __slots__ = ("parity", "axes", "starts", "step", "sizes", "strides", "offset")

    def __init__(self, parity, starts, step, sizes, strides, offset):
        self.parity = parity
        self.axes = tuple(j for j, p in enumerate(parity) if p)
        self.starts, self.step, self.sizes = starts, step, sizes
        self.strides, self.offset = strides, offset

    @property
    def size(self) -> int:
        return int(np.prod(self.sizes)) if self.sizes else 1

    def slices(self, origins=None) -> tuple[slice, ...]:
        """Grid slice tuple for this class, optionally shifted by ``origins``."""
        if origins is None:
            return tuple(
                slice(b, b + self.step * n, self.step)
                for b, n in zip(self.starts, self.sizes)
            )
        return tuple(
            slice(o + b, o + b + self.step * n, self.step)
            for o, b, n in zip(origins, self.starts, self.sizes)
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ClassLayout(parity={self.parity}, starts={self.starts}, "
            f"step={self.step}, sizes={self.sizes}, strides={self.strides}, "
            f"offset={self.offset})"
        )


class StageLayout:
    """A sub-stage as its parity classes plus the schedule metadata."""

    __slots__ = ("stride", "weight", "axes", "classes", "size")

    def __init__(self, stride, weight, axes, classes, size):
        self.stride, self.weight, self.axes = stride, weight, axes
        self.classes, self.size = classes, size


def _single_class(parity, starts, step, sizes):
    """Layout of a sub-stage that has exactly one class: plain row-major."""
    strides = [1] * len(sizes)
    for j in range(len(sizes) - 2, -1, -1):
        strides[j] = strides[j + 1] * sizes[j + 1]
    return ClassLayout(parity, tuple(starts), step, tuple(sizes), tuple(strides), 0)


def _fused_classes(shape, s, weight, low_axes):
    """Layouts of every weight-``weight`` parity class of one fused sub-stage.

    ``low_axes`` (grow mode) drops each named axis's coordinate-0 hyperplane,
    which shifts that axis's even-parity progression from ``0::2s`` to ``2s::2s``
    and takes one position out of the interleave -- a constant shift of the rank,
    so the layout stays affine.
    """
    ndim = len(shape)
    lo, cnt = [], []
    for j, n in enumerate(shape):
        base = 2 * s if j in low_axes else 0
        lo.append((1 if j in low_axes else 0, 0))
        cnt.append((_slice_count(n, base, 2 * s), _slice_count(n, s, 2 * s)))
    # F[j][r]: stage points in axes j.. with r odd axes left to place.
    # G[j][r]: the same for one fixed lattice index on axis j (either parity).
    F = [[0] * (weight + 1) for _ in range(ndim + 1)]
    F[ndim][0] = 1
    G = [[0] * (weight + 1) for _ in range(ndim + 1)]
    for j in range(ndim - 1, -1, -1):
        for r in range(weight + 1):
            odd = F[j + 1][r - 1] if r else 0
            F[j][r] = cnt[j][0] * F[j + 1][r] + cnt[j][1] * odd
            G[j][r] = F[j + 1][r] + odd
    out = []
    for combo in itertools.combinations(range(ndim), weight):
        parity = tuple(1 if j in combo else 0 for j in range(ndim))
        starts, sizes, strides = [], [], []
        offset, r = 0, weight
        for j, p in enumerate(parity):
            strides.append(G[j][r])
            starts.append(s if p else (2 * s if j in low_axes else 0))
            sizes.append(cnt[j][p])
            offset += lo[j][p] * G[j][r]
            if j in low_axes:
                if not p:
                    offset -= F[j + 1][r]
            elif p:
                offset += F[j + 1][r]
            r -= p
        out.append(
            ClassLayout(parity, tuple(starts), 2 * s, tuple(sizes), tuple(strides), offset)
        )
    return out, F[0][weight]


def stage_layouts(
    shape: tuple[int, ...],
    levels: int,
    anchor_stride: int,
    low_axes: tuple[int, ...] = (),
) -> list[StageLayout]:
    """``stage_plan``'s schedule as strided class layouts, built by arithmetic.

    Aligned one-for-one with ``stage_plan`` / ``stage_counts``: stage ``i``'s
    classes partition exactly the points of stage ``i``'s mask, and concatenating
    each class's points in ``as_strided`` order reproduces the mask's
    ``flatnonzero`` order. ``anchor_block`` is fixed at 1 (the codec's only
    setting); the schedule's remainder clause is empty whenever the dyadic
    levels reach stride 1, which ``_check_plan`` already requires.

    ``low_axes`` removes the coordinate-0 hyperplane of each listed axis from
    every stage -- the grow-mode column split, whose points the up/left
    neighbour already coded.
    """
    shape = _check_plan(shape, levels, anchor_stride, 1)
    ndim = len(shape)
    low_axes = tuple(sorted(set(low_axes)))
    zero = tuple(0 for _ in range(ndim))
    anchor_starts = [anchor_stride if j in low_axes else 0 for j in range(ndim)]
    out = [
        StageLayout(
            anchor_stride,
            0,
            (),
            [
                _single_class(
                    zero,
                    anchor_starts,
                    anchor_stride,
                    [_slice_count(n, b, anchor_stride) for n, b in zip(shape, anchor_starts)],
                )
            ],
            0,
        )
    ]
    out[0].size = out[0].classes[0].size
    prev_s = None
    for k in range(1, levels + 1):
        s = max(anchor_stride >> k, 1)
        if s == prev_s:  # repeated finest level: every point is already covered
            out += [StageLayout(s, w, (), [], 0) for w in [1] * ndim + list(range(2, ndim + 1))]
            continue
        prev_s = s
        for axis in range(ndim):
            parity = tuple(1 if j == axis else 0 for j in range(ndim))
            starts = [
                s if j == axis else (2 * s if j in low_axes else 0) for j in range(ndim)
            ]
            sizes = [_slice_count(n, b, 2 * s) for n, b in zip(shape, starts)]
            cl = _single_class(parity, starts, 2 * s, sizes)
            out.append(StageLayout(s, 1, (axis,), [cl], cl.size))
        for w in range(2, ndim + 1):
            classes, size = _fused_classes(shape, s, w, low_axes)
            out.append(StageLayout(s, w, tuple(range(ndim)), classes, size))
    return out
