"""Lightweight, dimension-agnostic GNN predictor for the LATEC closed loop.

Same interface as the other predictors (see predictor.py):
    predict(recon, known) -> pred
with recon float32 (C, *S) in original units, known bool (*S). It is a pure
function of (recon * known, known), so encoder and decoder reproduce it
bitwise. `S` may be any spatial shape of any rank; the network's weights are
shared across directions, so a model trained on 2-D images runs unchanged on
n-D grids.

Design (single hop; long-range context comes from the codec's hierarchical
stage schedule, which progressively densifies `known`):

  * for every *line* through a point (the (3^n - 1)/2 axis and diagonal
    directions) find the nearest known sample on each side;
  * store one embedding per lattice axis at every point. Per axis k
    independently, concatenate a small learned direction-state vector to each
    neighbour's axis-k embedding, indexed by the signed direction cosine
    cos(line direction, axis k) — a handful of discrete values (DirState) — so
    the axis-k channel reads its neighbours through their angle to axis k;
  * turn a two-sided pair into a trend/curvature message (BiDirEmbed) or a
    one-sided neighbour into an extrapolation message (DirEmbed);
  * pool the per-line messages into one context per axis with single-query
    attention, then pool axes with a second single-query attention before
    reading out a Laplacian mean/scale (PredHead), conditioned on the
    normalized error bound;
  * once a point's own value is revealed — known, but carrying the small error
    left by residual coding — embed it with InitEmbed and fuse it into the
    per-axis contexts with MixEmbed to form that point's finalized embeddings
    (the ones stored in the propagating field). In training the revealed value
    is the truth plus noise, so MixEmbed learns to trust it only up to that
    error.

Axes never mix during propagation; direction enters only as the concatenated
direction-state embedding, so with that table zeroed every axis channel is
identical — the axial structure is a pure inductive bias layered on the
direction-blind model.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .levels import point_levels, stage_masks, stage_strides

# torch is imported lazily inside the class / model so that importing this
# module (e.g. for FLAG constants) stays cheap.

CKPT_VERSION = 7

# Query-tile for the message pass: caps the transient (B, L, m, K, d) buffers so
# GPU peak scales with _M_TILE, not the stage's full M (line_pool is per-query,
# so tiling is math-identical within a stream). Stored in meta so decode replays
# the exact float path.
#
# The cost is launch count -- blocks = sum_stages ceil(M_s / tile), and the
# message pass is launch-bound -- but peak memory saturates long before that
# starts to bite, leaving a wide free band. Measured on a 32^4 4-D chunk (36
# stages, largest M = 393216), single chunk, idle V100, best of 5:
#
#   tile      off   131072   65536   32768   16384    8192    4096    2048   1024
#   blocks     36       39      44      59      90     143     277     539   1043
#   time     1.00x    1.02x   1.00x   1.02x   1.05x   1.17x   1.47x   2.05x  3.20x
#   peak MiB 6600     4574    3515    3404    3402    3402    3402    3402   3402
#
# 65536 is the knee: -47% peak for no measurable time. It is also the size a
# sub-stage would have had before same-weight axis sets were fused into one
# (chunk points / 2^ndim), so it splits a fused sub-stage back to the residency
# the pre-fusion schedule had. Well clear of the small-tile cliff that would
# shred a single-chunk 2-D encode, where a big stage is only a few blocks.
_M_TILE = int(os.environ.get("LATEC_M_TILE", 1 << 16))
# Compact frames duplicate most per-point neighbour indices and message-block
# selections. They are cheap enough to rebuild after the agg-level-1 arithmetic
# geometry optimization, so retaining them is opt-in; even one rank-4 frame can
# occupy hundreds of MiB. The base geometry remains cached separately.
_FRAME_CACHE_MIB = float(os.environ.get("LATEC_FRAME_CACHE_MIB", 0))
# Above this size, collecting only out-of-chunk neighbour references while the
# host geometry is already resident beats sorting/uniquing every reference on
# the GPU. The 64x33^3 target chunk is above it; a 32^4 single chunk is below.
_HOST_HALO_MIN_POINTS = 1 << 21
_AXIAL_IMPLICIT_GEOMETRY = True


def _cuda_working_budget(torch, device, fraction: float = 0.8) -> int:
    """Bytes safely available for new tensors, including reusable torch cache.

    CUDA's driver-level free count excludes blocks reserved by PyTorch even when
    they are currently unallocated.  Those blocks can satisfy later allocations
    (notably decode after encode), so omitting them produces false low-memory
    estimates.
    """
    driver_free = int(torch.cuda.mem_get_info(device)[0])
    reserved = int(torch.cuda.memory_reserved(device))
    allocated = int(torch.cuda.memory_allocated(device))
    reusable_cache = max(0, reserved - allocated)
    return int(fraction * (driver_free + reusable_cache))


def half_directions(ndim: int, agg_level: int | None = None) -> list[tuple[int, ...]]:
    """One representative per line: all offset vectors in {-1,0,1}^ndim whose
    first non-zero component is +1 (so d and -d collapse to one line).

    ``agg_level`` caps the *neighbourhood aggregation level* — the L1 length of
    the direction, i.e. its number of non-zero components (how many axes a hop
    moves along at once). Level 1 keeps only the ``ndim`` axis-aligned face
    directions (direct neighbours); level 2 adds the 2-axis diagonals (2 hops in
    L1); ... level ``ndim`` (or ``None``) keeps all ``(3^ndim - 1)/2`` lines, the
    full neighbourhood. Since the network is direction-blind (direction enters
    only as the direction-state embedding and the pool masks unused lines),
    dropping the
    higher-L1 lines is a pure inference-time cost/accuracy trade-off that shrinks
    the per-stage message tensor's L dimension — the dominant factor in high-D."""
    dirs = []
    for d in itertools.product((-1, 0, 1), repeat=ndim):
        first = next((x for x in d if x != 0), 0)
        if first > 0 and (agg_level is None or sum(x != 0 for x in d) <= agg_level):
            dirs.append(d)
    return dirs


def _nearest_steps(pat: np.ndarray, dvec, P: int, res=None) -> np.ndarray:
    """Smallest step t>=1 along ``dvec`` landing on a True cell of the periodic
    pattern ``pat`` (period P), or 0 if none. The hit sequence along any lattice
    line is periodic in t with period dividing P, so the nearest hit is the first
    within [1,P]. With ``res`` (a tuple of ndim residue arrays, len M), evaluated
    only at those M query residues -> O(P*M); without it, at every residue of the
    P^ndim tile -> O(P^(ndim+1)). Query points are all we ever index, so pass res.

    Resolved points are dropped from the sweep as they are hit, so the cost is
    ``sum_t |still unresolved at t|`` rather than ``(largest t needed) * M``.
    Most points hit at t = 1 or 2 and a handful need the full period, and one
    laggard used to drag every other point through every remaining step. That
    matters most for a *fused* sub-stage (``levels.stage_plan``), which mixes
    points of several odd-axis combinations whose hit distances differ, so its
    worst case applies to 2^ndim-ish times as many points as an unfused one."""
    if res is None:
        res = np.indices(pat.shape)  # every residue: (ndim, P, ...)
    shape = res[0].shape
    t0 = np.zeros(shape, np.int64).ravel()  # 0 == no hit yet
    live = np.arange(t0.size)  # flat positions still unresolved
    r = [np.ravel(res[k]) for k in range(pat.ndim)]
    # The period is a stride, so it is a power of two: wrap with a mask rather
    # than numpy's (much slower) signed remainder. Two's complement makes
    # ``x & (P - 1)`` agree with ``x % P`` for negative x too.
    wrap = (lambda x: x & (P - 1)) if P & (P - 1) == 0 else (lambda x: x % P)
    for t in range(1, P + 1):
        hit = pat[tuple(wrap(r[k] + t * dvec[k]) for k in range(pat.ndim))]
        if not hit.any():
            continue
        t0[live[hit]] = t
        keep = ~hit
        live = live[keep]
        if live.size == 0:
            break
        r = [rk[keep] for rk in r]
    return t0.reshape(shape)


_NEAREST_TILE_CACHE: dict = {}
_NEAREST_MEMO_CACHE: dict = {}  # (id(pat), dvec, P) -> (pat ref, partial tile)
_MEMO_UNSOLVED = 255  # sentinel; valid answers are 0..P
_MEMO_MAX_CELLS = 1 << 20  # same ceiling as the full-tile path below


def _nearest_steps_memo(pat: np.ndarray, dvec, P: int, res) -> np.ndarray:
    """`_nearest_steps` memoized per residue, across chunk shapes.

    The answer depends only on a query's residue mod P, and ``pat`` is the
    schedule's period tile (``_period_prefixes``) -- neither depends on the
    chunk shape. Grow mode gives every chunk of a small grid its own shape, so
    the same (stage, direction) answers were recomputed once per shape: on a
    64^4 field at chunk 32 that is 16 shapes and 39% of encode wall. Solve each
    residue once for the whole encode instead and gather thereafter; a chunk's
    first visit still pays the full sweep, later ones are O(M).

    The table is a lazily filled ``P**ndim`` byte per (pat, dvec) -- filled only
    at the residues a stage actually queries, which sum to one tile across the
    whole schedule. Keyed on ``id(pat)`` (cheap; ``pat.tobytes()`` would hash a
    megabyte per call) with the array itself pinned in the entry so the id
    cannot be recycled onto a different pattern."""
    key = (id(pat), tuple(int(c) for c in dvec), P)
    hit = _NEAREST_MEMO_CACHE.get(key)
    if hit is None:
        table = np.full(pat.size, _MEMO_UNSOLVED, np.uint8)
        _NEAREST_MEMO_CACHE[key] = (pat, table)  # pat ref pins id(pat)
    else:
        table = hit[1]
    flat = np.ravel_multi_index(tuple(res), pat.shape)
    vals = table[flat]
    miss = vals == _MEMO_UNSOLVED
    if miss.any():
        solved = _nearest_steps(pat, dvec, P, tuple(r[miss] for r in res))
        table[flat[miss]] = solved
        vals = table[flat]
    return vals.astype(np.int64)


def _nearest_steps_at(
    pat: np.ndarray, dvec, P: int, res, *, query_only: bool = False
) -> np.ndarray:
    """`_nearest_steps` evaluated at the query residues ``res``, via a cached
    full period tile when that is cheaper: the tile costs O(P^(ndim+1)) once
    per (pat, dvec) and O(M) per lookup, vs O(P*M) per call for the direct
    path. ponytail: tile capped at 2^20 cells (4-D at stride 32); beyond that
    fall back to the direct path rather than build a giant tile."""
    M = len(res[0])
    # A chunk schedule partitions one period tile across many stages. Building
    # a full P**ndim lookup independently for every (stage, direction) looks
    # cheaper for a single large stage, but is catastrophically expensive over
    # the whole schedule (76 stages for levels=5 in 4-D). Across all stages the
    # query counts sum to only P**ndim, so evaluating at query residues is the
    # linear-work strategy for chunk geometry.
    if query_only:
        if pat.size <= _MEMO_MAX_CELLS and P < _MEMO_UNSOLVED:
            return _nearest_steps_memo(pat, dvec, P, res)
        return _nearest_steps(pat, dvec, P, res)
    if pat.size > 1 << 20 or pat.size > P * M:
        return _nearest_steps(pat, dvec, P, res)
    key = (pat.tobytes(), pat.shape, tuple(int(c) for c in dvec), P)
    tile = _NEAREST_TILE_CACHE.get(key)
    if tile is None:
        tile = _nearest_steps(pat, dvec, P)
        _NEAREST_TILE_CACHE[key] = tile
    return tile[tuple(res)]


def _line_static(dvec, torch, device=None):
    """Unit line direction and its Euclidean log-distance correction."""
    vec = torch.as_tensor(dvec, dtype=torch.float32, device=device)
    nnz = (vec != 0).sum().to(torch.float32)
    return vec / torch.sqrt(nnz), 0.5 * torch.log2(nnz)


def _slice_lines(self, m0, m1):
    """``(ip, in_, dp, dn, vp, vn)`` for queries ``[m0:m1]``, sliced from stored
    tensors.

    ``_CompactGeom`` holds its flat neighbour indices outright, because they are
    not of the form ``query_idx + step * (d . strides)`` -- they have been pushed
    through a remap. It is built per chunk instead of being cached per chunk
    *shape*, so there is no ``2 ** ndim`` multiplier to amortise."""
    return (
        self.ip[:, m0:m1],
        self.in_[:, m0:m1],
        self.dp[:, m0:m1],
        self.dn[:, m0:m1],
        self.vp[:, m0:m1],
        self.vn[:, m0:m1],
    )


class _StageGeom:
    """Neighbour geometry for one stage: the fixed set of ``M`` query points and,
    per half-direction, the +/- side neighbour's *step distance* and validity as
    torch tensors of length M. Query points only — no full-grid tensors — so
    memory scales with the stage, not the image.

    The flat neighbour index is **derived, not stored** (see ``lines``). Since a
    neighbour sits at ``Q + step * d``, its flat index is
    ``ravel(Q) + step * (d . strides)`` -- one small integer per line-end, not a
    64-bit index and a 32-bit distance. The stored form is ~3.5x smaller, which
    matters because a grow-mode encode caches one of these per *chunk shape* and
    there are ``2 ** ndim`` distinct shapes: at rank 4 the flat-index form put
    ~2 GB of device memory behind a cache that only ever holds 16 entries."""

    __slots__ = (
        "sp",
        "sn",
        "off",
        "vp",
        "vn",
        "cos",
        "lognnz",
        "query_idx",
        "idx_np",
        "M",
        "ndim",
        "ref_halo_np",
        "message_blocks",
    )

    def __init__(
        self,
        pat,
        query_coords,
        shape,
        max_radius,
        torch,
        device,
        agg_level=None,
        query_only=False,
        precompute_messages=True,
        ref_bounds=None,
        axial_stride=None,
        period=None,
    ):
        ndim = len(shape)
        self.ndim = ndim
        P = int(period) if period is not None else pat.shape[0]
        shp = np.asarray(shape)
        limit = min(max_radius, int(shp.max()))
        Q = query_coords  # (M, ndim)
        self.M = int(len(Q))
        self.idx_np = (
            np.ravel_multi_index([Q[:, k] for k in range(ndim)], shape)
            if self.M
            else np.zeros(0, np.int64)
        )

        def t(a):
            x = torch.from_numpy(np.ascontiguousarray(a))
            return x.to(device) if device is not None else x

        self.query_idx = t(self.idx_np.astype(np.int64))
        # P is normally a stride, hence a power of two: mask instead of numpy's
        # (much slower) remainder, as in `_nearest_steps`. Query coords are
        # non-negative, so the two agree exactly.
        mod = (lambda x: x & (P - 1)) if P & (P - 1) == 0 else (lambda x: x % P)
        res = (
            tuple(mod(Q[:, k]) for k in range(ndim))
            if self.M and axial_stride is None
            else None
        )
        # Row-major strides of `shape`, so a neighbour's flat index is one dot
        # product. Out-of-bounds rows produce garbage here, but `valid` masks
        # them to 0 below -- which is why this can skip the clip that
        # `ravel_multi_index` would otherwise require.
        nstrides = np.cumprod((1,) + tuple(shape)[:0:-1])[::-1].astype(np.int64)
        # Steps are bounded by `limit`, so they fit a 16-bit slot in every real
        # configuration; widen rather than silently wrap if that ever changes.
        sdt = np.int16 if limit < 2**15 else np.int32
        line_data = {k: [] for k in ("sp", "sn", "vp", "vn")}
        offs, cos, lognnz = [], [], []
        ref_halo = []
        if ref_bounds is not None:
            ref_lo = np.asarray(ref_bounds[0], np.int64)
            ref_hi = np.asarray(ref_bounds[1], np.int64)
        for d in half_directions(ndim, agg_level):
            ln = {}
            for side, sd in (("p", np.asarray(d)), ("n", -np.asarray(d))):
                if not self.M:
                    ln["s" + side] = np.zeros(0, sdt)
                    ln["v" + side] = np.zeros(0, bool)
                    continue
                if axial_stride is None:
                    step = _nearest_steps_at(
                        pat, sd, P, res, query_only=query_only
                    )  # (M,) at query residues
                elif axial_stride == 0:  # anchors: nothing is known yet
                    step = np.zeros(self.M, np.int64)
                else:
                    # At an axial dyadic stage, only odd-stride axes have
                    # previously revealed neighbours, exactly one stride away.
                    axis = int(np.flatnonzero(sd)[0])
                    odd = ((Q[:, axis] // axial_stride) & 1).astype(bool)
                    step = np.where(odd, axial_stride, 0)
                nb = Q + step[:, None] * sd  # neighbour coords
                valid = (step >= 1) & (step <= limit)  # legacy: <=limit
                for k in range(ndim):  # ... & in-bounds, axis by axis so the
                    if sd[k] > 0:  # (M, ndim) bool temporaries never exist
                        valid &= nb[:, k] < shp[k]
                    elif sd[k] < 0:
                        valid &= nb[:, k] >= 0
                if ref_bounds is not None:
                    outside = np.zeros(self.M, bool)
                    for k in range(ndim):
                        outside |= (nb[:, k] < ref_lo[k]) | (nb[:, k] >= ref_hi[k])
                    use = valid & outside
                    if use.any():
                        ref_halo.append((nb[use] @ nstrides).astype(np.int64))
                # Only the step survives; `lines` rebuilds the index and the
                # distance from it, including the legacy where-invalid defaults.
                ln["s" + side] = np.where(valid, step, 0).astype(sdt)
                ln["v" + side] = valid
            line_data["sp"].append(ln["sp"])
            line_data["sn"].append(ln["sn"])
            line_data["vp"].append(ln["vp"])
            line_data["vn"].append(ln["vn"])
            # Flat-index displacement of one step along +d; the -d side is -off.
            offs.append(int(np.asarray(d) @ nstrides))
            # Build all line constants on the host and transfer each stacked
            # attribute once below.  Moving every line-side separately issued
            # hundreds of tiny HtoD copies while constructing a rank-4 schedule.
            c, ld = _line_static(d, torch, None)
            cos.append(c)
            lognnz.append(ld)
        for name, values in line_data.items():
            setattr(self, name, t(np.stack(values, axis=0)))
        self.off = t(np.asarray(offs, np.int64))
        cos_t = torch.stack(cos, dim=0)
        self.cos = cos_t.to(device) if device is not None else cos_t
        lognnz_t = torch.stack(lognnz, dim=0).unsqueeze(1)
        self.lognnz = lognnz_t.to(device) if device is not None else lognnz_t
        self.ref_halo_np = (
            np.concatenate(ref_halo) if ref_halo else np.zeros(0, np.int64)
        )
        self.message_blocks = (
            _build_message_blocks(self, torch) if precompute_messages else None
        )

    def lines(self, m0, m1):
        """``(ip, in_, dp, dn, vp, vn)`` for queries ``[m0:m1]``, rebuilt from the
        stored steps.

        Reproduces the legacy stored form exactly, defaults included: flat index
        0 and distance 1.0 where a side has no neighbour (1.0 keeps ``log2``
        finite; the pool masks those lines out via ``vp``/``vn`` anyway). Called
        once per query tile per stage, on a tile-sized slice, so the wide int64
        index tensor exists only for the tile being consumed."""
        q = self.query_idx[m0:m1].unsqueeze(0)  # (1, m)
        vp, vn = self.vp[:, m0:m1], self.vn[:, m0:m1]
        sp, sn = self.sp[:, m0:m1].long(), self.sn[:, m0:m1].long()
        off = self.off.unsqueeze(1)  # (L, 1)
        # Invalid line-ends were stored as step 0, which makes both defaults fall
        # out arithmetically: masking by validity gives index 0, and clamping the
        # step up to 1 gives distance 1.0. Tensor methods only -- ``torch`` is
        # imported lazily in this module and is not in scope here.
        return (
            (q + sp * off) * vp,
            (q - sn * off) * vn,
            sp.clamp(min=1).to(self.cos.dtype),
            sn.clamp(min=1).to(self.cos.dtype),
            vp,
            vn,
        )


class _MessageBlock:
    """Static selections for one tiled block of a geometry's message pass."""

    __slots__ = (
        "valid",
        "live_idx",
        "ip",
        "in_",
        "cos",
        "logdp",
        "logdn",
        "side_idx",
        "side_positive",
        "n_live",
        "n_side",
        "L",
        "M",
    )


def _build_message_blocks(geom, torch):
    """Precompute all geometry-only selections consumed by ``GNN.embed``.

    In particular, CUDA ``nonzero`` and its data-dependent output size stay out
    of the compiled/repeated model path.  Blocks mirror ``embed``'s query
    tiling, so their flattened indices are already local to each tile.
    """
    blocks = []
    for m0 in range(0, geom.M, _M_TILE):
        m1 = min(m0 + _M_TILE, geom.M)
        ip, in_, dp, dn, vp, vn = geom.lines(m0, m1)
        L, M = vp.shape
        valid = vp | vn
        live_idx = valid.reshape(-1).nonzero(as_tuple=True)[0]
        line_of = live_idx // M
        vp_f = vp.reshape(-1)[live_idx]
        vn_f = vn.reshape(-1)[live_idx]

        block = _MessageBlock()
        block.valid = valid
        block.live_idx = live_idx
        block.ip = ip.reshape(-1)[live_idx]
        block.in_ = in_.reshape(-1)[live_idx]
        block.cos = geom.cos[line_of]
        lognnz = geom.lognnz[line_of, 0]
        block.logdp = torch.log2(dp.reshape(-1)[live_idx]) + lognnz
        block.logdn = torch.log2(dn.reshape(-1)[live_idx]) + lognnz
        block.side_idx = (vp_f ^ vn_f).nonzero(as_tuple=True)[0]
        block.side_positive = vp_f[block.side_idx]
        block.n_live = int(live_idx.numel())
        block.n_side = int(block.side_idx.numel())
        block.L = L
        block.M = M
        blocks.append(block)
    return blocks


_PERIOD_PREFIX_CACHE: dict = {}


def _period_prefixes(shape, levels, stride, block):
    """Periodic `known`-before-stage pattern for every stage, on one period tile
    (P=stride). Because each schedule mask is a per-axis residue condition mod a
    divisor of the anchor stride, the real `known` mask satisfies
    ``known[idx] == pat[idx % P]``; evaluating the schedule on a P-sized grid
    yields that period tile with no boundary truncation.

    Cached on the *rank* rather than ``shape``: the tile is ``(stride,) * ndim``,
    so two chunk shapes of the same rank share these patterns exactly. Grow mode
    gives every chunk of a small grid its own shape (extent stride+1 on internal
    high faces, stride on the domain face -> 2^ndim shapes), and each one used to
    rebuild all the stage masks of a full period tile. Returned arrays are shared
    and must be treated as read-only."""
    key = (len(shape), levels, stride, block)
    hit = _PERIOD_PREFIX_CACHE.get(key)
    if hit is not None:
        return hit
    P = stride
    tile = (P,) * len(shape)
    pats, cum = [], np.zeros(tile, bool)
    for mask in stage_masks(tile, levels, stride, block):
        pat = cum.copy()  # known BEFORE this stage
        pat.flags.writeable = False  # shared across chunk shapes
        pats.append(pat)
        cum |= mask
    _PERIOD_PREFIX_CACHE[key] = pats
    return pats


_GEOM_CACHE: dict = {}
_MODEL_CACHE: dict = {}


def build_stage_geoms(
    shape, levels, stride, block, max_radius, torch, device=None, agg_level=None
):
    """Per-stage `_StageGeom` list (empty stages dropped) plus a
    ``|known|-before-stage -> list index`` map, for the whole schedule of one
    region shape. Closed-form lattice geometry, computed at the query points
    only; cached per (shape, levels, stride, block, max_radius, agg_level,
    device) and shared by encoder tuning sweeps, decoder, and the trainer.

    ``agg_level`` caps the neighbourhood aggregation level (see
    `half_directions`); ``None`` keeps the full neighbourhood.

    ponytail: unbounded cache, bounded in practice (a handful of shapes/configs);
    add an LRU cap only if a caller feeds unboundedly many distinct configs."""
    key = (
        tuple(int(n) for n in shape),
        levels,
        stride,
        block,
        max_radius,
        agg_level,
        str(device),
    )
    hit = _GEOM_CACHE.get(key)
    if hit is not None:
        return hit
    shape = tuple(int(n) for n in shape)
    masks = stage_masks(shape, levels, stride, block)
    pats = _period_prefixes(shape, levels, stride, block)
    geoms, count_to_i, cum = [], {}, 0
    for s, mask in enumerate(masks):
        n = int(mask.sum())
        if n:  # empty stages get no predict call; skipping keeps counts unique
            Q = np.stack(np.nonzero(mask), axis=1)
            count_to_i[cum] = len(geoms)
            geoms.append(
                _StageGeom(pats[s], Q, shape, max_radius, torch, device, agg_level)
            )
        cum += n
    out = (geoms, count_to_i)
    _GEOM_CACHE[key] = out
    return out


def _mlp(torch, sizes):
    import torch.nn as nn

    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.GELU()]
    layers.pop()  # drop trailing activation
    return nn.Sequential(*layers)


def build_model(d: int = 32, agg_level: int = 2):
    """Construct the axial, dimension-agnostic GNN.

    ``agg_level`` is the neighbourhood aggregation level the model was trained
    for (see ``half_directions``): it fixes the set of discrete signed direction
    cosines the message pass can see — ``{+-1/sqrt(j) : j=1..agg_level} u {0}``,
    i.e. ``2*agg_level + 1`` states — and therefore the size of DirState's
    learned table. It is a property of the checkpoint, not an inference knob, so
    it must match the value the checkpoint was trained with (the loader supplies
    it from the checkpoint)."""
    import torch
    import torch.nn as nn

    F = nn.functional

    assert d % 2 == 0, "d must be even for the paired axis embeddings"
    L = int(agg_level)
    if L < 1:
        raise ValueError("agg_level must be >= 1")
    n_states = 2 * L + 1  # signed cosines {+-1/sqrt(j)} u {0}

    # Hidden width of the message/fusion/readout MLPs. Decoupled from d so the
    # per-axis field (memory ~ ndim * d) stays cheap while these functions keep
    # capacity — the wide activations are transient, over one stage's points.
    h = 2 * d
    # Direction-state embedding: a small learned vector per discrete signed
    # direction cosine, concatenated to each neighbour's axis embedding before
    # the message MLPs (see DirState). ``de`` is the resulting per-side width.
    ds = max(1, d // 4)
    de = d + ds

    class InitEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [1, d, d])

        def forward(self, v):  # v: (..., 1) normalized value
            return self.net(v)

    class DirState(nn.Module):
        """Concatenate a small learned direction-state vector to each
        neighbour's per-axis embedding, in place of a rotary phase.

        The model is single-hop (long range comes from the codec's stage
        schedule, not composed message-passing hops), and the lines are lattice
        directions, so the signed direction cosine ``sign * cos(line, axis k)``
        is not a continuous position — it takes one of a small, fixed set of
        values. A line of L1 length ``j`` (``j`` non-zero components) has
        ``|cos| = 1/sqrt(j)`` on its non-zero axes and 0 elsewhere, so over an
        aggregation level ``L`` the whole set is ``{+-1/sqrt(j) : j=1..L} u {0}``
        — ``2L + 1`` states. A learned table indexed by that state is both
        cheaper than the rotation (no per-channel sin/cos on the finest level,
        the bulk of the forward) and strictly more expressive than a fixed
        phase. The axis-k channel still reads its neighbours through their angle
        to axis k — now as a concatenated feature the message MLP may use freely
        — and perpendicular axes (the center state) get a distinct learned
        vector rather than the identity rotation.

        The ``2L + 1`` states are ordered by signed cosine ascending:
        ``-1, -1/sqrt2, .., -1/sqrt(L), 0, +1/sqrt(L), .., +1/sqrt2, +1`` at
        indices ``0..2L`` (center = L). ``nnz = round(1/v**2)`` recovers a
        pair's L1 length from its cosine, giving index ``2L-nnz+1`` (v>0),
        ``nnz-1`` (v<0), ``L`` (v==0). ``L`` is fixed by the checkpoint's
        aggregation level, and geometry is built at the same level, so every
        pair's ``nnz`` lands in ``[1, L]`` (the clamp is only a guard). Zeroing
        the table makes every axis channel identical again — the axial structure
        stays a pure, removable inductive bias (as the freq bank was)."""

        def __init__(self):
            super().__init__()
            self.table = nn.Parameter(torch.randn(n_states, ds) * ds**-0.5)

        def _index(self, cos, sign):
            # Bucket in fp32 for a stable index even when the message pass runs
            # in fp16 (round() tolerates the noise, but keep cos out of half).
            v = sign * cos.float()
            nnz = v.square().reciprocal().round()  # 1..L for v!=0; inf at v==0
            idx = torch.where(
                v > 0,
                2 * L - nnz + 1,
                torch.where(v < 0, nnz - 1, v.new_full((), float(L))),
            )
            return idx.clamp_(0, 2 * L).long()

        def forward_flat(self, e, cos, sign):
            # Flat pair layout for the live-pair message pass: e (B, N, K, d),
            # cos (N, K) -> (B, N, K, d + ds), one direction state per gathered
            # (line, point) pair, broadcast over the batch dim.
            idx = self._index(cos, sign)  # (N, K)
            state = self.table[idx].to(e.dtype)  # (N, K, ds)
            state = state.unsqueeze(0).expand(e.shape[0], *state.shape)
            return torch.cat([e, state], dim=-1)

    class DirEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [de + 2, h, d])

        def forward(self, e, sign, logd):  # one neighbour + (sign, log2 dist)
            # Split the first Linear over its input blocks instead of concatenating
            # e (B,L,M,K,de) with the two scalar columns — avoids materializing the
            # big (…,de+2) buffer (the `cat` was ~14% of GPU time). ``e`` already
            # carries the concatenated direction state (width de = d + ds).
            w = self.net[0].weight  # (h, de+2)
            x = F.linear(e, w[:, :de], self.net[0].bias) + F.linear(
                torch.cat([sign, logd], -1), w[:, de:]
            )
            for layer in self.net[1:]:
                x = layer(x)
            return x

    class BiDirEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [2 * de + 2, h, d])

        def forward(self, e_neg, e_pos, logd_neg, logd_pos):
            w = self.net[0].weight  # (h, 2de+2)
            x = (
                F.linear(e_neg, w[:, :de])
                + F.linear(e_pos, w[:, de : 2 * de], self.net[0].bias)
                + F.linear(torch.cat([logd_neg, logd_pos], -1), w[:, 2 * de :])
            )
            for layer in self.net[1:]:
                x = layer(x)
            return x

    class AttnPool(nn.Module):
        def __init__(self):
            super().__init__()
            # A Linear(d, d) key projection dotted with a fixed learned query
            # is just a linear functional of msgs, so a bare Linear(d, 1) here
            # would match it exactly. Keep a hidden GELU layer so the score is
            # a genuine nonlinear function of msgs rather than a degenerate
            # linear fusion.
            self.wk = _mlp(torch, [d, d, 1])
            self.wv = nn.Linear(d, d)
            self.null_k = nn.Parameter(torch.randn(()) * d**-0.5)
            self.null_v = nn.Parameter(torch.zeros(d))

        def forward(self, msgs, valid):
            # msgs: (L, B, N, d); valid: (L, N) bool
            v = self.wv(msgs)
            scale = self.wv.in_features**-0.5
            scores = self.wk(msgs).squeeze(-1) * scale  # (L, B, N)
            scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
            L, B, N, dd = msgs.shape
            sn = self.null_k * scale
            scores = torch.cat([scores, sn.expand(1, B, N)], dim=0)
            v = torch.cat([v, self.null_v.expand(1, B, N, dd)], dim=0)
            # Softmax in fp32 even when the model is fp16: -inf-masked scores
            # overflow/NaN in half. Cast the weights back to v's dtype so the
            # weighted sum stays in the model dtype (was autocast's fp32 rule).
            w = torch.softmax(scores.float(), dim=0).to(v.dtype)  # (L+1, B, N)
            return (w.unsqueeze(-1) * v).sum(0)  # (B, N, d)

    class PredHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [d + 1, h, 2])
            # Start unconfident: delta ~= 8 (b ~= 256*eb), inside the clamp so
            # gradients flow both ways. A zero init means b ~= eb, and with
            # untrained predictions (|r| ~ 0.1) at eb=1e-6 the NLL tail then
            # fires ~1/b gradient spikes from step 1.
            with torch.no_grad():
                self.net[-1].bias[1] = 8.0

        def forward(self, e, eb):
            B, M, _ = e.shape
            # A python-float eb must NOT go through ``as_tensor``: that is a
            # pageable host->device copy, which blocks until the stream drains.
            # As the first op of every stage forward it caps CPU run-ahead in a
            # launch-bound encode (measured: 0.04ms on an idle stream, 430ms
            # behind queued work). ``full`` passes the value as a kernel
            # argument instead -- same bits, no copy, no sync.
            if isinstance(eb, (int, float)):
                eb = torch.full((1,), float(eb), dtype=e.dtype, device=e.device)
            else:
                eb = torch.as_tensor(eb, dtype=e.dtype, device=e.device).reshape(-1)
            if eb.numel() == 1:
                eb = eb.expand(B)
            elif eb.numel() != B:
                raise ValueError(f"eb has {eb.numel()} entries for batch {B}")
            log_eb = torch.log2(eb.clamp_min(torch.finfo(e.dtype).tiny))
            cond = log_eb.view(B, 1, 1).expand(B, M, 1)
            out = self.net(torch.cat([e, cond], dim=-1))
            mu = torch.sigmoid(out[..., 0])
            # Laplace scale is eb-relative: `delta` spans the deployed rANS scale
            # grid [eb/16, 4096*eb] (log2 offsets -4..12, see rans.SCALE_LO_DIV/
            # SCALE_HI_MULT), so the head can express sub-eb confidence at ANY eb.
            # The old span-relative clamp(-8,0) pinned every point to the broadest
            # grid levels at low eb; the earlier eb-relative ceiling of +6 pinned
            # ~half the points at very low eb (<=1e-5), where prediction error
            # stays orders of magnitude above eb (bench_levels sat+%).
            delta = out[..., 1].clamp(-4.0, 12.0)
            log_b = log_eb.view(B, 1) + delta
            return mu, log_b

    class MixEmbed(nn.Module):
        """Fuse a point's pooled neighbour context with the embedding of its
        own now-known value into the finalized embedding stored in the field.
        The value carries the small residual-coding error (noise in training),
        so this lets the field remember what was actually reconstructed there
        rather than the raw prediction."""

        def __init__(self):
            super().__init__()
            self.net = _mlp(torch, [2 * d, h, d])

        def forward(self, ctx, value_emb):  # (B, N, d), (B, N, d) -> (B, N, d)
            return self.net(torch.cat([ctx, value_emb], dim=-1))

    class GNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d
            self.init = InitEmbed()
            self.dir_state = DirState()
            self.dir = DirEmbed()
            self.bidir = BiDirEmbed()
            self.line_pool = AttnPool()
            self.axis_pool = AttnPool()
            self.head = PredHead()
            self.mix = MixEmbed()

        def _line_messages(self, E, block):
            """Per-line messages for one precomputed query ``block``, built
            from neighbour *embeddings* E (not raw values) so
            trends/periodicity propagate hop by hop. Returns msgs (L, B, m, K, d)
            and valid (L, m) — the axis dim K is carried through Dir/BiDir as a
            batch dim, each axis reading its neighbours through the direction
            state.
            The block holds the geometry-only selections, so this touches O(m)
            rows of the field and performs no data-dependent index discovery.

            Live-pair gather: a (line, point) pair with no valid neighbour on
            either side (``~vp & ~vn``) is masked out by line_pool anyway, yet the
            dense form ran dir_state/dir/bidir over it regardless — ~47% of pairs are
            dead at these schedules (75% at the coarsest sub-stages), and
            torch.compile can't skip masked GEMM rows. So gather the live pairs
            first and run the message MLPs only on them, splitting two-sided pairs
            (bidir) from one-sided (dir) so neither MLP runs on rows the other
            owns. line_pool already masks dead pairs, so the pooled result is
            bit-identical to the old dense dir + where(both, bidir) form."""
            # Keep the private geometry call form used by diagnostics/tests for
            # untiled stages; the hot path passes the block directly.
            if hasattr(block, "message_blocks"):
                if len(block.message_blocks) != 1:
                    raise ValueError("direct _line_messages call requires one tile")
                block = block.message_blocks[0]
            B = E.shape[0]
            K, d = E.shape[2], self.d
            L, m, Nl = block.L, block.M, block.n_live
            if not Nl:  # no neighbours anywhere
                msg = E.new_zeros(B, L, m, K, d)
                return (msg.permute(1, 0, 2, 3, 4).contiguous(), block.valid)
            # Distances are precomputed in fp32; match model dtype so the
            # dir/bidir MLPs get fp16 inputs under true-half.
            lp = block.logdp.to(E.dtype)
            lnn = block.logdn.to(E.dtype)
            ep = self.dir_state.forward_flat(E[:, block.ip], block.cos, 1.0)
            en = self.dir_state.forward_flat(E[:, block.in_], block.cos, -1.0)
            # bidir over every live pair — two-sided is ~93% of them, so running
            # it on the ~3.6% one-sided too (then overwriting) is far cheaper than
            # a second gather, and peak memory stays at the live-pair size (<=
            # dense). One-sided pairs read a dummy row on their missing side; the
            # dir overwrite below discards that, exactly as the old `where` did.
            lp_e = lp.view(1, Nl, 1, 1).expand(B, Nl, K, 1)
            lnn_e = lnn.view(1, Nl, 1, 1).expand(B, Nl, K, 1)
            msg_live = self.bidir(en, ep, lnn_e, lp_e)  # (B, Nl, K, d)
            if block.n_side:
                s_idx = block.side_idx
                ns = block.n_side
                vpo = block.side_positive.view(1, ns, 1, 1)  # only + valid?
                e_side = torch.where(vpo, ep[:, s_idx], en[:, s_idx])
                one = e_side.new_ones(())
                sign = torch.where(vpo, one, -one).expand(B, ns, K, 1)
                lp_s = lp[s_idx].view(1, ns, 1, 1).expand(B, ns, K, 1)
                lnn_s = lnn[s_idx].view(1, ns, 1, 1).expand(B, ns, K, 1)
                msg_live[:, s_idx] = self.dir(
                    e_side, sign, torch.where(vpo, lp_s, lnn_s)
                ).to(msg_live.dtype)
            msg = E.new_zeros(B, L * m, K, d, dtype=msg_live.dtype)  # dead pairs stay 0
            msg[:, block.live_idx] = msg_live
            msg = msg.reshape(B, L, m, K, d)
            return (msg.permute(1, 0, 2, 3, 4).contiguous(), block.valid)

        def _embed_block(self, E, block):
            msgs, valid = self._line_messages(E, block)  # (L,B,m,K,d),(L,m)
            L, B, m, K, _ = msgs.shape
            flat = msgs.reshape(L, B, m * K, self.d)
            vflat = valid.repeat_interleave(K, dim=1)  # (L, m*K)
            ctx = self.line_pool(flat, vflat)  # (B, m*K, d)
            return ctx.reshape(B, m, K, self.d)  # (B, m, ndim, d)

        def embed(self, E, geom):
            """Per-axis contexts at geom's query points: single-query attention
            over the per-line neighbour messages (no self value), pooled
            independently per axis. For an anchor with no known neighbours every
            line is masked and each axis falls back to the learned null token.

            Tiled over the query dim: the transient (B, L, m, K, d) message
            buffers and their 2*d-wide MLP activations are the dominant GPU peak
            at the finest stages, so we cap m at _M_TILE and stream blocks into a
            small (B, M, K, d) output. line_pool is per-query independent, so
            this is bit-identical to embedding all M at once."""
            M = geom.M
            if M <= _M_TILE:
                return self._embed_block(E, geom.message_blocks[0])
            ctx = E.new_empty(E.shape[0], M, geom.ndim, self.d)
            for bi, m0 in enumerate(range(0, M, _M_TILE)):
                m1 = min(m0 + _M_TILE, M)
                ctx[:, m0:m1] = self._embed_block(E, geom.message_blocks[bi])
            return ctx

        def finalize(self, ctx, self_val):
            """Finalized embedding for points whose value has just been
            revealed: embed the (noisy) known value with InitEmbed and fuse it
            into every axis context via MixEmbed. `self_val` is the
            reconstructed value — truth + noise in training, the quantised
            recon at inference — so the mix learns to trust it up to eb."""
            if ctx.dim() != 4:
                raise ValueError(
                    f"finalize requires axial context (B, M, ndim, d), got "
                    f"shape {tuple(ctx.shape)}"
                )
            value_emb = self.init(self_val.unsqueeze(-1)).unsqueeze(2)
            return self.mix(ctx, value_emb.expand_as(ctx))

        def head_of(self, ctx, eb):
            # The predicted value (mu, log_b) is what gets quantized against, so
            # keep the readout in fp32 even under fp16 autocast — confines fp16 to
            # the message pass and protects compression ratio at small eb.
            with torch.autocast(device_type=ctx.device.type, enabled=False):
                ctx = ctx.float()
                K, M = ctx.shape[2], ctx.shape[1]
                msgs = ctx.permute(2, 0, 1, 3)
                valid = torch.ones(K, M, dtype=torch.bool, device=ctx.device)
                return self.head(self.axis_pool(msgs, valid), eb)

    return GNN()


def file_sha256(path) -> bytes:
    """Streamed sha256 of a file; read_bytes() would hold it all in RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.digest()


def _load_inference_model(checkpoint_path, torch, device):
    """Load immutable inference weights once per checkpoint revision/device."""
    path = Path(checkpoint_path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, str(device))
    hit = _MODEL_CACHE.get(key)
    if hit is not None:
        return hit
    # One live revision per (path, device): the trainer overwrites the eval
    # checkpoint every eval, so keying on mtime alone leaks a model (and its
    # compiled embed) per revision until OOM.
    for k in [k for k in _MODEL_CACHE if k[0] == key[0] and k[3] == key[3]]:
        del _MODEL_CACHE[k]

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    version = int(ckpt.get("version", 1))
    if version != CKPT_VERSION:
        raise ValueError(
            "Axial GNN checkpoint format v7 is required. Retrain with "
            "scripts/train_gnn.py."
        )
    d = int(ckpt["d"])
    # Aggregation level is frozen at training time and sizes the direction-state
    # table, so it must come from the checkpoint (not an inference argument).
    if "agg_level" not in ckpt:
        raise ValueError(
            "checkpoint is missing 'agg_level'; retrain with scripts/train_gnn.py."
        )
    agg_level = int(ckpt["agg_level"])
    model = build_model(d, agg_level).eval()
    model.load_state_dict(ckpt["state_dict"])
    del ckpt  # a second CPU copy of the weights, dead once loaded
    model.to(device)
    checkpoint_hash = file_sha256(path)[:16]
    out = (d, model, checkpoint_hash, agg_level)
    _MODEL_CACHE[key] = out
    return out


def stage_forward(
    model, E, geom_prev, geom_head, finalize_vals, torch, finalize_ctx=None, eb=0.01
):
    """One codec stage of the propagating GNN: finalize the previous stage's
    revealed points, then predict this stage's queries. Returns
    ``((mu, log_b), E, head_ctx)``; ``head_ctx`` feeds the next stage's
    ``finalize_ctx``."""
    if geom_prev is not None and geom_prev.M:
        ctx = finalize_ctx if finalize_ctx is not None else model.embed(E, geom_prev)
        finalized = model.finalize(ctx, finalize_vals).to(E.dtype)  # fp16 -> E dtype
        # ponytail: in-place. no grad here, and each model.embed re-stages E, so
        # mutating it between calls is safe; out-of-place clones E every stage
        # (~200ms of DtoD at chunk 32). Only unsafe if E were a CUDA-graph static
        # buffer, but reduce-overhead re-copies inputs, so eager E stays mutable.
        E.index_copy_(1, geom_prev.query_idx, finalized)  # write newly-known
    head_ctx = model.embed(E, geom_head)
    return model.head_of(head_ctx, eb), E, head_ctx


# ---------------------------------------------------------------------------
# Chunked inference: the tensor is coded chunk by chunk (global anchors first),
# dense embeddings exist only for the current chunk + halo; out-of-chunk
# neighbours are represented from their reconstructed value alone (see
# value_halo_embed / ChunkedGNNPredictor).
# ---------------------------------------------------------------------------


class _ChunkGeoms:
    """Stage geometry and index metadata for one chunk shape, in the
    halo-padded local frame (halo = ``anchor_stride`` on every side — every
    valid periodic neighbour is within ``anchor_stride`` steps, see
    `_nearest_steps`). Origin-independent: chunk origins are multiples of the
    stride and the halo equals it, so local coordinates are congruent to global
    ones mod the pattern period and every aligned chunk of the same shape
    shares this object (cached in ``build_chunk_geoms``)."""

    def __init__(
        self,
        chunk_shape,
        levels,
        stride,
        block,
        torch,
        device,
        agg_level=None,
        progress=None,
    ):
        self.chunk_shape = tuple(int(n) for n in chunk_shape)
        self.levels, self.stride, self.block = levels, stride, block
        self.agg_level = agg_level
        self.halo = stride
        ndim = len(self.chunk_shape)
        self.ndim = ndim
        self.padded_shape = tuple(n + 2 * stride for n in self.chunk_shape)
        self.n_padded = int(np.prod(self.padded_shape))

        masks = stage_masks(self.chunk_shape, levels, stride, block)
        schedule_strides = stage_strides(ndim, levels, stride)
        use_axial = (
            _AXIAL_IMPLICIT_GEOMETRY and agg_level == 1 and block == 1
        )
        pats = (
            [None] * len(masks)
            if use_axial
            else _period_prefixes(self.chunk_shape, levels, stride, block)
        )
        self.geoms = []  # per stage; None for empty stages
        self._offsets: dict = {}  # tensor strides -> per-stage flat offsets
        host_halo = int(np.prod(self.chunk_shape)) >= _HOST_HALO_MIN_POINTS
        seen_halo = []
        for s, (mask, stage_stride) in enumerate(zip(masks, schedule_strides)):
            Q = np.stack(np.nonzero(mask), axis=1)  # chunk-frame coords
            if not len(Q):
                self.geoms.append(None)
                if progress is not None:
                    progress(1)
                continue
            geom = _StageGeom(
                pats[s],
                Q + stride,
                self.padded_shape,
                stride,
                torch,
                device,
                agg_level,
                query_only=True,
                precompute_messages=False,
                ref_bounds=(
                    (
                        (stride,) * ndim,
                        tuple(stride + n for n in self.chunk_shape),
                    )
                    if host_halo
                    else None
                ),
                axial_stride=(
                    (0 if s == 0 else stage_stride)
                    if use_axial
                    else None
                ),
                period=stride if use_axial else None,
            )
            self.geoms.append(geom)
            if len(geom.ref_halo_np):
                seen_halo.append(geom.ref_halo_np)
            if progress is not None:
                progress(1)
        # prediction chain: stage 0 is always the base (anchors, possibly empty
        # in a ragged tail chunk -> None geom, nothing to finalize), followed by
        # every non-empty refinement stage in order.
        self.chain = [0] + [
            s for s in range(1, len(self.geoms)) if self.geoms[s] is not None
        ]

        # Interior padded-flat indices, built directly (O(interior), never the
        # O(shell) full padded grid). The compact field lays interior first, so
        # interior cell i has compact index i + 1 (row 0 is a dummy the invalid
        # / not-yet-decoded neighbour lines point at).
        idx0 = np.indices(self.chunk_shape).reshape(ndim, -1)  # chunk-frame
        self.interior_flat = np.ravel_multi_index(idx0 + stride, self.padded_shape)

        # Padded-flat halo cells that appear as a *valid* neighbour of some
        # stage: the thin band the field must actually hold. Derived from the
        # stage geometries (O(interior)), so the dead rest of the shell is never
        # materialised. Its chunk-frame coords let the per-chunk halo pass test
        # usability without an O(shell) mask.
        if host_halo:
            ref = (
                np.unique(np.concatenate(seen_halo))
                if seen_halo
                else np.zeros(0, np.int64)
            )
        else:
            seen = []
            for g in self.geoms:
                if g is None:
                    continue
                ip, in_, _, _, vp, vn = g.lines(0, g.M)
                seen.append(ip[vp])
                seen.append(in_[vn])
            ref = (
                torch.unique(torch.cat(seen)).cpu().numpy()
                if seen
                else np.zeros(0, np.int64)
            )
        # "Not interior" is a box test on padded coords, not a set membership
        # test: interior_flat holds every cell of the chunk, so `isin` against it
        # sorts ~1M indices per shape and cost more than the rest of this
        # constructor. Unravel `ref` once and reuse it for the coords below.
        rc = np.stack(np.unravel_index(ref, self.padded_shape), 1) - stride
        outside = np.zeros(len(ref), bool)
        for k in range(ndim):
            outside |= (rc[:, k] < 0) | (rc[:, k] >= self.chunk_shape[k])
        self.ref_halo_flat = ref[outside].astype(np.int64)
        self.ref_halo_coords = rc[outside]

    def stage_offsets(self, strides):
        """Per-stage flat offsets of a stage's points from the chunk origin, in
        the *full tensor's* index space (``None`` for empty stages).

        Adding a chunk's origin-flat gives the recon rows the stage reads, which
        is the only thing the chunk-frame coordinates were ever used for. Keeping
        the coordinates themselves would cost ``ndim`` int64 columns per point
        for every cached chunk shape -- two thirds of this object -- so they are
        recovered on demand from the stage's padded-flat query index and the
        result is memoized per stride vector (one vector per encode)."""
        key = tuple(int(s) for s in strides)
        off = self._offsets.get(key)
        if off is None:
            pst = np.cumprod((1,) + self.padded_shape[:0:-1])[::-1].astype(np.int64)
            off = []
            for g in self.geoms:
                if g is None:
                    off.append(None)
                    continue
                # Unravel and re-ravel in one sweep, axis by axis, so the (M, ndim)
                # coordinate block never exists -- it would be as large as the
                # array this method exists to avoid storing.
                rem = g.idx_np.astype(np.int64, copy=True)
                o = np.zeros(len(rem), np.int64)
                for k in range(self.ndim):
                    ck = rem // pst[k]
                    rem -= ck * pst[k]
                    o += (ck - self.halo) * key[k]
                off.append(o)
            self._offsets[key] = off
        return off


def build_chunk_geoms(
    chunk_shape,
    levels,
    stride,
    block,
    torch,
    device=None,
    agg_level=None,
    progress=None,
    cache=None,
):
    """`_ChunkGeoms` for one chunk shape, memoised in the caller's ``cache``
    dict (``None`` builds a fresh one) under (chunk shape, schedule, agg_level,
    device). The cache is the caller's: `ChunkedGNNPredictor` owns one and the
    codec may retain that predictor to reuse geometry across calls.
    Interior chunks all share one entry; ragged edge chunks add at most a few
    shape variants.

    ``agg_level`` caps the neighbourhood aggregation level (see
    `half_directions`); ``None`` keeps the full neighbourhood."""
    key = (
        tuple(int(n) for n in chunk_shape),
        levels,
        stride,
        block,
        agg_level,
        str(device),
    )
    hit = None if cache is None else cache.get(key)
    if hit is None:
        hit = _ChunkGeoms(
            chunk_shape, levels, stride, block, torch, device, agg_level, progress
        )
        if cache is not None:
            cache[key] = hit
    elif progress is not None:
        progress(len(hit.geoms))
    return hit


def _build_remap(present, n_padded, torch, device):
    """Map padded-flat index -> compact field row. ``present`` lists the padded
    cells the field holds (interior first, then usable halo); cell ``present[i]``
    lives at row ``i + 1`` (row 0 is the dummy that invalid / not-yet-decoded
    neighbour lines point at). A dense table over the padded frame makes each
    remap one gather instead of a sort + searchsorted + compare chain.
    ponytail: table capped at 2^24 cells (~128MB, covers 4-D chunks at stride
    32); bigger frames take the searchsorted path."""
    if n_padded <= 1 << 24:
        table = np.zeros(n_padded, np.int64)
        table[present] = np.arange(1, len(present) + 1)
        tt = torch.from_numpy(table).to(device)
        return lambda flat: tt[flat]

    order = np.argsort(present, kind="stable")
    spres = torch.from_numpy(np.ascontiguousarray(present[order])).to(device)
    comp = torch.from_numpy((order + 1).astype(np.int64)).to(device)
    n = spres.numel()

    def remap(flat):  # (torch int64) -> (torch int64) compact
        if n == 0:
            return torch.zeros_like(flat)
        pos = torch.searchsorted(spres, flat).clamp_(max=n - 1)
        return torch.where(spres[pos] == flat, comp[pos], torch.zeros_like(flat))

    return remap


class _CompactGeom:
    """A `_StageGeom` with its neighbour indices remapped into a chunk's compact
    field and its periodic validity ANDed with runtime usability. A neighbour is
    usable iff it landed on a real compact row (remap != 0): interior always
    does, halo only when that cell is decoded and referenced. Shares every other
    tensor with the base geometry."""

    __slots__ = (
        "ip",
        "in_",
        "dp",
        "dn",
        "vp",
        "vn",
        "cos",
        "lognnz",
        "query_idx",
        "idx_np",
        "M",
        "ndim",
        "message_blocks",
    )

    def __init__(self, base, remap, torch):
        for name in ("cos", "lognnz", "idx_np", "M", "ndim"):
            setattr(self, name, getattr(base, name))
        ip, in_, self.dp, self.dn, vp, vn = base.lines(0, base.M)
        self.ip = remap(ip)
        self.in_ = remap(in_)
        self.query_idx = remap(base.query_idx)
        self.vp = vp & (self.ip != 0)
        self.vn = vn & (self.in_ != 0)
        self.message_blocks = _build_message_blocks(self, torch)

    lines = _slice_lines


class _CompactFrame:
    """Per-chunk compact field layout: geoms with remapped indices, and the
    (contiguous) halo row block plus the metadata to fill it from the halo
    cells' reconstructed values. ``n_compact`` = 1 dummy + interior +
    usable-referenced halo."""

    __slots__ = (
        "geoms",
        "n_interior",
        "n_compact",
        "halo_rows",
        "h_gflat",
        "h_offsets",
    )

    def __init__(self, cg, origin, shape, edges, grid, coded, torch, device):
        halo_present, h_gflat = chunk_halo_info(cg, origin, shape, edges, grid, coded)
        self.n_interior = int(len(cg.interior_flat))
        present = np.concatenate([cg.interior_flat, halo_present])
        self.n_compact = 1 + len(present)
        remap = _build_remap(present, cg.n_padded, torch, device)
        self.geoms = [
            None if g is None else _CompactGeom(g, remap, torch) for g in cg.geoms
        ]
        # halo cells are laid out right after interior, so their rows are a
        # contiguous slice — no remap needed to fill them.
        self.halo_rows = slice(self.n_interior + 1, self.n_compact)
        self.h_gflat = h_gflat
        strides = np.cumprod((1,) + tuple(shape)[:0:-1])[::-1].astype(np.int64)
        origin_base = int(np.asarray(origin, np.int64) @ strides)
        self.h_offsets = torch.from_numpy(
            np.ascontiguousarray(h_gflat.astype(np.int64) - origin_base)
        ).to(device)


def _storage_bytes(obj, seen=None):
    """Array/tensor storage retained by a nested cache value.

    Count host tensors and NumPy arrays as well as CUDA tensors. The former
    device-only accounting made every CPU cache entry appear free and ignored
    the host half of mixed geometry objects.
    """
    seen = set() if seen is None else seen
    if obj is None or id(obj) in seen:
        return 0
    seen.add(id(obj))
    if hasattr(obj, "element_size") and hasattr(obj, "nelement"):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(_storage_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_storage_bytes(v, seen) for v in obj)
    slots = getattr(type(obj), "__slots__", ())
    total = sum(_storage_bytes(getattr(obj, name, None), seen) for name in slots)
    attrs = getattr(obj, "__dict__", None)
    return total + (_storage_bytes(attrs, seen) if attrs is not None else 0)


def chunk_halo_info(cg, origin, shape, edges, grid, coded):
    """Usable, referenced halo cells for one chunk of a chunk grid.

    Walks only the referenced band ``cg.ref_halo_flat`` (never the O(shell)
    padded frame). Returns ``(halo_present, gflat)`` for the band cells that
    are inside the tensor and already decoded (coded chunk, or a global
    anchor): their padded flat index and global flat index. Shared by the
    inference predictor and the trainer so both build identical context."""
    ndim = len(shape)
    gc = cg.ref_halo_coords + np.asarray(origin, np.int64)
    shp = np.asarray(shape)
    inb = np.all((gc >= 0) & (gc < shp), axis=1)
    gci = gc[inb]
    chunk_ids = np.ravel_multi_index([gci[:, k] // edges[k] for k in range(ndim)], grid)
    lv = point_levels([gci[:, k] for k in range(ndim)], cg.levels, cg.stride, cg.block)
    ok = np.asarray(coded)[chunk_ids] | (lv == 0)
    halo_present = cg.ref_halo_flat[inb][ok]
    gflat = np.ravel_multi_index([gci[ok][:, k] for k in range(ndim)], shape)
    return halo_present, gflat


def value_halo_embed(model, vals, ndim):
    """Representation of an out-of-chunk known neighbour: the reconstructed
    value's InitEmbed broadcast over the ndim axes. The extended-block schedule
    carries cross-chunk context through the decoded recon array, so the halo
    needs nothing more than the value itself. ``vals``: (B, H) normalized ->
    (B, H, ndim, d)."""
    ve = model.init(vals.unsqueeze(-1))  # (B, H, d)
    return ve.unsqueeze(2).expand(*ve.shape[:-1], ndim, ve.shape[-1])


class ChunkedGNNPredictor:
    """Chunk-by-chunk GNN predictor with bounded memory.

    Coding order (mirrored bitwise by the decoder): a global anchor pass, then
    chunks in raster order (extended-block partition -- see ``chunk_slices`` --
    so each chunk owns its internal high-face planes and inherits its low faces
    from already-decoded neighbours). Each chunk runs its local stage schedule
    with a dense embedding field over chunk + ``anchor_stride`` halo only; halo
    neighbours are embedded from their reconstructed value alone
    (``value_halo_embed``). Everything model-sized is O(chunk); the only O(N)
    state is the caller's recon array, which carries all cross-chunk context.

    Per-tensor protocol driven by the codec (encode and decode identically):
        begin(shape, chunk_edges, channels)
        for each wave (raster order):
            start_wave(chunk_ids, recon)
            for each non-empty local stage s >= 1, in order:
                pred, scale = predict_wave_stage(s, recon, eb)
                ... caller quantizes and writes recon ...
            finish_wave(recon)
    """

    provides_scale = True
    fp16 = False  # fp16 autocast on the message pass (codec sets it)
    compile = False  # torch.compile the embed pass (codec sets it)

    def _maybe_compile(self):
        # Wrap the embed pass once (fuses the elementwise message-pass ops that
        # aren't in the GEMMs). dynamic=True keeps one graph across all M sizes;
        # enc and dec both compile (flag replayed) so their float paths match.
        if self.compile and not getattr(self, "_compiled", False):
            # LATEC_COMPILE_MODE=reduce-overhead -> CUDA graphs, kills per-kernel
            # launch latency on the ~30 tiny message-pass kernels (launch-bound).
            # ponytail: CUDA graphs want static shapes; with varying stage M they
            # recapture per new shape, so it only wins once shapes settle/repeat.
            mode = os.environ.get("LATEC_COMPILE_MODE") or None
            self.model.embed = self._torch.compile(
                self.model.embed, dynamic=True, mode=mode
            )
            self._compiled = True

    def __init__(
        self,
        checkpoint_path,
        vmin: float,
        vmax: float,
        device: str = "cpu",
        levels: int = 4,
        anchor_stride: int = 16,
        anchor_block: int = 1,
    ):
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        span = self.vmax - self.vmin
        self.span = span if span > 0 else 1.0
        self.levels = int(levels)
        self.anchor_stride = int(anchor_stride)
        self.anchor_block = int(anchor_block)
        # Chunk geometry is reused across chunks and across begin() calls. The
        # owning codec caches this predictor because rebuilding rank-4 geometry
        # can cost more than the model pass itself.
        self._geom_cache: dict = {}
        # Compact remaps repeat across encode/decode and across interior chunks
        # with the same low-face topology. Keep a byte-bounded LRU: these device
        # tensors are expensive to rebuild but can be large at rank 4.
        self._frame_cache: OrderedDict = OrderedDict()
        self._cache_signature = None
        # Neighbourhood aggregation level (see half_directions),
        # frozen into the checkpoint at training time.
        self.d, self.model, self.checkpoint_hash, self.agg_level = (
            _load_inference_model(checkpoint_path, torch, self.device)
        )

    # -- per-tensor lifecycle -------------------------------------------------
    def begin(self, shape, chunk_edges, channels: int = 1, geometry_progress=None):
        self.shape = tuple(int(n) for n in shape)
        self.edges = tuple(int(e) for e in chunk_edges)
        if len(self.edges) != len(self.shape):
            raise ValueError("chunk_edges must have one entry per axis")
        for e in self.edges:
            if e < self.anchor_stride or e % self.anchor_stride:
                raise ValueError(
                    "chunk edges must be positive multiples of anchor_stride"
                )
        self.grid = tuple(-(-n // e) for n, e in zip(self.shape, self.edges))
        self.n_chunks = int(np.prod(self.grid))
        self.C = int(channels)
        signature = (self.shape, self.edges)
        if signature != self._cache_signature:
            self._geom_cache.clear()
            self._frame_cache.clear()
            self._cache_signature = signature
        ndim = len(self.shape)
        self._check_field_budget(ndim, channels, geometry_progress)
        self.coded = np.zeros(self.n_chunks, bool)
        self._cg = None

    def _check_field_budget(self, ndim, channels, geometry_progress=None):
        """Warn when the static estimate exceeds available memory.

        The estimate is deliberately advisory: allocator reuse and the number
        of simultaneously live intermediates can make it overly conservative,
        so the caller is allowed to proceed and let the backend enforce the
        real memory limit.
        """
        torch = self._torch
        cshape = tuple(min(e, n) for e, n in zip(self.edges, self.shape))
        cg = build_chunk_geoms(
            cshape,
            self.levels,
            self.anchor_stride,
            self.anchor_block,
            torch,
            self.device,
            self.agg_level,
            geometry_progress,
            cache=self._geom_cache,
        )
        n_interior = int(len(cg.interior_flat))
        n_band = int(len(cg.ref_halo_flat))  # upper bound (all referenced)
        field_bytes = channels * (1 + n_interior + n_band) * ndim * self.d * 4
        M = max((g.M for g in cg.geoms if g is not None), default=0)
        m = min(M, _M_TILE)
        L = len(half_directions(ndim, self.agg_level))
        act_bytes = 4 * channels * L * m * ndim * self.d * 4  # ~4 live copies
        need = field_bytes + act_bytes
        if self.device.type == "cuda":
            budget = _cuda_working_budget(torch, self.device)
        else:  # cpu: cap at 64 GiB, still catches it
            budget = 64 << 30
        if need > budget:
            warnings.warn(
                f"chunked GNN needs up to {need / 1e9:.0f} GB per chunk "
                f"(field {field_bytes / 1e9:.1f} + stage activation "
                f"{act_bytes / 1e9:.1f} GB for tile={m:,} of M={M:,} queries "
                f"x L={L} lines, budget {budget / 1e9:.1f} GB). Lower "
                f"LATEC_M_TILE or --chunk-size. Continuing because "
                f"this estimate is advisory.",
                RuntimeWarning,
                stacklevel=2,
            )

    def geometry_cached(self, shape, chunk_edges) -> bool:
        """Whether begin()'s representative chunk geometry is already resident."""
        cshape = tuple(min(int(e), int(n)) for e, n in zip(chunk_edges, shape))
        key = (
            cshape,
            self.levels,
            self.anchor_stride,
            self.anchor_block,
            self.agg_level,
            str(self.device),
        )
        return key in self._geom_cache

    def clear_runtime_cache(self):
        """Release tensor-shaped geometry while retaining the loaded model."""
        self._geom_cache.clear()
        self._frame_cache.clear()
        self._cache_signature = None

    def chunk_slices(self, ci: int):
        """Extended-block partition: each chunk's block is grown by one cell on
        every internal high face, so it *owns* (decodes) the shared boundary
        hyperplane with its right/bottom neighbours. The neighbour inherits that
        plane as an already-decoded low face (see ``low_axes``)."""
        cidx = np.unravel_index(ci, self.grid)
        sls = []
        for i, e, n in zip(cidx, self.edges, self.shape):
            hi = min((i + 1) * e, n)
            if (i + 1) * e < n:  # own the shared boundary plane on internal high faces
                hi = (i + 1) * e + 1
            sls.append(slice(i * e, hi))
        return tuple(sls)

    def low_axes(self, ci: int) -> tuple[int, ...]:
        """Axes on which this chunk has an internal *low* neighbour -- its local
        coord-0 hyperplane is that neighbour's already-decoded high face, so the
        codec column-splits those cells out of this chunk's coded set."""
        cidx = np.unravel_index(ci, self.grid)
        return tuple(a for a, i in enumerate(cidx) if i > 0)

    def _norm(self, vals: np.ndarray):
        v = (np.clip(vals, self.vmin, self.vmax) - self.vmin) / self.span
        return self._torch.from_numpy(v.astype(np.float32)).to(self.device)

    def _norm_t(self, vals_t):
        """Device-tensor twin of ``_norm``: normalize recon values already on the
        GPU, so the wave inner loop never round-trips recon through host memory.
        Both encoder and decoder call this, so the normalization is identical."""
        return (vals_t.clamp(self.vmin, self.vmax).float() - self.vmin) / self.span

    # ---- chunk-wave path ----------------------------------------------------

    def _amp(self):
        self._maybe_compile()
        if self.fp16 and self.device.type == "cuda":
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return contextlib.nullcontext()

    def start_wave(self, chunk_ids, recon):
        """Begin a batch of mutually-independent, identical-geometry chunks. One
        representative frame drives the shared stage geometry; the halo/interior
        field values are gathered per chunk into the model's B dim.

        ``recon`` is a device float32 tensor (C, *S): the wave inner loop keeps
        the reconstruction resident on the GPU so no per-stage host round-trip is
        needed. Encoder and decoder drive this path identically."""
        torch = self._torch
        ndim = len(self.shape)
        B = len(chunk_ids)
        origins = np.array(
            [[sl.start for sl in self.chunk_slices(ci)] for ci in chunk_ids], np.int64
        )  # (B, ndim)
        cshape = tuple(sl.stop - sl.start for sl in self.chunk_slices(chunk_ids[0]))
        cg = build_chunk_geoms(
            cshape,
            self.levels,
            self.anchor_stride,
            self.anchor_block,
            torch,
            self.device,
            self.agg_level,
            cache=self._geom_cache,
        )
        frame_key = (
            self.shape,
            self.edges,
            cshape,
            self.low_axes(chunk_ids[0]),
            _M_TILE,
        )
        cached = self._frame_cache.get(frame_key)
        if cached is None:
            frame = _CompactFrame(
                cg,
                origins[0],
                self.shape,
                self.edges,
                self.grid,
                self.coded,
                torch,
                self.device,
            )
            nbytes = _storage_bytes(frame)
            budget = _FRAME_CACHE_MIB * 2**20
            total = sum(nbytes for _, nbytes in self._frame_cache.values())
            while self._frame_cache and total + nbytes > budget:
                _, (_, dropped) = self._frame_cache.popitem(last=False)
                total -= dropped
            # The current wave holds ``frame.geoms`` independently. If one frame
            # exceeds the entire cache budget, use it once without pinning it.
            if nbytes <= budget:
                self._frame_cache[frame_key] = (frame, nbytes)
        else:
            frame = cached[0]
            self._frame_cache.move_to_end(frame_key)
            # The compact remap is origin-independent, but halo reconstruction
            # indices are global and shift with the current chunk.
            _, frame.h_gflat = chunk_halo_info(
                cg, origins[0], self.shape, self.edges, self.grid, self.coded
            )
        E = torch.zeros(B, frame.n_compact, ndim, self.d, device=self.device)
        self._wave_fill_halo(E, frame, origins, recon)
        # per-chunk global flat indices per stage, for finalize reads from recon.
        # ravel(c + o) = c @ strides + o @ strides for in-bounds coords, so one
        # dot per stage replaces the (B, M, ndim) ravel_multi_index. Kept as device
        # long tensors so the per-stage recon gather stays on the GPU.
        strides = np.cumprod((1,) + self.shape[:0:-1])[::-1].astype(np.int64)
        obase = origins @ strides  # (B,)
        self._wave_gidx = [
            None
            if off is None
            else torch.from_numpy(off[None, :] + obase[:, None]).to(self.device)
            for off in cg.stage_offsets(strides)
        ]  # (B, M) long
        self._cg = cg
        self._wave_ids = list(chunk_ids)
        self._E = E
        self._geoms = frame.geoms
        self._ctx = None
        self._pos = 0

    def _wave_fill_halo(self, E, frame, origins, recon):
        torch = self._torch
        ndim = len(self.shape)
        if frame.halo_rows.stop <= frame.halo_rows.start:
            return
        # The cached compact frame already stores halo offsets from its chunk
        # origin.  Only the global flat origin changes between chunks, so one
        # batched gather replaces per-wave unravel/ravel work and many tiny H2D
        # index transfers.
        strides = np.cumprod((1,) + self.shape[:0:-1])[::-1].astype(np.int64)
        origin_bases = origins @ strides
        bases_t = torch.from_numpy(
            np.ascontiguousarray(origin_bases[:, None])
        ).to(self.device)
        gflat = bases_t + frame.h_offsets[None, :]
        flat = recon.reshape(-1)  # C == 1, device
        vals_all = self._norm_t(flat[gflat])  # (B, H)
        with torch.inference_mode(), self._amp():
            emb = value_halo_embed(self.model, vals_all, ndim)
            E[:, frame.halo_rows] = emb.to(E.dtype)

    def predict_wave_stage(self, s: int, recon, eb: float):
        """Batched `predict_stage`: returns (pred, scale) of shape (B, M) for the
        wave's B chunks, ordered like ``np.nonzero`` of the local stage mask.

        ``recon`` is a device tensor and the returned (pred, scale) are device
        float32 tensors -- the codec quantizes and gates on the GPU, so nothing
        crosses to host on the per-stage critical path. Encoder and decoder run
        this identical arithmetic, so their reconstructions stay bit-consistent
        even though ``exp2`` here differs from numpy by a ULP."""
        torch = self._torch
        cg = self._cg
        j = self._pos + 1
        if j >= len(cg.chain) or cg.chain[j] != s:
            raise ValueError(f"stage {s} out of order for this wave")
        prev = cg.chain[j - 1]
        gp, gh = self._geoms[prev], self._geoms[s]
        fvals = (
            None
            if gp is None
            else self._norm_t(recon.reshape(-1)[self._wave_gidx[prev]])
        )  # (B, M)
        with torch.inference_mode(), self._amp():
            (values, log_b), self._E, self._ctx = stage_forward(
                self.model,
                self._E,
                gp,
                gh,
                fvals,
                torch,
                finalize_ctx=self._ctx,
                eb=float(eb) / self.span,
            )
        self._pos = j
        pred = (values.float() * self.span + self.vmin).clamp(
            self.vmin, self.vmax
        )  # (B, M) f32
        scale = torch.exp2(log_b.float()) * self.span
        return pred, scale

    def finish_wave(self, recon):
        cg = self._cg
        if self._pos != len(cg.chain) - 1:
            raise ValueError("finish_wave before all non-empty stages predicted")
        # The extended-block schedule carries cross-chunk context through the
        # decoded recon array (owned high faces, inherited low faces), so a
        # finished chunk leaves no per-chunk embedding state behind -- just mark
        # it coded and drop the dense field.
        self.coded[np.array(self._wave_ids)] = True
        self._E = self._ctx = self._cg = None
        # Compact geometries and global stage indices are wave-specific.  A
        # cached predictor must release them here before the next begin() builds
        # the decoder's frame, otherwise both complete frames overlap in memory.
        self._geoms = self._wave_gidx = self._wave_ids = None
