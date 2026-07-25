"""Edges-first chunk scheduling for the GNN codec.

The default chunked path (``gnn_codec._compress_chunked``) codes one chunk at a
time, all of its refinement levels before moving to a neighbour. A finest-level
interior point next to a chunk face therefore has no *decoded* fine neighbour on
the far side (that neighbour lives in a not-yet-coded chunk and is not a global
anchor), so it is predicted one-sided -- a prediction *seam* at every internal
chunk face. On smooth-but-sharp fields (S3D) the seam is expensive: the seam-free
whole-tensor path beats the chunked one by 17% (pure GNN) to 72% (gated) at the
same bound.

This module removes the seam while keeping bounded memory by splitting each
chunk's refinement points into two globally ordered phases:

    phase 1 (faces):   the cells lying on this chunk's *internal* low faces
                       (``coord % edge == 0`` and ``coord > 0`` on some axis).
                       Half-open chunks make every internal boundary hyperplane
                       the low face of exactly one chunk, so each face is coded
                       once. Coded from anchors + already-decoded face cells only.
    phase 2 (interiors): every other refinement cell, coded with *all* bounding
                       faces (own low faces + the neighbours' low faces, i.e. this
                       chunk's high faces) already decoded -> both-sided context.

Phase 2 is order-free: a face hyperplane sits on the anchor grid (``edge`` is a
multiple of ``anchor_stride``), so at every level the face cells are exactly the
+/-stride bracket of the interior cells adjacent to them -- a strict interior
never references a neighbour chunk's interior, only anchors and (decoded) faces.
So phase-2 chunks are mutually independent (any order, no color waves).

Unlike the ``skel`` codec this is built on, there is no separate global line pass:
the chunk-boundary *lines* are simply the level>=1 face cells coded in phase 1.
The out-of-chunk context (anchors / decoded faces) is embedded via
``anchor_finalize`` (value + null pooled context), so no per-chunk coarse table
is needed.
"""

from __future__ import annotations

import numpy as np

from .gnn_predictor import (
    _StageGeom,
    _CompactGeom,
    _build_message_blocks,
    _build_remap,
    anchor_finalize,
    build_chunk_geoms,
)
from .quantizer import dequantize, quantize
from .bitstream import pack_stage, unpack_stage
from .rans import build_laplace_tables, scale_to_level
from .levels import stage_plan
from .predictor import default_interp_center
from .gnn_codec import (
    _anchor_axes,
    _code_anchor_stage,
    _decode_anchor_stage,
    _dequantize_t,
    _gate_apply_t,
    _gate_select_t,
    _interp_stage_pred_t,
    _log,
    _progress_bar,
    _quantize_t,
)


# --- point classification --------------------------------------------------
def _offgrid_count(cshape, stride):
    """Per-cell count of coordinates off the anchor grid, chunk-local. Origins
    are multiples of the edge (a multiple of stride), so ``local % stride ==
    global % stride`` and the count is origin-independent."""
    ndim = len(cshape)
    cnt = np.zeros(cshape, np.int64)
    for k in range(ndim):
        off = np.arange(cshape[k]) % stride != 0
        shp = [1] * ndim
        shp[k] = cshape[k]
        cnt += off.reshape(shp).astype(np.int64)
    return cnt


def _offgrid_global(gc, stride):
    """Off-grid coordinate count for global coords ``gc`` (N, ndim)."""
    return (gc % stride != 0).sum(axis=1)


def _on_internal_boundary(gc, edges):
    """Whether each global coord (N, ndim) lies on >=1 *internal* chunk-boundary
    hyperplane: some axis ``a`` with ``coord_a % edge_a == 0`` and ``coord_a > 0``
    (coord 0 is the tensor border, not a shared chunk seam)."""
    on = np.zeros(len(gc), bool)
    for a in range(gc.shape[1]):
        on |= (gc[:, a] % edges[a] == 0) & (gc[:, a] > 0)
    return on


def _classify(cshape, origin, edges, stride):
    """Chunk-frame boolean masks splitting a chunk's *refinement* cells
    (>=1 off-grid; anchors are coded globally) into face and strict-interior:

        face   = >=1 off-grid AND on this chunk's internal low face (local coord
                 0 on an axis whose chunk index > 0). Coded in phase 1.
        strict = the remaining refinement cells. Coded in phase 2 with every
                 bounding face already decoded.
    """
    ndim = len(cshape)
    refine = _offgrid_count(cshape, stride) >= 1
    on_low_face = np.zeros(cshape, bool)
    for a in range(ndim):
        if origin[a] % edges[a] == 0 and origin[a] > 0:  # internal low face
            sl = [slice(None)] * ndim
            sl[a] = 0
            on_low_face[tuple(sl)] = True
    face = refine & on_low_face
    strict = refine & ~on_low_face
    return face, strict


# --- query-subset geometry -------------------------------------------------
class _SubsetStageGeom:
    """Query-row subset of a ``_StageGeom``. The neighbour search depends only on
    the base stage and the query coordinate, so slice the query dimension instead
    of rebuilding it. Static per-direction tensors (``cos``/``lognnz``) are shared;
    the message-block precompute is query-tiled so it is rebuilt for the subset."""

    __slots__ = _StageGeom.__slots__

    def __init__(self, base, take, torch, device):
        take = np.asarray(take, bool)
        idx_np = np.flatnonzero(take).astype(np.int64)
        idx = torch.from_numpy(idx_np).to(device)
        for name in ("ip", "in_", "dp", "dn", "vp", "vn"):
            setattr(self, name, getattr(base, name).index_select(1, idx))
        self.query_idx = base.query_idx.index_select(0, idx)
        self.idx_np = base.idx_np[take]
        self.M = int(len(idx_np))
        self.ndim = base.ndim
        self.cos = base.cos
        self.lognnz = base.lognnz
        self.message_blocks = _build_message_blocks(self, torch)


class _SubGeoms:
    """Per-stage geometry restricted to a subset (``qmask``) of a chunk's
    refinement points, in the halo-padded local frame. Mirrors ``_ChunkGeoms``
    but lays only the subset cells as field rows -- off-subset neighbours fall
    outside the present set and are masked by ``_CompactGeom``, giving the
    "faces / anchors only" context each phase needs for free."""

    __slots__ = (
        "ndim", "padded_shape", "n_padded", "levels", "stride", "block",
        "geoms", "coords", "chain", "interior_flat", "ref_halo_flat",
        "ref_halo_coords", "cache_key",
    )

    def __init__(self, base, qmask, torch, device, cache_key):
        self.ndim = base.ndim
        self.levels, self.stride, self.block = base.levels, base.stride, base.block
        self.padded_shape = base.padded_shape
        self.n_padded = base.n_padded
        self.cache_key = cache_key
        self.geoms, self.coords = [], []
        for base_geom, base_coords in zip(base.geoms, base.coords):
            if base_geom is None:
                self.geoms.append(None)
                self.coords.append(None)
                continue
            take = qmask[tuple(base_coords[:, k] for k in range(self.ndim))]
            if not take.any():
                self.geoms.append(None)
                self.coords.append(None)
                continue
            self.geoms.append(_SubsetStageGeom(base_geom, take, torch, device))
            self.coords.append(base_coords[take])
        self.chain = [0] + [
            s for s in range(1, len(self.geoms)) if self.geoms[s] is not None
        ]
        qc = np.stack(np.nonzero(qmask), axis=1)
        self.interior_flat = (
            np.ravel_multi_index(
                [(qc[:, k] + self.stride) for k in range(self.ndim)],
                self.padded_shape,
            )
            if len(qc)
            else np.zeros(0, np.int64)
        )
        seen = []
        for g in self.geoms:
            if g is None:
                continue
            seen.append(g.ip[g.vp])
            seen.append(g.in_[g.vn])
        ref = (
            torch.unique(torch.cat(seen)).cpu().numpy()
            if seen
            else np.zeros(0, np.int64)
        )
        self.ref_halo_flat = ref[~np.isin(ref, self.interior_flat)].astype(np.int64)
        self.ref_halo_coords = (
            np.stack(np.unravel_index(self.ref_halo_flat, self.padded_shape), 1)
            - self.stride
        )


_SUB_GEOM_CACHE: dict = {}
_SUB_GEOM_CACHE_MAX = 2


def _build_sub_geoms(cshape, levels, stride, block, qmask, torch, device,
                     agg_level, cache_key):
    key = (tuple(cshape), levels, stride, block, agg_level, str(device), cache_key)
    hit = _SUB_GEOM_CACHE.get(key)
    if hit is None:
        base = build_chunk_geoms(
            cshape, levels, stride, block, torch, device, agg_level
        )
        hit = _SubGeoms(base, qmask, torch, device, key)
        _SUB_GEOM_CACHE[key] = hit
        if len(_SUB_GEOM_CACHE) > _SUB_GEOM_CACHE_MAX:
            _SUB_GEOM_CACHE.pop(next(iter(_SUB_GEOM_CACHE)))
    return hit


# --- per-chunk sub-frame + predictor driving --------------------------------
def _sub_frame(predictor, sub, origin, known_pred):
    """Compact frame for a sub-chunk: subset cells first (field rows), then the
    referenced halo band filtered to cells ``known_pred`` marks as already
    reconstructed. Returns (geoms, n_interior, n_compact, halo_rows, h_gflat)."""
    torch = predictor._torch
    ndim = len(predictor.shape)
    gc = sub.ref_halo_coords + np.asarray(origin, np.int64)
    shp = np.asarray(predictor.shape)
    inb = np.all((gc >= 0) & (gc < shp), axis=1)
    gci = gc[inb]
    keep = known_pred(gci) if len(gci) else np.zeros(0, bool)
    halo_present = sub.ref_halo_flat[inb][keep]
    gflat = (
        np.ravel_multi_index([gci[keep][:, k] for k in range(ndim)], predictor.shape)
        if keep.any()
        else np.zeros(0, np.int64)
    )
    n_interior = int(len(sub.interior_flat))
    present = np.concatenate([sub.interior_flat, halo_present])
    n_compact = 1 + len(present)
    remap = _build_remap(present, sub.n_padded, torch, predictor.device)
    geoms = [None if g is None else _CompactGeom(g, remap, torch) for g in sub.geoms]
    halo_rows = slice(n_interior + 1, n_compact)
    return geoms, n_interior, n_compact, halo_rows, gflat


def _start_sub(predictor, ci, recon, sub, known_pred):
    """Set the predictor's per-chunk state for a subset chain, reusing
    ``predict_stage``. Halo context (anchors / decoded faces) is embedded from its
    reconstructed value with a null pooled context (``anchor_finalize``)."""
    torch = predictor._torch
    sls = predictor.chunk_slices(ci)
    origin = np.array([sl.start for sl in sls], np.int64)
    geoms, n_interior, n_compact, halo_rows, h_gflat = _sub_frame(
        predictor, sub, origin, known_pred
    )
    E = torch.zeros(
        predictor.C, n_compact, sub.ndim, predictor.d, device=predictor.device
    )
    if len(h_gflat):
        vals = predictor._norm(recon.reshape(predictor.C, -1)[:, h_gflat])
        with torch.no_grad():
            E[:, halo_rows] = anchor_finalize(predictor.model, vals, sub.ndim).to(
                E.dtype
            )
    predictor._cg = sub
    predictor._ci = ci
    predictor._E = E
    predictor._geoms = geoms
    predictor._gidx = [
        None
        if c is None
        else np.ravel_multi_index(
            [(c[:, k] + origin[k]) for k in range(len(predictor.shape))],
            predictor.shape,
        )
        for c in sub.coords
    ]
    predictor._ctx = None
    predictor._pos = 0


def _finish_sub(predictor):
    predictor._E = predictor._ctx = predictor._cg = None


# --- two-phase driver (shared by encode and decode) -------------------------
def _edge_sched_passes(values, recon, ebs, radius, round_output, predictor,
                       shape, stride, block, *, decode=False, payload=None,
                       off0=0):
    """Phase 1 (faces) then phase 2 (strict interiors), each a full pass over the
    chunks. Encodes (returns packed parts) or decodes (returns the new offset)."""
    _SUB_GEOM_CACHE.clear()
    edges = predictor.edges
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    c = 1 if decode else values.shape[0]
    parts: list[bytes] = []
    off = off0

    def known_face(gci):  # phase 1: anchors only are pre-known
        return _offgrid_global(gci, stride) == 0

    def known_interior(gci):  # phase 2: anchors + every decoded face
        return (_offgrid_global(gci, stride) == 0) | _on_internal_boundary(gci, edges)

    for phase, known_pred in (("faces", known_face),
                              ("interiors", known_interior)):
        bar = _progress_bar(
            f"edge {phase} {'decode' if decode else 'encode'}",
            predictor.n_chunks,
            unit="chunk",
        )
        for ci in range(predictor.n_chunks):
            sls = predictor.chunk_slices(ci)
            origin = np.array([sl.start for sl in sls], np.int64)
            cshape = tuple(sl.stop - sl.start for sl in sls)
            face, strict = _classify(cshape, origin, edges, stride)
            qmask = face if phase == "faces" else strict
            sub = _build_sub_geoms(
                cshape, predictor.levels, stride, block, qmask,
                predictor._torch, predictor.device, predictor.agg_level,
                (phase, cshape, tuple(int(o) for o in origin % np.asarray(edges))),
            )
            if len(sub.chain) <= 1:  # no non-empty refinement stage in this subset
                bar.update(1)
                continue
            _start_sub(predictor, ci, recon, sub, known_pred)
            for jj in range(1, len(sub.chain)):
                s = sub.chain[jj]
                pred, sc = predictor.predict_stage(s, recon, ebs[s])
                Q = sub.coords[s]
                qidx = (slice(None), *(Q[:, k] for k in range(len(cshape))))
                M = len(Q)
                levels64 = scale_to_level(sc, ebs[s]).reshape(-1)
                block_slc = (slice(None), *sls)
                if decode:
                    codes, outliers, off = unpack_stage(
                        payload, off, rans_levels=levels64, rans_tables=stage_tables[s]
                    )
                    recon[block_slc][qidx] = dequantize(
                        pred, codes, outliers, ebs[s], radius
                    ).reshape(c, M)
                else:
                    cvals = values[block_slc][qidx]
                    codes, outliers = quantize(
                        cvals, pred, ebs[s], radius, round_output=round_output
                    )
                    recon[block_slc][qidx] = dequantize(
                        pred, codes, outliers, ebs[s], radius
                    ).reshape(c, M)
                    parts.append(
                        pack_stage(
                            codes, outliers, rans_levels=levels64,
                            rans_tables=stage_tables[s],
                        )
                    )
            _finish_sub(predictor)
            bar.update(1)
        bar.close()
    return off if decode else parts


def compress_chunked_edges(values, ebs, radius, round_output, predictor, edges):
    """Encode: global anchors, then the two-phase edges-first schedule."""
    c = values.shape[0]
    shape = values.shape[1:]
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros_like(values)
    anch = _anchor_axes(shape, stride, block)
    parts = [_code_anchor_stage(values, recon, anch, ebs[0], radius, round_output)]
    _log(f"edge-sched encode: shape={shape} edges={edges} anchors done")
    predictor.begin(shape, edges, channels=c)
    parts += _edge_sched_passes(
        values, recon, ebs, radius, round_output, predictor, shape, stride, block
    )
    return b"".join(parts)


def decompress_chunked_edges(payload, shape, ebs, radius, predictor, edges):
    """Decoder twin of ``compress_chunked_edges`` (identical phase/chunk/stage
    order)."""
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros((1, *shape), np.float32)
    anch = _anchor_axes(shape, stride, block)
    off = _decode_anchor_stage(payload, 0, recon, anch, ebs[0], radius)
    predictor.begin(shape, edges, channels=1)
    off = _edge_sched_passes(
        None, recon, ebs, radius, None, predictor, shape, stride, block,
        decode=True, payload=payload, off0=off,
    )
    if off != len(payload):
        raise ValueError("trailing bytes in edge-sched payload")
    return recon[0]


# ===========================================================================
# Device path: two-phase edges-first WITH the scale-gated cubic fallback.
#
# The gate lives only in the device inner loop (gnn_codec._compress_chunked). On
# sharp fields the cubic fallback beats the GNN, but at a chunk face its stencil
# is one-sided (the far side is undecoded) so the gate can't fire and falls back
# to the weak GNN -- that is the dominant seam. Here faces are decoded first, so
# a phase-2 interior's interp region is EXTENDED to include the decoded bounding
# face planes and the cubic/linear fallback becomes both-sided across every seam.
# ===========================================================================
_SUB_FRAME_CACHE: dict = {}
_SUB_FRAME_CACHE_MAX = 4


def _sub_frame_rel(predictor, sub, origin, known_pred):
    """Compact frame for a sub-chunk (subset cells as field rows + the known halo
    band), returning the halo cells' coords *relative to origin* so one layout
    serves every chunk of the same boundary signature. Cached."""
    torch = predictor._torch
    ndim = len(predictor.shape)
    cidx = tuple(int(origin[k] // predictor.edges[k]) for k in range(ndim))
    bsig = tuple((i == 0, i == predictor.grid[k] - 1) for k, i in enumerate(cidx))
    key = (sub.cache_key, predictor.shape, predictor.edges, bsig)
    hit = _SUB_FRAME_CACHE.get(key)
    if hit is not None:
        return hit
    gc = sub.ref_halo_coords + np.asarray(origin, np.int64)
    shp = np.asarray(predictor.shape)
    inb = np.all((gc >= 0) & (gc < shp), axis=1)
    gci = gc[inb]
    keep = known_pred(gci) if len(gci) else np.zeros(0, bool)
    halo_present = sub.ref_halo_flat[inb][keep]
    h_rel = (
        gci[keep] - np.asarray(origin, np.int64)
        if keep.any()
        else np.zeros((0, ndim), np.int64)
    )
    n_interior = int(len(sub.interior_flat))
    present = np.concatenate([sub.interior_flat, halo_present])
    n_compact = 1 + len(present)
    remap = _build_remap(present, sub.n_padded, torch, predictor.device)
    geoms = [None if g is None else _CompactGeom(g, remap, torch) for g in sub.geoms]
    hit = (geoms, n_interior, n_compact, slice(n_interior + 1, n_compact), h_rel)
    _SUB_FRAME_CACHE[key] = hit
    if len(_SUB_FRAME_CACHE) > _SUB_FRAME_CACHE_MAX:
        _SUB_FRAME_CACHE.pop(next(iter(_SUB_FRAME_CACHE)))
    return hit


def _start_sub_wave_dev(predictor, ids, recon_t, sub, known_pred):
    """Device twin of ``ChunkedGNNPredictor.start_wave`` for a subset sub-chain.
    Halo context (anchors / decoded faces) is embedded from its reconstructed
    value with a null pooled context (``anchor_finalize``); no coarse table."""
    torch = predictor._torch
    ndim = len(predictor.shape)
    B = len(ids)
    origins = np.array(
        [[sl.start for sl in predictor.chunk_slices(ci)] for ci in ids], np.int64
    )
    geoms, n_interior, n_compact, halo_rows, h_rel = _sub_frame_rel(
        predictor, sub, origins[0], known_pred
    )
    E = torch.zeros(B, n_compact, ndim, predictor.d, device=predictor.device)
    strides = np.cumprod((1,) + predictor.shape[:0:-1])[::-1].astype(np.int64)
    if len(h_rel):
        gflat = (h_rel @ strides)[None, :] + (origins @ strides)[:, None]  # (B, H)
        vals = predictor._norm_t(recon_t.reshape(-1)[torch.from_numpy(gflat).to(predictor.device)])
        with torch.no_grad(), predictor._amp():
            E[:, halo_rows] = anchor_finalize(predictor.model, vals, ndim).to(E.dtype)
    obase = origins @ strides
    predictor._wave_gidx = [
        None
        if coords is None
        else torch.from_numpy((coords @ strides)[None, :] + obase[:, None]).to(
            predictor.device
        )
        for coords in sub.coords
    ]
    predictor._cg = sub
    predictor._wave_ids = list(ids)
    predictor._E = E
    predictor._geoms = geoms
    predictor._ctx = None
    predictor._pos = 0


def _finish_sub_wave_dev(predictor):
    predictor._E = predictor._ctx = predictor._cg = None


def _sub_plan(torch, dev, sub, cshape, full_shape, levels, stride, block):
    """Per-stage device indices for a subset sub-chain: value-block positions
    (gather), full-tensor recon offsets (scatter) and interp coords/stride/axes
    (the gate). Keyed by stage index (aligned with ``stage_plan``)."""
    plan = stage_plan(cshape, levels, stride, block)
    full_strides = np.cumprod((1,) + tuple(full_shape)[:0:-1])[::-1].astype(np.int64)
    cs_strides = np.cumprod((1,) + tuple(cshape)[:0:-1])[::-1].astype(np.int64)
    out: dict = {}
    for s in sub.chain[1:]:
        Q = sub.coords[s]  # (M, ndim) chunk-frame
        pos_block = (Q * cs_strides).sum(1).astype(np.int64)
        recon_off = (Q * full_strides).sum(1).astype(np.int64)
        _mask, st, ax = plan[s]
        out[s] = (
            torch.from_numpy(np.ascontiguousarray(pos_block)).to(dev),
            torch.from_numpy(np.ascontiguousarray(recon_off)).to(dev),
            Q,
            st,
            ax,
        )
    return out


def _ext_slices(origin, cshape, full_shape, stride, extend):
    """Chunk block, optionally extended by ``stride`` on each internal side so the
    gate's interp reaches the decoded bounding face planes. Returns (slices,
    lo_ext per axis)."""
    ndim = len(cshape)
    sls, lo = [], []
    for a in range(ndim):
        lo_a = stride if (extend and origin[a] > 0) else 0
        hi_a = stride if (extend and origin[a] + cshape[a] < full_shape[a]) else 0
        sls.append(slice(origin[a] - lo_a, origin[a] + cshape[a] + hi_a))
        lo.append(lo_a)
    return tuple(sls), lo


def _edge_dev_passes(values, recon_t, ebs, radius, round_output, predictor,
                     shape, stride, block, *, gate=True, decode=False,
                     payload=None, off0=0, gates=None):
    """Device two-phase driver. Phase 1 (faces) gates a chunk-local both-sided
    in-plane interp; phase 2 (interiors) gates an interp extended to the decoded
    bounding faces. With ``gate=False`` the classical fallback is skipped and the
    pure GNN prediction/scale is coded (isolates the prediction seam). Encodes
    (returns (parts, gate_bytes)) or decodes (returns the new offset)."""
    torch = predictor._torch
    dev = predictor.device
    edges = predictor.edges
    c = predictor.C
    center = default_interp_center(len(shape))
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    full_strides = np.cumprod((1,) + shape[:0:-1])[::-1].astype(np.int64)
    recon_flat = recon_t.reshape(c, -1)
    parts: list[bytes] = []
    gate_out: list = []
    off = off0
    gi = 0
    _SUB_GEOM_CACHE.clear()
    _SUB_FRAME_CACHE.clear()

    def known_face(gci):
        return _offgrid_global(gci, stride) == 0

    def known_interior(gci):
        return (_offgrid_global(gci, stride) == 0) | _on_internal_boundary(gci, edges)

    phases = (("faces", known_face, False), ("interiors", known_interior, True))
    for phase, known_pred, extend in phases:
        qsel = phase
        bar = _progress_bar(
            f"edge {phase} {'decode' if decode else 'encode'}",
            predictor.n_chunks,
            unit="chunk",
        )
        plan_cache: dict = {}
        for ci in range(predictor.n_chunks):
            sls = predictor.chunk_slices(ci)
            origin = np.array([sl.start for sl in sls], np.int64)
            cshape = tuple(sl.stop - sl.start for sl in sls)
            face, strict = _classify(cshape, origin, edges, stride)
            qmask = face if qsel == "faces" else strict
            sub = _build_sub_geoms(
                cshape, predictor.levels, stride, block, qmask,
                predictor._torch, dev, predictor.agg_level,
                (phase, cshape, tuple(int(o) for o in origin % np.asarray(edges))),
            )
            if len(sub.chain) <= 1:
                bar.update(1)
                continue
            pkey = (phase, cshape)
            if pkey not in plan_cache:
                plan_cache[pkey] = _sub_plan(
                    torch, dev, sub, cshape, shape, predictor.levels, stride, block
                )
            splan = plan_cache[pkey]
            ext_sls, lo_ext = _ext_slices(origin, cshape, shape, stride, extend)
            origin_base = int((origin * full_strides).sum())
            if not decode:
                vblock = (
                    torch.from_numpy(
                        np.ascontiguousarray(values[(slice(None), *sls)])
                    )
                    .to(dev)
                    .reshape(c, -1)
                )
            _start_sub_wave_dev(predictor, [ci], recon_t, sub, known_pred)
            for s in sub.chain[1:]:
                pos_block_dev, recon_off_dev, Q, st_i, ax_i = splan[s]
                tables = stage_tables[s]
                pred, scale = predictor.predict_wave_stage(s, recon_t, ebs[s])
                p = pred[0][None, :]
                sc = scale[0]
                if gate:
                    # interp over the (possibly extended) block, both-sided
                    coords_t = tuple(
                        torch.from_numpy(
                            np.ascontiguousarray(Q[:, k] + lo_ext[k])
                        ).to(dev)
                        for k in range(len(cshape))
                    )
                    ip_cubic = _interp_stage_pred_t(
                        torch, recon_t, ext_sls, coords_t, st_i, ax_i, center, cubic=True
                    )
                    ip_lin0 = _interp_stage_pred_t(
                        torch, recon_t, ext_sls, coords_t, st_i, ax_i, 1, cubic=False
                    )
                    ip_lin1 = _interp_stage_pred_t(
                        torch, recon_t, ext_sls, coords_t, st_i, ax_i, 2, cubic=False
                    )
                if decode:
                    if gate:
                        g = int(gates[gi]); gi += 1
                        gk = g >> 8
                        ip = ip_lin0 if gk == 2 else (ip_lin1 if gk == 3 else ip_cubic)
                        p, sc = _gate_apply_t(
                            torch, pred[0], sc, ip, ebs[s],
                            torch.tensor((g >> 4) & 15, device=dev),
                            torch.tensor(g & 15, device=dev),
                        )
                    levels64 = scale_to_level(sc.cpu().numpy()[None, :], ebs[s]).reshape(-1)
                    codes, outliers, off = unpack_stage(
                        payload, off, rans_levels=levels64, rans_tables=tables
                    )
                    recon_stage = _dequantize_t(
                        torch, p,
                        torch.from_numpy(codes.astype(np.int64)).to(dev),
                        torch.from_numpy(outliers).to(dev),
                        ebs[s], radius,
                    )
                else:
                    cvals = vblock.index_select(1, pos_block_dev)
                    if gate:
                        gk, gt, gs = _gate_select_t(
                            torch, (cvals - p).abs(),
                            ((cvals - ip_cubic).abs(), (cvals - ip_lin0).abs(),
                             (cvals - ip_lin1).abs()),
                            sc, ebs[s], radius,
                        )
                        gate_out.append(gk * 256 + gt * 16 + gs)
                        ip = torch.where(
                            gk == 2, ip_lin0, torch.where(gk == 3, ip_lin1, ip_cubic)
                        )
                        p, sc = _gate_apply_t(torch, pred[0], sc, ip, ebs[s], gt, gs)
                    codes_t, recon_stage, outliers_t = _quantize_t(
                        torch, cvals, p, ebs[s], radius, round_output
                    )
                    levels64 = scale_to_level(
                        sc.cpu().numpy()[None, :], ebs[s]
                    ).reshape(-1)
                    parts.append(
                        pack_stage(
                            codes_t.cpu().numpy().astype(np.uint32),
                            outliers_t.cpu().numpy().astype(np.float32),
                            rans_levels=levels64, rans_tables=tables,
                        )
                    )
                gpos = recon_off_dev + origin_base
                recon_flat.index_copy_(1, gpos, recon_stage.reshape(c, -1))
            _finish_sub_wave_dev(predictor)
            bar.update(1)
        bar.close()
    if decode:
        return off
    return parts, [int(x) for x in torch.stack(gate_out).cpu().tolist()] if gate_out else []


def compress_chunked_edges_dev(values, ebs, radius, round_output, predictor,
                               edges, gate=True):
    """Encode: global anchors, then the two-phase edges-first schedule on the
    device inner loop. Returns (payload, gate_bytes). ``gate=False`` codes the
    pure GNN (no cubic fallback), isolating the prediction seam."""
    c = values.shape[0]
    shape = values.shape[1:]
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros_like(values)
    anch = _anchor_axes(shape, stride, block)
    parts = [_code_anchor_stage(values, recon, anch, ebs[0], radius, round_output)]
    _log(f"edge-sched(dev) encode: shape={shape} edges={edges} gate={gate} anchors done")
    predictor.begin(shape, edges, channels=c)
    recon_t = predictor._torch.from_numpy(recon).to(predictor.device)
    body, gates = _edge_dev_passes(
        values, recon_t, ebs, radius, round_output, predictor, shape, stride, block,
        gate=gate,
    )
    if predictor.device.type == "cuda":
        predictor._torch.cuda.synchronize(predictor.device)
    return b"".join(parts + body), gates


def decompress_chunked_edges_dev(payload, shape, ebs, radius, predictor, edges,
                                 gates, gate=True):
    """Decoder twin of ``compress_chunked_edges_dev``."""
    torch = predictor._torch
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros((1, *shape), np.float32)
    anch = _anchor_axes(shape, stride, block)
    off = _decode_anchor_stage(payload, 0, recon, anch, ebs[0], radius)
    predictor.begin(shape, edges, channels=1)
    recon_t = torch.from_numpy(recon).to(predictor.device)
    off = _edge_dev_passes(
        None, recon_t, ebs, radius, None, predictor, shape, stride, block,
        gate=gate, decode=True, payload=payload, off0=off, gates=gates,
    )
    if off != len(payload):
        raise ValueError("trailing bytes in edge-sched payload")
    return recon_t[0].cpu().numpy()
