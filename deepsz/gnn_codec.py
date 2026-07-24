"""Checkpoint-backed, tensor-shaped GNN compressor codec."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import zstandard

from .codec import _compress_region
from . import gnn_predictor as _gp
from .gnn_predictor import ChunkedGNNPredictor, GNNPredictor
from .levels import stage_ebs, stage_masks, stage_plan
from .predictor import default_interp_center
from .quantizer import dequantize, quantize
from .bitstream import pack_stage, unpack_stage
from .rans import SCALE_HI_MULT, SCALE_LO_DIV, build_laplace_tables, scale_to_level


_MAGIC = b"DEEPSZGN"
_VERSION = 9
_PREFIX = "<8sII"
_PREFIX_SIZE = struct.calcsize(_PREFIX)
_ANCHOR_BLOCK = 1

# auto mode: whole-tensor below this many points, chunked above (whole-tensor
# memory is ~30*L*K*d bytes/point in transients — ~2^21 points is a few GB)
_AUTO_CHUNK_THRESHOLD = 1 << 21
# Minimum chunk count before an *explicit* compile=True is honored: below this,
# dynamo warmup can't be amortized, so a forced compile is silently skipped.
_COMPILE_MIN_CHUNKS = 64
# Chunk count past which compile="auto" turns compile ON. None = "no crossover
# measured -> auto keeps compile OFF". Set from scripts/bench_compile.py.
#
# Measured (Volta cap 7.0, d64/agg2/levels2/edge8/fp16, cache_size_limit=8 = the
# torch default the codec runs at), compress wall-time vs eager (off):
#     chunks     off   default-compile   reduce-overhead(CUDA graphs)
#         16   7.1 s     81.9 (0.09x)        83.0 (0.09x)
#         64  18.4 s     91.8 (0.20x)       107.7 (0.17x)
#        256  63.5 s    142.4 (0.45x)       190.7 (0.33x)
# compile NEVER wins and the curves diverge: its marginal per-chunk cost
# (0.26 default / 0.43 reduce-overhead s/chunk) exceeds eager's (0.24), so there
# is no finite crossover. Cause: the message-pass ``embed`` recompiles per distinct
# ``n_live`` neighbour-geometry -- capped at 8 by dynamo, so after 8 geometries it
# falls back to eager and the warmup is pure loss; reduce-overhead is worse still
# because varying per-wave M forces a CUDA-graph recapture every wave. This is a
# Python-control-flow guard issue, not GPU-specific, so it is expected to hold on
# H100/A100 too. Re-run bench_compile there and set an int here if a crossover
# appears (e.g. after the n_live recompiles are removed). dynamic=False was tried
# and is WORSE: it specializes per stage geometry and never converges (>420 s at 64
# chunks) instead of amortizing -- dynamic=True's symbolic-M graph is the lesser evil.
_COMPILE_AUTO_CROSSOVER: int | None = None
# Schedule depth per input rank for ``levels="auto"``. The common ranks are set
# explicitly (2-D 9, 3-D 7, 4-D 5 -> anchor_stride 512 / 128 / 32; 4-D keeps the
# edge-32 operating point). Ranks outside the table fall back to the
# constant-points-per-chunk rule below.
_RANK_LEVELS = {2: 8, 3: 6, 4: 5, 5: 4}


def _auto_levels(shape: tuple[int, ...]) -> int:
    """Levels for ``levels="auto"``, chosen from the rank of the input.

    The schedule depth follows the input rank, not its size: a chunk is about one
    to a few anchor cells (``anchor_stride**ndim = 2**(levels*ndim)`` points), so
    keying the stride off the rank keeps an ~constant number of points per chunk.
    ``_RANK_LEVELS`` fixes the tuned common ranks; other ranks fall back to
    ``floor(log2(points_per_chunk) / ndim)`` from the auto-chunk point budget
    (``_AUTO_CHUNK_THRESHOLD``), so a cell stays within budget -- higher-rank
    tensors get a smaller stride, lower-rank a larger one.

    A size guard then keeps ``anchor_stride <= max_axis / 2`` so a small or
    low-rank input still carries at least two anchors on its largest axis rather
    than collapsing to a lone corner anchor."""
    ndim = len(shape)
    if ndim in _RANK_LEVELS:
        rank_levels = _RANK_LEVELS[ndim]
    else:
        budget_log2 = _AUTO_CHUNK_THRESHOLD.bit_length() - 1  # log2 points/chunk
        rank_levels = budget_log2 // ndim
    max_axis = int(max(shape))
    size_cap = max(1, (max_axis.bit_length() - 1) - 1)  # anchor_stride <= max/2
    return max(1, min(rank_levels, size_cap))


# eb_ratio for the GNN codec is a *coarsest-level* factor r_end, not a raw
# per-step decay: the finest level always keeps the full eb and the coarsest
# level lands on eb * r_end. It is depth-normalised into stage_ebs's per-step
# ratio at compress time (see _per_step_eb_ratio), so the coarse/fine spread is
# invariant to the now rank-dependent schedule depth. Without this, a fixed
# per-step ratio compounds as ratio**(levels-1): 0.8 gives 0.8**4=0.41x at 4-D
# (levels 5) but 0.8**8=0.17x at 2-D (levels 9), i.e. wildly different budgets.
# The default keeps the coarsest stage at 90% of the requested bound. The size
# sweep explores that operating point together with flatter and tighter spreads.
_GNN_EB_COARSE_FACTOR = 0.9
_GNN_EB_COARSE_SWEEP = (1.0, 0.9, 0.65, 0.41)


def _per_step_eb_ratio(coarse_factor: float, levels: int) -> float:
    """Depth-normalise a coarsest-level factor into ``stage_ebs``'s per-step ratio.

    ``stage_ebs`` tightens the level at ``depth`` by ``ratio ** depth`` with
    ``depth`` in ``[0, levels-1]``. Returning ``coarse_factor ** (1/(levels-1))``
    makes the coarsest level land on exactly ``eb * coarse_factor`` for any
    schedule depth, so the same knob means the same thing across ranks. At
    ``levels == 1`` there is a single (finest) level, so the factor is meaningless
    and a flat 1.0 is returned."""
    if levels <= 1:
        return 1.0
    return float(coarse_factor) ** (1.0 / (levels - 1))


def _log(msg):
    # ponytail: env-gated so tests/CLI stay quiet; set DEEPSZ_PROGRESS=1 to see it
    if not os.environ.get("DEEPSZ_PROGRESS"):
        return
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _cuda_peak(predictor):
    """Process GPU allocation peak, or ``None`` on CPU.

    Do not reset the global PyTorch counter here: callers such as benchmark
    harnesses reset it at the start of their measured region, and an internal
    reset after every wave would silently corrupt their result.
    """
    torch = getattr(predictor, "_torch", None)
    dev = getattr(predictor, "device", None)
    if torch is None or dev is None or dev.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(dev)


def _progress_bar(tag, n, unit="wave"):
    # env-gated (DEEPSZ_PROGRESS) so tests/CLI stay quiet; disabled bar is a no-op.
    from tqdm import tqdm

    return tqdm(
        total=n,
        desc=tag,
        unit=unit,
        file=sys.stderr,
        disable=not os.environ.get("DEEPSZ_PROGRESS"),
    )


def _geometry_stages(ndim: int, levels: int) -> int:
    """Number of masks/geometries in the dimension-generic stage schedule."""
    return 1 + levels * ((1 << ndim) - 1)


def _as_numpy(x: Any) -> np.ndarray:
    """Accept numpy arrays and torch tensors without importing torch eagerly."""
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    cpu = getattr(x, "cpu", None)
    numpy = getattr(x, "numpy", None)
    if detach is not None and cpu is not None:
        return x.detach().cpu().numpy()
    if numpy is not None:
        return x.numpy()
    return np.asarray(x)


def _restore_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        values = np.clip(np.rint(values), info.min, info.max)
    elif dtype.kind == "b":
        values = values >= 0.5
    return values.astype(dtype, copy=False)


def _write_stream(meta: dict[str, Any], payload: bytes, zstd_level: int) -> bytes:
    header = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = zstandard.ZstdCompressor(level=zstd_level).compress(payload)
    return struct.pack(_PREFIX, _MAGIC, _VERSION, len(header)) + header + body


def _read_stream(stream: bytes) -> tuple[dict[str, Any], bytes]:
    if len(stream) < _PREFIX_SIZE:
        raise ValueError("not a DeepSZ GNN stream")
    magic, version, header_len = struct.unpack_from(_PREFIX, stream, 0)
    if magic != _MAGIC:
        raise ValueError(f"not a DeepSZ GNN stream (bad magic {magic!r})")
    if version != _VERSION:
        raise ValueError(f"unsupported DeepSZ GNN stream version {version}")
    off = _PREFIX_SIZE
    meta = json.loads(stream[off : off + header_len].decode("utf-8"))
    payload = zstandard.ZstdDecompressor().decompress(stream[off + header_len :])
    return meta, payload


def _empty_stats(n_stages: int) -> dict[str, Any]:
    return {
        "predict_s": 0.0,
        "quantize_s": 0.0,
        "entropy_s": 0.0,
        "outliers": 0,
        "stage_codes": [0] * n_stages,
        "stage_outliers": [0] * n_stages,
        "stage_payload_bytes": [0] * n_stages,
        "stage_model_bits": [0.0] * n_stages,
        "stage_pred_sae": [0.0] * n_stages,
        "stage_pred_sse": [0.0] * n_stages,
        "stage_recon_sae": [0.0] * n_stages,
        "stage_recon_sse": [0.0] * n_stages,
        "stage_recon_max": [0.0] * n_stages,
    }


def _decompress_region(
    payload: bytes,
    shape: tuple[int, ...],
    masks: list[np.ndarray],
    ebs: list[float],
    radius: int,
    predictor: GNNPredictor,
    use_rans: bool,
) -> np.ndarray:
    recon = np.zeros((1, *shape), np.float32)
    known = np.zeros(shape, bool)
    off = 0
    for stage_idx, pos in enumerate(masks):
        n = int(pos.sum())
        if n == 0:
            if use_rans:
                tables = build_laplace_tables(ebs[stage_idx], radius)
                codes, outliers, off = unpack_stage(
                    payload, off, rans_levels=np.zeros(0, np.uint8), rans_tables=tables
                )
            else:
                codes, outliers, off = unpack_stage(payload, off)
            continue
        if stage_idx == 0:
            pred = np.zeros((1, n), np.float32)
            scale = np.full((1, n), ebs[stage_idx], np.float32)
        else:
            if use_rans:
                pred, scale = predictor.predict(recon, known, pos, eb=ebs[stage_idx])
            else:
                got = predictor.predict(recon, known, pos, eb=ebs[stage_idx])
                pred = got[0] if isinstance(got, tuple) else got
                scale = None
        if use_rans:
            tables = build_laplace_tables(ebs[stage_idx], radius)
            levels64 = scale_to_level(scale, ebs[stage_idx]).reshape(-1)
            codes, outliers, off = unpack_stage(
                payload, off, rans_levels=levels64, rans_tables=tables
            )
        else:
            codes, outliers, off = unpack_stage(payload, off)
        recon[:, pos] = dequantize(
            pred, codes, outliers, ebs[stage_idx], radius
        ).reshape(1, n)
        known |= pos
    if off != len(payload):
        raise ValueError("trailing bytes in DeepSZ GNN payload")
    return recon[0]


def _chunk_stage_ebs(shape, levels, stride, block, eb, eb_ratio) -> list[float]:
    """Per-stage error bounds for the chunked path. The stage strides depend
    only on (rank, levels, stride), so evaluate ``stage_ebs`` on a tiny
    same-rank shape — never on the full tensor (that would materialise
    full-shape stage masks, the memory bug this path removes)."""
    return stage_ebs((2 * stride,) * len(shape), levels, stride, block, eb, eb_ratio)


def _anchor_axes(shape: tuple[int, ...], stride: int, block: int) -> list[np.ndarray]:
    """Per-axis anchor coordinates. The anchor set is separable (every
    coordinate has residue < block mod stride), so the global anchor pass can
    index it with np.ix_ and never materialise a full-shape mask."""
    axes = []
    for n in shape:
        c = np.arange(n)
        axes.append(c[(c % stride) < block])
    return axes


def _auto_chunk_edges(shape: tuple[int, ...], stride: int) -> tuple[int, ...]:
    """Largest near-isotropic chunk below the automatic point budget.

    Short axes are included in full so elongated tensors do not waste most of
    the budget.  Edges are multiples of the anchor stride as required by the
    chunk geometry.
    """
    target = _AUTO_CHUNK_THRESHOLD
    fixed = 1
    remaining = len(shape)
    free_edge = float(target) ** (1.0 / remaining)
    full_axes: set[int] = set()
    for axis in sorted(range(len(shape)), key=shape.__getitem__):
        free_edge = (target / fixed) ** (1.0 / remaining)
        if shape[axis] > free_edge:
            break
        full_axes.add(axis)
        fixed *= shape[axis]
        remaining -= 1
        if not remaining:
            break
    if remaining:
        free_edge = (target / fixed) ** (1.0 / remaining)
    return tuple(
        max(stride, -(-n // stride) * stride)
        if axis in full_axes
        else max(stride, int(free_edge) // stride * stride)
        for axis, n in enumerate(shape)
    )


def _code_anchor_stage(values, recon, axes, eb0, radius, round_output):
    """Encoder side of the global anchor pass: quantize anchors against pred 0,
    write their recon, return the packed stage."""
    c = values.shape[0]
    sub = (slice(None), *np.ix_(*axes))
    avals = values[sub].reshape(c, -1)
    n = avals.shape[1]
    pred = np.zeros((c, n), np.float32)
    codes, outliers = quantize(avals, pred, eb0, radius, round_output=round_output)
    recon[sub] = dequantize(pred, codes, outliers, eb0, radius).reshape(
        recon[sub].shape
    )
    tables = build_laplace_tables(eb0, radius)
    levels64 = scale_to_level(np.full((c, n), eb0, np.float32), eb0).reshape(-1)
    return pack_stage(codes, outliers, rans_levels=levels64, rans_tables=tables)


def _decode_anchor_stage(payload, off, recon, axes, eb0, radius):
    c = recon.shape[0]
    sub = (slice(None), *np.ix_(*axes))
    n = int(np.prod([len(a) for a in axes]))
    tables = build_laplace_tables(eb0, radius)
    levels64 = scale_to_level(np.full((c, n), eb0, np.float32), eb0).reshape(-1)
    codes, outliers, off = unpack_stage(
        payload, off, rans_levels=levels64, rans_tables=tables
    )
    pred = np.zeros((c, n), np.float32)
    recon[sub] = dequantize(pred, codes, outliers, eb0, radius).reshape(
        recon[sub].shape
    )
    return off


def _chunk_waves(grid: tuple[int, ...]) -> list[list[int]]:
    """Group chunk ids into color waves. A wave = chunks with the same per-axis
    parity ("color") and the same tensor-boundary signature. Same-color chunks
    are >=2 apart on every axis they differ, so their halos (thickness one chunk)
    never overlap. Ordered by color so a wave's cross-color neighbours in earlier
    colors are already coded. Correctness (the error bound) holds for any order;
    only which context is available, hence the ratio, shifts."""
    groups: dict = {}
    for ci in range(int(np.prod(grid))):
        cidx = np.unravel_index(ci, grid)
        color = tuple(int(i) % 2 for i in cidx)
        bsig = tuple((int(i) == 0, int(i) == g - 1) for i, g in zip(cidx, grid))
        groups.setdefault((color, bsig), []).append(ci)

    def rank(key):
        color, bsig = key
        return (sum(b << k for k, b in enumerate(color)), bsig)

    return [groups[k] for k in sorted(groups, key=rank)]


# Scale-gated classical-predictor fallback: where the model's predicted scale b
# sits below eb*2^T, its confidence is either genuine (high eb: the point is
# ~free) or the learned-predictor precision floor talking (low eb) — only
# measuring can tell. The encoder sweeps (predictor, T, shift) per chunk-stage
# against the true residuals. The candidates are the original cubic/axis-average
# interpolation and SZ3's strong 3-D CLOUD choice, linear interpolation along
# the first active axis. The winner travels in the header, so the gate
# self-disables (T=0) wherever neither classical predictor pays.
_GATE_T = np.array([2, 3, 4, 5, 6, 7, 8], np.float64)
_GATE_SHIFTS = (0, 2, 4, 6)


# --- device inner-loop primitives -----------------------------------------
# The chunked encode/decode inner loop runs entirely on the GPU: recon stays
# resident on the device and quantize / dequantize / gate all execute in torch,
# so the per-stage critical path never syncs to host. Only the rANS pack/unpack
# (constriction, host-only) crosses over -- and it does not feed recon, so on
# encode it is deferred off the critical path. Encoder and decoder call the
# identical functions, so their reconstructions match bit for bit even where a
# torch transcendental (exp2/log2) differs from numpy by a ULP; correctness is
# anchored on enc/dec agreement, not on matching the old numpy stream.


def _interp_axis_at_t(torch, W, coords, axis, s, shape, cubic):
    """Linear or cubic interpolation on device.

    ``W`` is ``(C, *S)`` float64 and ``coords`` is a tuple of ``(M,)`` long
    tensors.
    """

    def gather(off):
        ca = coords[axis] + off
        valid = (ca >= 0) & (ca < shape[axis])
        idx = list(coords)
        idx[axis] = ca.clamp(0, shape[axis] - 1)
        return W[(slice(None), *idx)], valid

    Lm1, vm1 = gather(-s)
    Lp1, vp1 = gather(+s)
    pred = 0.5 * (Lm1 + Lp1)
    if cubic:
        Lm3, vm3 = gather(-3 * s)
        Lp3, vp3 = gather(+3 * s)
        cub = (-Lm3 + 9 * Lm1 + 9 * Lp1 - Lp3) / 16.0
        pred = torch.where((vm3 & vp3).unsqueeze(0), cub, pred)
    both = (vm1 & vp1).unsqueeze(0)
    only_left = (vm1 & ~vp1).unsqueeze(0)
    return torch.where(both, pred, torch.where(only_left, Lm1, Lp1))


def _interp_stage_pred_t(
    torch, recon_t, sls, coords_t, stride, axes, center, *, cubic
):
    """Chunk-local interpolation from the causal device reconstruction."""
    W = recon_t[(slice(None), *sls)].double()
    shape = tuple(W.shape[1:])
    if center == 0 or len(axes) == 1:
        ip = sum(
            _interp_axis_at_t(torch, W, coords_t, a, stride, shape, cubic)
            for a in axes
        ) / len(axes)
    else:
        ax = axes[0] if center == 1 else axes[-1]
        ip = _interp_axis_at_t(torch, W, coords_t, ax, stride, shape, cubic)
    return ip.to(torch.float32)  # (C, n)


def _laplace_bits_t(torch, absr, b, eb):
    """Ideal discretized-Laplace model cost used to rank gate settings."""
    b = b.double().clamp(eb / SCALE_LO_DIV, eb * SCALE_HI_MULT)
    k = torch.round(absr.double().abs() / (2 * eb))
    p = torch.where(
        k == 0,
        -torch.expm1(-eb / b),
        0.5 * torch.exp(-((2 * k - 1) * eb / b)) * -torch.expm1(-2 * eb / b),
    )
    return (-torch.log2(p)).clamp_max(32.0)


def _gate_select_t(torch, r_g, r_is, b, eb):
    """Choose the best (classical predictor, T, shift) for one chunk-stage.

    Returns device scalar ``(kind, T, shift)`` tensors, with kind 0 meaning off
    and positive kinds indexing ``r_is`` from one. The selection is branch-free
    so it issues no host sync.
    """
    dev = b.device
    GTf = torch.tensor(_GATE_T, dtype=torch.float64, device=dev)
    GTi = GTf.to(torch.int64)
    SHi = torch.tensor(_GATE_SHIFTS, dtype=torch.int64, device=dev)
    nb = _GATE_T.size + 1
    bucket = torch.bucketize(b.double(), eb * torch.exp2(GTf))
    # one-hot bucket membership: per-bucket sums via a plain (deterministic)
    # reduction, not bincount/scatter_add -- those use atomic float adds on CUDA
    # and would make the gate choice, and thus the stream, non-deterministic.
    onehot = (bucket.unsqueeze(-1) == torch.arange(nb, device=dev)).double()  # (n, nb)

    def cum(r, bb):
        w = _laplace_bits_t(torch, r, bb, eb).sum(0)  # (n,)
        binc = (w.unsqueeze(-1) * onehot).sum(0)  # (nb,)
        return torch.cumsum(binc, 0)

    base = cum(r_g, b)
    tot = base[-1]
    rows = [
        tot - base[:-1] + cum(r_i, b * 2.0**-sh)[:-1]
        for r_i in r_is
        for sh in _GATE_SHIFTS
    ]
    costs = torch.stack(rows)  # (n_predictor * n_shift, nb-1)
    mv, fi = costs.reshape(-1).min(0)
    fired = mv < tot
    ncol = nb - 1
    nsh = len(_GATE_SHIFTS)
    row = fi // ncol
    z = torch.zeros((), dtype=torch.int64, device=dev)
    gate_kind = torch.where(fired, row // nsh + 1, z)
    gate_t = torch.where(fired, GTi[fi % ncol], z)
    gate_sh = torch.where(fired, SHi[row % nsh], z)
    return gate_kind, gate_t, gate_sh


def _gate_apply_t(torch, pred_bi, scale_bi, ip, eb, gate_t, gate_sh):
    """Device twin of ``_gate_apply``, called unconditionally on both sides: a
    ``gate_t`` of 0 is a no-op (mirrors numpy's ``if gate_t`` skip), so encoder
    and decoder stay symmetric without a host-sync branch. (pred, coded scale)."""
    active = gate_t > 0
    m = active & (scale_bi < eb * torch.exp2(gate_t.double()))
    p = torch.where(m.unsqueeze(0), ip, pred_bi.unsqueeze(0))
    sc = torch.where(
        m, scale_bi * torch.exp2(-gate_sh.double()).to(scale_bi.dtype), scale_bi
    )
    return p.to(torch.float32), sc.to(torch.float32)


def _quantize_t(torch, x, pred, eb, radius, round_output):
    """Device twin of ``quantize``+``dequantize`` for one chunk-stage. Returns
    (codes int64, recon float32 with outliers substituted, outliers float32 in
    scan order). ``recon`` is exactly the numpy encoder's committed
    reconstruction and ``codes`` are bit-identical to the numpy quantizer
    (verified), so the host decoder reproduces this recon exactly.

    ``round_output`` matches ``quantizer.quantize``: ``True`` verifies against
    the rounded-to-integer reconstruction, a ``(span, offset)`` pair verifies
    against ``round(recon * span + offset)`` converted back to ``x``'s
    (normalized) units, ``False``/falsy skips the check."""
    x = x.reshape(-1).to(torch.float32)
    pred = pred.reshape(-1).to(torch.float32)
    w = 2.0 * eb
    q = torch.round((x.double() - pred.double()) / w).to(torch.int64)
    in_range = q.abs() < radius
    z = torch.zeros_like(q)
    codes = torch.where(in_range, q + radius, z)
    recon = (pred.double() + w * (codes - radius).double()).to(torch.float32)
    if round_output is True:
        recon_chk = torch.round(recon)
    elif round_output:
        span, offset = round_output
        recon_chk = (
            (torch.round(recon.double() * span + offset) - offset) / span
        ).to(torch.float32)
    else:
        recon_chk = recon
    ok = in_range & ((x - recon_chk).abs() <= float(np.float32(eb)))
    codes = torch.where(ok, codes, z)
    is_out = codes == 0
    # reconstruct from the final codes (== dequantize), outliers exact
    recon = (pred.double() + w * (codes - radius).double()).to(torch.float32)
    recon = torch.where(is_out, x, recon)
    return codes, recon, x[is_out]


def _dequantize_t(torch, pred, codes, outliers, eb, radius):
    """Device twin of ``dequantize``: reconstruct float32 from codes + exact
    outliers, all on the GPU. Matches ``_quantize_t``'s recon bit for bit."""
    pred = pred.reshape(-1).to(torch.float32)
    recon = (pred.double() + (2.0 * eb) * (codes - radius).double()).to(torch.float32)
    is_out = codes == 0
    return recon.masked_scatter(is_out, outliers)


def _compress_chunked(
    values: np.ndarray,
    ebs: list[float],
    radius: int,
    round_output: bool,
    predictor: ChunkedGNNPredictor,
    edges: tuple[int, ...],
    gate: bool = False,
) -> tuple[bytes, list[int] | None]:
    """Encode a global anchor pass followed by one chunk at a time.

    Color-wave ordering (see ``_chunk_waves``) determines which neighbouring
    chunks are available as causal context, but model memory is devoted to one
    maximally sized chunk instead of batching several chunks together.

    Device-resident inner loop: after the (host) anchor pass, the reconstruction
    lives on the GPU and every per-stage step -- forward, pred/scale, cubic-interp
    gate, quantize/dequantize, recon scatter -- runs in torch, so the critical
    path (recon feeding the next forward) never syncs to host. The only host-side
    work is the rANS pack, which does not feed recon; it is deferred to the end of
    each wave (codes streamed off the GPU during the wave), keeping it off the
    per-stage path."""
    torch = predictor._torch
    dev = predictor.device
    c = values.shape[0]
    shape = values.shape[1:]
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros_like(values)
    axes = _anchor_axes(shape, stride, block)
    _log(
        f"encode: shape={shape} edges={edges} device={predictor.device} "
        f"coding anchors..."
    )
    anchor_bar = _progress_bar("encode anchors", 1, unit="stage")
    parts = [_code_anchor_stage(values, recon, axes, ebs[0], radius, round_output)]
    anchor_bar.update(1)
    anchor_bar.close()
    geom_bar = _progress_bar(
        "encode geometry", _geometry_stages(len(shape), predictor.levels), unit="stage"
    )
    predictor.begin(shape, edges, channels=c, geometry_progress=geom_bar.update)
    geom_bar.close()
    coarse_bar = _progress_bar(
        "encode anchor embeddings", predictor.n_chunks, unit="chunk"
    )
    predictor.anchor_coarse(recon, progress=coarse_bar.update)
    coarse_bar.close()
    # hand recon to the GPU for the wave loop; it never returns to host on encode
    recon_t = torch.from_numpy(recon).to(dev)
    recon_flat = recon_t.reshape(c, -1)
    waves = _chunk_waves(predictor.grid)
    _log(
        f"encode: anchors done, {predictor.n_chunks} chunks, "
        f"{predictor.n_chunks} model passes"
    )
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    index_cache: dict = {}  # cshape -> device stage schedule (see _chunk_device_plan)
    full_strides = np.cumprod((1,) + shape[:0:-1])[::-1].astype(np.int64)
    gates_t: list = [] if gate else None  # per stage-chunk gate byte, device scalars
    bar = _progress_bar("encode", predictor.n_chunks)
    for group in waves:
        for ci in group:
            ids = [ci]
            cshape = tuple(sl.stop - sl.start for sl in predictor.chunk_slices(ids[0]))
            if cshape not in index_cache:
                index_cache[cshape] = _chunk_device_plan(
                    torch, dev, cshape, shape, predictor.levels, stride, block
                )
            counts, pos_dev, recon_off_dev, interp_dev, center = index_cache[cshape]
            # The chunk value block is uploaded once, not once per stage.
            vblocks = [
                torch.from_numpy(
                    np.ascontiguousarray(
                        values[(slice(None), *predictor.chunk_slices(ci))]
                    )
                )
                .to(dev)
                .reshape(c, -1)
                for ci in ids
            ]
            origin_bases = [
                int(
                    sum(
                        sl.start * st
                        for sl, st in zip(predictor.chunk_slices(ci), full_strides)
                    )
                )
                for ci in ids
            ]
            predictor.start_wave(ids, recon_t)
            wave_pending: list = []  # (codes, outliers, sc, tables, eb) or None marker
            for s in range(1, len(counts)):
                n = counts[s]
                tables = stage_tables[s]
                if n == 0:
                    wave_pending.extend([(None, tables)] * len(ids))
                    continue
                pos = pos_dev[s]
                pred, scale = predictor.predict_wave_stage(s, recon_t, ebs[s])
                for bi in range(len(ids)):
                    sls = predictor.chunk_slices(ids[bi])
                    cvals = vblocks[bi].index_select(1, pos)  # (C, n)
                    p = pred[bi][None, :]
                    sc = scale[bi]
                    if gate:
                        coords_t, st_i, ax_i = interp_dev[s]
                        ip_cubic = _interp_stage_pred_t(
                            torch,
                            recon_t,
                            sls,
                            coords_t,
                            st_i,
                            ax_i,
                            center,
                            cubic=True,
                        )
                        ip_linear = _interp_stage_pred_t(
                            torch,
                            recon_t,
                            sls,
                            coords_t,
                            st_i,
                            ax_i,
                            1,
                            cubic=False,
                        )
                        gk, gt, gs = _gate_select_t(
                            torch,
                            (cvals - p).abs(),
                            ((cvals - ip_cubic).abs(), (cvals - ip_linear).abs()),
                            sc,
                            ebs[s],
                        )
                        gates_t.append(gk * 256 + gt * 16 + gs)
                        ip = torch.where(gk == 2, ip_linear, ip_cubic)
                        p, sc = _gate_apply_t(torch, pred[bi], sc, ip, ebs[s], gt, gs)
                    codes, recon_stage, outliers = _quantize_t(
                        torch, cvals, p, ebs[s], radius, round_output
                    )
                    gpos = recon_off_dev[s] + origin_bases[bi]
                    recon_flat.index_copy_(1, gpos, recon_stage.reshape(c, -1))
                    wave_pending.append(
                        (
                            codes.to("cpu", non_blocking=True),
                            outliers.to("cpu", non_blocking=True),
                            sc.to("cpu", non_blocking=True),
                            tables,
                            ebs[s],
                        )
                    )
            predictor.finish_wave(recon_t)
            # deferred rANS: the codes streamed off the GPU during the wave; sync
            # once, then pack in stream order (host-only, off the recon path).
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            for item in wave_pending:
                if len(item) == 2:  # empty stage
                    parts.append(
                        pack_stage(
                            np.zeros(0, np.uint32),
                            np.zeros(0, np.float32),
                            rans_levels=np.zeros(0, np.uint8),
                            rans_tables=item[1],
                        )
                    )
                    continue
                codes_c, out_c, sc_c, tables, eb_s = item
                levels = scale_to_level(sc_c.numpy()[None, :], eb_s).reshape(-1)
                parts.append(
                    pack_stage(
                        codes_c.numpy().astype(np.uint32),
                        out_c.numpy().astype(np.float32),
                        rans_levels=levels,
                        rans_tables=tables,
                    )
                )
            peak = _cuda_peak(predictor)
            if peak:
                bar.set_postfix_str(f"peak {peak / 1e9:.2f}GB")
            bar.update(1)
    bar.close()
    gates = None
    if gate:
        gates = [int(x) for x in torch.stack(gates_t).cpu().tolist()] if gates_t else []
    return b"".join(parts), gates


def _chunk_device_plan(torch, dev, cshape, full_shape, levels, stride, block):
    """Per-chunk-shape integer-index schedule for the device wave inner loop.

    ``pos_dev`` addresses a contiguous flattened chunk value block.  The
    corresponding ``recon_off_dev`` uses full-tensor strides, so adding a
    chunk's global-flat origin addresses the device-resident reconstruction
    without scanning a full chunk-sized boolean mask at every stage.  Coordinate
    tuples are retained only for interpolation.  The plan is cached per
    chunk shape within one tensor encode/decode.
    """
    plan = stage_plan(cshape, levels, stride, block)
    full_strides = np.cumprod((1,) + tuple(full_shape)[:0:-1])[::-1].astype(np.int64)
    counts: list[int] = []
    pos_dev: list = []
    recon_off_dev: list = []
    interp_dev: list = []
    for m, st, ax in plan:
        pos = np.flatnonzero(m).astype(np.int64, copy=False)
        coords_np = np.unravel_index(pos, cshape)
        recon_off = np.zeros(pos.shape, np.int64)
        for cc, gs in zip(coords_np, full_strides):
            recon_off += cc * gs
        counts.append(int(pos.size))
        pos_dev.append(torch.from_numpy(np.ascontiguousarray(pos)).to(dev))
        recon_off_dev.append(
            torch.from_numpy(np.ascontiguousarray(recon_off, dtype=np.int64)).to(dev)
        )
        if ax and pos.size:
            coords = tuple(
                torch.from_numpy(np.ascontiguousarray(cc)).to(dev) for cc in coords_np
            )
            interp_dev.append((coords, st, ax))
        else:
            interp_dev.append(None)
    center = default_interp_center(len(cshape))
    return counts, pos_dev, recon_off_dev, interp_dev, center


def _decompress_chunked(
    payload: bytes,
    shape: tuple[int, ...],
    ebs: list[float],
    radius: int,
    predictor: ChunkedGNNPredictor,
    edges: tuple[int, ...],
    gates: list[int] | None = None,
) -> np.ndarray:
    c = 1
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros((c, *shape), np.float32)
    axes = _anchor_axes(shape, stride, block)
    _log(f"decode: shape={shape} edges={edges} decoding anchors...")
    anchor_bar = _progress_bar("decode anchors", 1, unit="stage")
    off = _decode_anchor_stage(payload, 0, recon, axes, ebs[0], radius)
    anchor_bar.update(1)
    anchor_bar.close()
    geom_bar = _progress_bar(
        "decode geometry", _geometry_stages(len(shape), predictor.levels), unit="stage"
    )
    predictor.begin(shape, edges, channels=c, geometry_progress=geom_bar.update)
    geom_bar.close()
    coarse_bar = _progress_bar(
        "decode anchor embeddings", predictor.n_chunks, unit="chunk"
    )
    predictor.anchor_coarse(recon, progress=coarse_bar.update)
    coarse_bar.close()
    torch = predictor._torch
    dev = predictor.device
    recon_t = torch.from_numpy(recon).to(dev)  # device-resident, same as encode
    gates_t = (
        None if gates is None else torch.tensor(gates, dtype=torch.int64, device=dev)
    )
    waves = _chunk_waves(predictor.grid)
    _log(f"decode: anchors done, {predictor.n_chunks} chunks/model passes")
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    index_cache: dict = {}  # cshape -> device stage schedule (see _chunk_device_plan)
    full_strides = np.cumprod((1,) + shape[:0:-1])[::-1].astype(np.int64)
    recon_flat = recon_t.reshape(c, -1)
    gi = 0
    bar = _progress_bar("decode", predictor.n_chunks)
    for group in waves:
        for ci in group:
            ids = [ci]
            cshape = tuple(sl.stop - sl.start for sl in predictor.chunk_slices(ids[0]))
            if cshape not in index_cache:
                index_cache[cshape] = _chunk_device_plan(
                    torch, dev, cshape, shape, predictor.levels, stride, block
                )
            counts, _, recon_off_dev, interp_dev, center = index_cache[cshape]
            origin_bases = [
                int(
                    sum(
                        sl.start * st
                        for sl, st in zip(predictor.chunk_slices(ci), full_strides)
                    )
                )
                for ci in ids
            ]
            predictor.start_wave(ids, recon_t)
            for s in range(1, len(counts)):
                n = counts[s]
                tables = stage_tables[s]
                if n == 0:
                    for _ in ids:
                        _c, _o, off = unpack_stage(
                            payload,
                            off,
                            rans_levels=np.zeros(0, np.uint8),
                            rans_tables=tables,
                        )
                    continue
                pred, scale = predictor.predict_wave_stage(s, recon_t, ebs[s])
                for bi in range(len(ids)):
                    sls = predictor.chunk_slices(ids[bi])
                    p = pred[bi][None, :]
                    sc = scale[bi]
                    if gates_t is not None:
                        g = gates_t[gi]
                        gi += 1
                        coords_t, st_i, ax_i = interp_dev[s]
                        ip_cubic = _interp_stage_pred_t(
                            torch,
                            recon_t,
                            sls,
                            coords_t,
                            st_i,
                            ax_i,
                            center,
                            cubic=True,
                        )
                        ip_linear = _interp_stage_pred_t(
                            torch,
                            recon_t,
                            sls,
                            coords_t,
                            st_i,
                            ax_i,
                            1,
                            cubic=False,
                        )
                        ip = torch.where((g >> 8) == 2, ip_linear, ip_cubic)
                        p, sc = _gate_apply_t(
                            torch,
                            pred[bi],
                            sc,
                            ip,
                            ebs[s],
                            (g >> 4) & 15,
                            g & 15,
                        )
                    levels64 = scale_to_level(
                        sc.cpu().numpy()[None, :], ebs[s]
                    ).reshape(-1)
                    codes, outliers, off = unpack_stage(
                        payload, off, rans_levels=levels64, rans_tables=tables
                    )
                    recon_stage = _dequantize_t(
                        torch,
                        p,
                        torch.from_numpy(codes.astype(np.int64)).to(dev),
                        torch.from_numpy(outliers).to(dev),
                        ebs[s],
                        radius,
                    )
                    gpos = recon_off_dev[s] + origin_bases[bi]
                    recon_flat.index_copy_(1, gpos, recon_stage.reshape(c, -1))
            predictor.finish_wave(recon_t)
            peak = _cuda_peak(predictor)
            if peak:
                bar.set_postfix_str(f"peak {peak / 1e9:.2f}GB")
            bar.update(1)
    bar.close()
    if off != len(payload):
        raise ValueError("trailing bytes in DeepSZ GNN payload")
    if gates is not None and gi != len(gates):
        raise ValueError("gate list length does not match the stream")
    return recon_t[0].cpu().numpy()


class GNNCompressorCodec:
    """Usable Python codec for GNN-backed DeepSZ tensor compression.

    The codec is initialized from a GNN checkpoint path. ``compress`` accepts a
    numpy array or torch tensor of any rank and returns bytes. ``uncompress``
    accepts those bytes and returns a torch tensor with the original shape and
    dtype.

    The tensor is always min-max normalized to [0, 1] internally (the GNN
    operates in that range regardless of the input's raw scale), and
    ``error_bound`` is relative to the data range: it is applied directly to
    the normalized tensor, so ``eb=0.01`` means 1% of ``max(x) - min(x)``
    regardless of the tensor's raw units.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        error_bound: float = 1e-2,  # relative to the tensor's (max - min); see class docstring
        *,
        levels: int | str = "auto",
        radius: int = 1 << 15,
        device: str | None = None,  # None -> cuda if available, else cpu
        zstd_level: int = 9,
        eb_ratio: float | None = None,  # coarsest-level factor; None=auto (fast/sweep)
        tune: str = "fast",
        strict_checkpoint: bool = True,
        chunk_size: int | tuple[int, ...] | None = None,
        fp16: bool = True,
        compile: bool | str = "auto",
        gate: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"GNN checkpoint not found: {self.checkpoint_path}")
        if error_bound <= 0:
            raise ValueError("error_bound must be > 0")
        if tune not in ("fast", "size"):
            raise ValueError("tune must be 'fast' or 'size'")

        self.error_bound = float(error_bound)
        # levels: an explicit int fixes the dyadic schedule depth; "auto" (the
        # default) picks it per input shape at compress time (see _auto_levels),
        # since anchor_stride = 2**levels is capped by the smallest axis. In auto
        # mode self.levels / self.anchor_stride stay None until compress resolves
        # them; decode always reads the resolved levels back from the stream.
        if isinstance(levels, str):
            if levels != "auto":
                raise ValueError("levels must be a positive int or 'auto'")
            self.auto_levels = True
            self.levels = None
            self.anchor_stride = None
        else:
            self.auto_levels = False
            self.levels = int(levels)
            if self.levels < 1:
                raise ValueError("levels must be >= 1")
            # A level is one dyadic refinement, so inference always starts on the
            # unique coarse grid that reaches unit stride after ``levels`` steps.
            self.anchor_stride = 1 << self.levels
        self.radius = int(radius)
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.zstd_level = int(zstd_level)
        # coarsest-level error-bound factor (finest level always keeps full eb);
        # None -> auto (fast: _GNN_EB_COARSE_FACTOR, size: sweep). Depth-normalised
        # per resolved levels at compress time so it is rank-invariant.
        self.eb_ratio = eb_ratio
        self.tune = tune
        self.strict_checkpoint = bool(strict_checkpoint)
        # chunk_size: None = auto (whole-tensor for small inputs, otherwise the
        # largest near-isotropic chunk within _AUTO_CHUNK_THRESHOLD points);
        # 0 = force whole-tensor; an int or per-axis tuple forces chunked with
        # those edges (multiples of anchor_stride).
        self.chunk_size = chunk_size
        # fp16: run the message-pass matmuls in fp16 autocast (cuda only; the
        # readout stays fp32). ~2x on the GNN forward, may cost a little ratio at
        # small eb. Stored in meta so decode uses the same float path.
        self.fp16 = bool(fp16)
        # compile: torch.compile the message-pass embed. "auto" (default) decides
        # per chunk count from a benchmark-backed crossover (_COMPILE_AUTO_CROSSOVER,
        # currently None -> off: compile never beat eager here, see the constant).
        # True forces it (still gated by _COMPILE_MIN_CHUNKS); False disables it.
        # The resolved decision is stored in meta so decode replays the same float
        # path. First encode pays a one-off compilation cost.
        if isinstance(compile, str):
            if compile != "auto":
                raise ValueError("compile must be a bool or 'auto'")
            self.auto_compile = True
            self.compile = False
        else:
            self.auto_compile = False
            self.compile = bool(compile)
        # gate: scale-gated classical fallback. Applies to both the chunked and
        # whole-tensor path -- the gate is device-only, so a gated whole-tensor
        # encode is realised as a single chunk covering the shape (see compress).
        # The encoder rate-selects GNN, cubic/averaged interpolation, or
        # linear/single-axis interpolation per chunk-stage and stores the winner
        # in the header. It self-disables wherever neither fallback pays.
        self.gate = bool(gate)
        self.checkpoint_hash = self._checkpoint_hash()

    def _chunk_edges(
        self, shape: tuple[int, ...], anchor_stride: int
    ) -> tuple[int, ...] | None:
        """Chunk edges for this shape, or None for the whole-tensor path."""
        cs = self.chunk_size
        if cs == 0:
            return None
        if cs is None:
            if int(np.prod(shape)) <= _AUTO_CHUNK_THRESHOLD:
                return None
            return _auto_chunk_edges(shape, anchor_stride)
        edges = (
            (int(cs),) * len(shape) if np.isscalar(cs) else tuple(int(e) for e in cs)
        )
        if len(edges) != len(shape):
            raise ValueError("chunk_size must be scalar or one entry per axis")
        for e in edges:
            if e < anchor_stride or e % anchor_stride:
                raise ValueError(
                    "chunk_size must be a positive multiple of anchor_stride"
                )
        return edges

    def compress(self, x: Any, error_bound: float | None = None) -> bytes:
        """Compress a numpy array or torch tensor of any rank into bytes."""
        arr = np.asarray(_as_numpy(x))
        if arr.size == 0:
            raise ValueError("cannot compress an empty tensor")
        if arr.dtype.kind not in "biuf":
            raise TypeError(f"unsupported dtype {arr.dtype}; expected numeric data")

        dtype = np.dtype(arr.dtype)
        original_shape = tuple(int(n) for n in arr.shape)
        shape = original_shape if original_shape else (1,)
        values = arr.reshape(shape).astype(np.float32, copy=False)
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        # Normalize to [0, 1]: the GNN always operates in that range, and it
        # makes error_bound naturally relative -- applied directly below, with
        # no separate rescale, it means a fraction of (vmax - vmin).
        values = (values - vmin) / (vmax - vmin)
        # Integer sources: the final decompressed value is rounded to the
        # nearest raw integer (_restore_dtype), so the quantizer must verify
        # the bound against that rounded value, not the normalized one -- see
        # quantize()'s round_output=(span, offset) contract.
        round_output = (vmax - vmin, vmin) if dtype.kind in "bi" else False
        eb = self.error_bound if error_bound is None else float(error_bound)
        if eb <= 0:
            raise ValueError("error_bound must be > 0")

        # eb_ratio is a coarsest-level factor (see _GNN_EB_COARSE_FACTOR); depth-
        # normalise it against the resolved schedule depth so the coarse/fine
        # spread is the same across ranks. The set dedups the levels==1 case
        # where every factor collapses to a flat 1.0 (one encode, not four).
        coarse_candidates = (
            [float(self.eb_ratio)]
            if self.eb_ratio is not None
            else (
                list(_GNN_EB_COARSE_SWEEP)
                if self.tune == "size"
                else [_GNN_EB_COARSE_FACTOR]
            )
        )
        levels = _auto_levels(shape) if self.auto_levels else self.levels
        anchor_stride = 1 << levels
        ratio_candidates = sorted(
            {_per_step_eb_ratio(c, levels) for c in coarse_candidates}
        )
        edges = self._chunk_edges(shape, anchor_stride)
        if edges is None and self.gate:
            # The scale-gated interp fallback lives entirely in the device chunked
            # inner loop; the numpy whole-tensor path has no gate. So when the gate
            # is enabled, realise a "gated whole-tensor" encode as a single chunk
            # covering the whole shape (grid 1 per axis, edges rounded up to the
            # anchor stride) -- one GPU pass, same memory footprint as whole-tensor,
            # but the gate applies. gate=False keeps the plain numpy whole path.
            edges = tuple(-(-n // anchor_stride) * anchor_stride for n in shape)
        # torch.compile costs seconds of dynamo warmup per process; only worth
        # it when there are enough chunk waves to amortize. "auto" defers to the
        # benchmark-backed crossover; explicit True is honored past a floor.
        # Frozen into the stream meta so decode replays the same float path.
        nchunks = (
            int(np.prod([-(-n // e) for n, e in zip(shape, edges)]))
            if edges is not None
            else 0
        )
        if self.auto_compile:
            want_compile = (
                _COMPILE_AUTO_CROSSOVER is not None
                and nchunks >= _COMPILE_AUTO_CROSSOVER
            )
        else:
            want_compile = self.compile and nchunks >= _COMPILE_MIN_CHUNKS
        use_compile = want_compile and edges is not None
        candidates: list[tuple[int, bytes]] = []
        for ratio in ratio_candidates:
            gates = None
            if edges is None:
                payload = self._compress_payload(
                    values, round_output, eb, ratio, levels, anchor_stride
                )
            else:
                payload, gates = self._compress_chunked_payload(
                    values, round_output, eb, ratio, edges, use_compile,
                    levels, anchor_stride,
                )
                if gates is not None and not any(gates):
                    gates = None  # gate never fired -> plain ungated stream
            meta = {
                "shape": list(original_shape),
                "dtype": dtype.str,
                "error_bound": eb,
                "levels": levels,
                "radius": self.radius,
                "vmin": vmin,
                "vmax": vmax,
                "eb_ratio": ratio,
                "checkpoint_hash": self.checkpoint_hash.hex(),
            }
            if edges is not None:
                meta["chunks"] = list(edges)
                meta["m_tile"] = int(_gp._M_TILE)  # replay the exact float path
                meta["fp16"] = bool(self.fp16)
                meta["compiled"] = bool(use_compile)
            if gates is not None:
                # ponytail: JSON list of one small int per chunk-stage; pack to
                # base64 bytes if header size ever matters at huge chunk counts
                meta["gates"] = gates
            stream = _write_stream(meta, payload, self.zstd_level)
            candidates.append((len(stream), stream))
        return min(candidates, key=lambda item: item[0])[1]

    def uncompress(self, stream: bytes | bytearray | memoryview):
        """Decompress bytes from ``compress`` and return a torch tensor."""
        import torch

        meta, payload = _read_stream(bytes(stream))
        got_hash = meta["checkpoint_hash"]
        if self.strict_checkpoint and got_hash != self.checkpoint_hash.hex():
            raise ValueError("checkpoint hash differs from the stream metadata")

        original_shape = tuple(int(n) for n in meta["shape"])
        shape = original_shape or (1,)
        dtype = np.dtype(meta["dtype"])
        vmin = float(meta["vmin"])
        vmax = float(meta["vmax"])
        if vmax <= vmin:
            vmax = vmin + 1.0

        if "chunks" in meta:
            edges = tuple(int(e) for e in meta["chunks"])
            levels = int(meta["levels"])
            anchor_stride = 1 << levels
            predictor = self._chunked_predictor(levels, meta)
            ebs = _chunk_stage_ebs(
                shape,
                levels,
                anchor_stride,
                _ANCHOR_BLOCK,
                float(meta["error_bound"]),
                float(meta["eb_ratio"]),
            )
            saved_tile = _gp._M_TILE
            _gp._M_TILE = int(meta["m_tile"])  # match encode path
            try:
                values = _decompress_chunked(
                    payload,
                    shape,
                    ebs,
                    int(meta["radius"]),
                    predictor,
                    edges,
                    gates=meta.get("gates"),
                )
            finally:
                _gp._M_TILE = saved_tile
            values = values * (vmax - vmin) + vmin  # undo compress()'s [0, 1] normalize
            out = _restore_dtype(values.reshape(original_shape), dtype)
            return torch.as_tensor(out)

        levels = int(meta["levels"])
        anchor_stride = 1 << levels
        predictor = self._predictor(levels, meta)
        masks = stage_masks(
            shape,
            levels,
            anchor_stride,
            _ANCHOR_BLOCK,
        )
        ebs = stage_ebs(
            shape,
            levels,
            anchor_stride,
            _ANCHOR_BLOCK,
            float(meta["error_bound"]),
            float(meta["eb_ratio"]),
        )
        values = _decompress_region(
            payload, shape, masks, ebs, int(meta["radius"]), predictor, True
        )
        values = values * (vmax - vmin) + vmin  # undo compress()'s [0, 1] normalize
        out = _restore_dtype(values.reshape(original_shape), dtype)
        return torch.as_tensor(out)

    decompress = uncompress

    def _compress_payload(
        self,
        values: np.ndarray,
        round_output: bool | tuple[float, float],
        eb: float,
        eb_ratio: float,
        levels: int,
        anchor_stride: int,
    ) -> bytes:
        predictor = self._predictor(levels)
        masks = stage_masks(values.shape, levels, anchor_stride, _ANCHOR_BLOCK)
        ebs = stage_ebs(
            values.shape,
            levels,
            anchor_stride,
            _ANCHOR_BLOCK,
            eb,
            eb_ratio,
        )
        stats = _empty_stats(len(masks))
        payload, _ = _compress_region(
            values[None, ...],
            masks,
            ebs,
            predictor,
            self.radius,
            round_output,
            stats,
        )
        return payload

    def _compress_chunked_payload(
        self,
        values: np.ndarray,
        round_output: bool | tuple[float, float],
        eb: float,
        eb_ratio: float,
        edges: tuple[int, ...],
        use_compile: bool,
        levels: int,
        anchor_stride: int,
    ) -> tuple[bytes, list[int] | None]:
        predictor = self._chunked_predictor(levels)
        predictor.compile = bool(use_compile)
        ebs = _chunk_stage_ebs(
            values.shape,
            levels,
            anchor_stride,
            _ANCHOR_BLOCK,
            eb,
            eb_ratio,
        )
        payload, gates = _compress_chunked(
            values[None, ...],
            ebs,
            self.radius,
            round_output,
            predictor,
            edges,
            gate=self.gate,
        )
        return payload, gates

    def _chunked_predictor(
        self,
        levels: int,
        meta: dict[str, Any] | None = None,
    ) -> ChunkedGNNPredictor:
        # vmin/vmax are always 0.0/1.0: compress() normalizes the tensor to
        # [0, 1] up front, so the predictor never sees raw-scale values.
        anchor_stride = 1 << levels
        predictor = ChunkedGNNPredictor(
            self.checkpoint_path,
            0.0,
            1.0,
            device=self.device,
            levels=levels,
            anchor_stride=anchor_stride,
            anchor_block=_ANCHOR_BLOCK,
        )
        # encode: from the codec flag; decode: replay the stream's float path
        predictor.fp16 = self.fp16 if meta is None else bool(meta["fp16"])
        predictor.compile = self.compile if meta is None else bool(meta["compiled"])
        return predictor

    def _predictor(
        self,
        levels: int,
        meta: dict[str, Any] | None = None,
    ) -> GNNPredictor:
        anchor_stride = 1 << levels
        return GNNPredictor(
            self.checkpoint_path,
            0.0,
            1.0,
            max_radius=anchor_stride,
            device=self.device,
            levels=levels,
            anchor_stride=anchor_stride,
            anchor_block=_ANCHOR_BLOCK,
        )

    def _checkpoint_hash(self) -> bytes:
        import hashlib

        return hashlib.sha256(self.checkpoint_path.read_bytes()).digest()[:16]


GNNCodec = GNNCompressorCodec
