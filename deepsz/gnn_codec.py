"""Checkpoint-backed, tensor-shaped GNN compressor codec."""

from __future__ import annotations

import json
import itertools
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
_VERSION = 13
_PREFIX = "<8sII"
_PREFIX_SIZE = struct.calcsize(_PREFIX)
_ANCHOR_BLOCK = 1

# auto mode: whole-tensor below this many points, chunked above (whole-tensor
# memory is ~30*L*K*d bytes/point in transients — ~2^21 points is a few GB)
_AUTO_CHUNK_THRESHOLD = 1 << 21
# Pointwise scale masks can change the reconstruction seen by every finer
# stage. Restrict them to sufficiently dense stages; sparse/coarse stages still
# rate-select GNN versus a coherent whole-stage classical predictor.
_IMPLICIT_GATE_MIN_POINTS = 256
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
# explicitly (2-D 8, 4-D 5 -> anchor_stride 256 / 32; 4-D keeps the edge-32
# operating point). 3-D is also checkpoint-dependent: aggregation level 1 uses
# level 7, while the wider level-2 neighbourhood uses level 6. Ranks outside the
# table fall back to the constant-points-per-chunk rule below.
_RANK_LEVELS = {2: 8, 4: 5, 5: 4}


def _auto_levels(shape: tuple[int, ...], agg_level: int = 2) -> int:
    """Levels for ``levels="auto"``, chosen from the rank of the input.

    The schedule depth follows the input rank, not its size: a chunk is about one
    to a few anchor cells (``anchor_stride**ndim = 2**(levels*ndim)`` points), so
    keying the stride off the rank keeps an ~constant number of points per chunk.
    ``_RANK_LEVELS`` fixes the tuned common ranks. For 3-D, the checkpoint's
    neighbourhood aggregation level selects 7 levels for aggregation level 1
    and 6 levels for aggregation level 2 or greater. Other ranks fall back to
    ``floor(log2(points_per_chunk) / ndim)`` from the auto-chunk point budget
    (``_AUTO_CHUNK_THRESHOLD``), so a cell stays within budget -- higher-rank
    tensors get a smaller stride, lower-rank a larger one.

    A size guard then keeps ``anchor_stride <= max_axis / 2`` so a small or
    low-rank input still carries at least two anchors on its largest axis rather
    than collapsing to a lone corner anchor."""
    ndim = len(shape)
    if ndim == 3:
        rank_levels = 7 if agg_level == 1 else 6
    elif ndim in _RANK_LEVELS:
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


# Implicit point gate. Each chunk-stage rate-selects the GNN alone or one of
# three interpolation predictors for points whose GNN scale is at least a
# threshold. The decoder predicts the same scale field, so the point mask is
# free. The control byte stores predictor kind in two bits (0=GNN only,
# 1=cubic/axis-average, 2=first-axis linear, 3=last-axis linear) and threshold
# in six bits. Hybrid controls have one following byte containing the classical
# rANS scale level in six bits and the threshold direction in bit six. Thus an
# ignored/coarse stage costs one byte and an enabled point gate costs two,
# independent of its number of points.


def _gate_pack_t(kind, threshold_level):
    """Pack predictor kind and 6-bit implicit threshold into one device byte."""
    return kind | (threshold_level << 2)


def _gate_unpack_t(packed):
    """Inverse of :func:`_gate_pack_t`, returning kind and threshold."""
    return packed & 3, (packed >> 2) & 63


def _split_gate_payload(meta: dict[str, Any], payload: bytes) -> tuple[list[int] | None, bytes]:
    """Remove packed gate bytes from the decompressed payload."""
    count = int(meta.get("gate_count", 0))
    if count < 0 or count > len(payload):
        raise ValueError("invalid packed gate count")
    if not count:
        return None, payload
    return list(payload[:count]), payload[count:]


def _chunk_slices(shape: tuple[int, ...], edges: tuple[int, ...]):
    """Yield raster-ordered independent chunk slices."""
    grid = tuple(-(-n // edge) for n, edge in zip(shape, edges))
    for index in itertools.product(*(range(n) for n in grid)):
        yield tuple(
            slice(i * edge, min((i + 1) * edge, n))
            for i, edge, n in zip(index, edges, shape)
        )


def _pack_rate_selected_chunk(mode: int, stream: bytes) -> bytes:
    """Frame one independently decodable per-chunk codec choice."""
    return struct.pack("<BQ", int(mode), len(stream)) + stream


def _unpack_rate_selected_chunk(
    payload: bytes, off: int
) -> tuple[int, bytes, int]:
    """Inverse of :func:`_pack_rate_selected_chunk` with bounds checks."""
    if off + 9 > len(payload):
        raise ValueError("truncated rate-selected chunk header")
    mode, size = struct.unpack_from("<BQ", payload, off)
    if mode not in (0, 1):
        raise ValueError("invalid rate-selected chunk mode")
    off += 9
    end = off + size
    if end > len(payload):
        raise ValueError("truncated rate-selected chunk stream")
    return mode, payload[off:end], end


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


def _laplace_bits_t(torch, absr, b, eb, radius):
    """rANS-aligned rate estimate used to rank gate settings.

    Scales are snapped to the same 64-level grid as ``scale_to_level``. Regular
    symbols are capped at the rANS table precision (24 bits), while outliers pay
    that marker cost plus the raw float32 stored by ``pack_stage``.
    """
    lo = eb / SCALE_LO_DIV
    hi = eb * SCALE_HI_MULT
    b = b.double().clamp(lo, hi)
    level = torch.round(
        (torch.log(b) - np.log(lo)) / (np.log(hi) - np.log(lo)) * 63.0
    )
    b = torch.exp(np.log(lo) + level * ((np.log(hi) - np.log(lo)) / 63.0))
    k = torch.round(absr.double().abs() / (2 * eb))
    p = torch.where(
        k == 0,
        -torch.expm1(-eb / b),
        0.5 * torch.exp(-((2 * k - 1) * eb / b)) * -torch.expm1(-2 * eb / b),
    )
    regular = (-torch.log2(p)).clamp_max(24.0)
    return torch.where(k >= radius, torch.full_like(regular, 56.0), regular)


def _scale_level_t(torch, scale, eb):
    """Torch twin of :func:`scale_to_level` for implicit gate decisions."""
    lo = float(eb) / SCALE_LO_DIV
    hi = float(eb) * SCALE_HI_MULT
    return torch.round(
        (torch.log(scale.double().clamp(lo, hi)) - np.log(lo))
        / (np.log(hi) - np.log(lo))
        * 63.0
    ).to(torch.int64)


def _gate_select_t(torch, r_g, r_is, b, eb, radius=1 << 15):
    """Choose an implicit scale gate by modeled rANS rate.

    Returns device scalars ``(kind, threshold_level, classical_scale_level,
    low_side)``. Kind zero keeps the GNN everywhere. For a positive kind,
    points on the selected side of the scale threshold use the indexed
    classical predictor and its constant entropy scale. The selector charges
    the extra parameter byte, so sparse coarse stages can leave the gate off.
    """
    dev = b.device
    lo = float(eb) / SCALE_LO_DIV
    hi = float(eb) * SCALE_HI_MULT
    log_step = (np.log(hi) - np.log(lo)) / 63.0
    gnn_bits = _laplace_bits_t(torch, r_g, b, eb, radius)
    gnn_cost = gnn_bits.sum()
    b_levels = _scale_level_t(torch, b, eb)
    best_cost = gnn_cost
    best_kind = torch.zeros((), dtype=torch.int64, device=dev)
    best_threshold = torch.zeros((), dtype=torch.int64, device=dev)
    best_level = torch.zeros((), dtype=torch.int64, device=dev)
    best_low_side = torch.zeros((), dtype=torch.int64, device=dev)
    for kind, r_i in enumerate(r_is, 1):
        mle = r_i.double().abs().mean().clamp(lo, hi)
        center = torch.round((torch.log(mle) - np.log(lo)) / log_step).to(torch.int64)
        levels = (
            center + torch.arange(-3, 4, dtype=torch.int64, device=dev)
        ).clamp(0, 63)
        for level in levels:
            scale = torch.exp(
                torch.tensor(np.log(lo), dtype=torch.float64, device=dev)
                + level.double() * log_step
            )
            delta = (
                _laplace_bits_t(torch, r_i, scale, eb, radius) - gnn_bits
            ).sum(dim=0)
            by_level = torch.zeros(64, dtype=torch.float64, device=dev)
            by_level.scatter_add_(0, b_levels.reshape(-1), delta.reshape(-1))
            # Entries t are the cost changes from selecting level >= t and
            # level <= t respectively. The latter retains the useful behavior
            # of the former precision-floor gate.
            tail_delta = torch.flip(
                torch.cumsum(torch.flip(by_level, dims=(0,)), dim=0),
                dims=(0,),
            )
            low_delta = torch.cumsum(by_level, dim=0)
            if b.numel() >= _IMPLICIT_GATE_MIN_POINTS:
                side_costs = torch.stack((tail_delta, low_delta))
                cost_delta, flat_choice = side_costs.reshape(-1).min(0)
                low_side = flat_choice // 64
                threshold = flat_choice % 64
            else:
                # threshold=0 on the high side selects the whole stage.
                cost_delta = tail_delta[0]
                low_side = torch.zeros((), dtype=torch.int64, device=dev)
                threshold = torch.zeros((), dtype=torch.int64, device=dev)
            cost = gnn_cost + cost_delta
            better = cost < best_cost
            best_cost = torch.where(better, cost, best_cost)
            best_kind = torch.where(
                better,
                torch.tensor(kind, dtype=torch.int64, device=dev),
                best_kind,
            )
            best_threshold = torch.where(better, threshold, best_threshold)
            best_level = torch.where(better, level, best_level)
            best_low_side = torch.where(better, low_side, best_low_side)
    # A hybrid record consumes one more raw byte than a GNN-only control.
    use_classical = best_cost + 8.0 < gnn_cost
    zero = torch.zeros((), dtype=torch.int64, device=dev)
    return (
        torch.where(use_classical, best_kind, zero),
        torch.where(use_classical, best_threshold, zero),
        torch.where(use_classical, best_level, zero),
        torch.where(use_classical, best_low_side, zero),
    )


def _gate_apply_t(
    torch,
    pred_bi,
    ip,
    scale,
    eb,
    gate_kind,
    threshold_level,
    low_side,
):
    """Apply a decoder-reproducible point mask derived from the GNN scale."""
    scale_level = _scale_level_t(torch, scale, eb)
    in_scale_region = torch.where(
        low_side > 0,
        scale_level <= threshold_level,
        scale_level >= threshold_level,
    )
    use_classical = (gate_kind > 0) & in_scale_region
    return torch.where(
        use_classical.reshape(1, -1),
        ip,
        pred_bi.unsqueeze(0),
    ).to(torch.float32)


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

    Extended-block raster schedule: each chunk owns its internal high-face
    planes and inherits its low faces from the already-decoded up/left
    neighbour (column-split out of its coded set), so interiors see both-sided
    context across every chunk seam. Chunks run in strict raster order (owner
    i-1 before inheritor i on every axis); the error bound holds for any order
    since every cell is coded exactly once by its owner.

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
    # hand recon to the GPU for the wave loop; it never returns to host on encode
    recon_t = torch.from_numpy(recon).to(dev)
    recon_flat = recon_t.reshape(c, -1)
    waves = [list(range(predictor.n_chunks))]  # raster order (see docstring)
    _log(
        f"encode: anchors done, {predictor.n_chunks} chunks, "
        f"{predictor.n_chunks} model passes"
    )
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    index_cache: dict = {}  # cshape -> device stage schedule (see _chunk_device_plan)
    full_strides = np.cumprod((1,) + shape[:0:-1])[::-1].astype(np.int64)
    # Per stage-chunk (control, classical scale) device scalars. The second
    # byte is omitted from the serialized list when the control keeps the GNN.
    gates_t: list = [] if gate else None
    bar = _progress_bar("encode", predictor.n_chunks)
    for group in waves:
        for ci in group:
            ids = [ci]
            cshape = tuple(sl.stop - sl.start for sl in predictor.chunk_slices(ids[0]))
            low_ax = predictor.low_axes(ids[0])
            key = (cshape, low_ax)
            if key not in index_cache:
                index_cache[key] = _chunk_device_plan(
                    torch, dev, cshape, shape, predictor.levels, stride, block, low_ax
                )
            full_counts, counts, pos_dev, recon_off_dev, interp_dev, pred_idx_dev, center = (
                index_cache[key]
            )
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
            for s in range(1, len(full_counts)):
                tables = stage_tables[s]
                if full_counts[s] == 0:  # no cells at this level -> no forward
                    wave_pending.extend([(None, tables)] * len(ids))
                    continue
                pred, scale = predictor.predict_wave_stage(s, recon_t, ebs[s])
                pi = pred_idx_dev[s]
                if pi is not None:  # column-split: keep only the coded rows
                    pred = pred.index_select(1, pi)
                    scale = scale.index_select(1, pi)
                if counts[s] == 0:  # every cell here is an inherited low face
                    wave_pending.extend([(None, tables)] * len(ids))
                    continue
                pos = pos_dev[s]
                for bi in range(len(ids)):
                    sls = predictor.chunk_slices(ids[bi])
                    cvals = vblocks[bi].index_select(1, pos)  # (C, n)
                    p = pred[bi][None, :]
                    sc = scale[bi]
                    gate_record = None
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
                        ip_linear_last = _interp_stage_pred_t(
                            torch,
                            recon_t,
                            sls,
                            coords_t,
                            st_i,
                            ax_i,
                            2,
                            cubic=False,
                        )
                        gk, gt, gl, gd = _gate_select_t(
                            torch,
                            (cvals - p).abs(),
                            (
                                (cvals - ip_cubic).abs(),
                                (cvals - ip_linear).abs(),
                                (cvals - ip_linear_last).abs(),
                            ),
                            sc,
                            ebs[s],
                            radius,
                        )
                        packed_gate = _gate_pack_t(gk, gt)
                        gate_param = gl | (gd << 6)
                        gate_record = torch.stack((packed_gate, gate_param))
                        gates_t.append(gate_record)
                        ip = torch.where(
                            gk == 2,
                            ip_linear,
                            torch.where(gk == 3, ip_linear_last, ip_cubic),
                        )
                        p = _gate_apply_t(
                            torch, pred[bi], ip, sc, ebs[s], gk, gt, gd
                        )
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
                            (
                                gate_record.to("cpu", non_blocking=True)
                                if gate_record is not None
                                else None
                            ),
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
                codes_c, out_c, sc_c, tables, eb_s, gate_c = item
                codes_np = codes_c.numpy().astype(np.uint32)
                out_np = out_c.numpy().astype(np.float32)
                gate_kind, gate_threshold, gate_level, gate_low_side = (
                    (0, 0, 0, 0)
                    if gate_c is None
                    else (
                        *(
                            int(v)
                            for v in _gate_unpack_t(gate_c[0])
                        ),
                        int(gate_c[1]) & 63,
                        (int(gate_c[1]) >> 6) & 1,
                    )
                )
                levels = scale_to_level(sc_c.numpy()[None, :], eb_s).reshape(-1)
                if gate_kind:
                    mask = (
                        levels <= gate_threshold
                        if gate_low_side
                        else levels >= gate_threshold
                    )
                    levels[mask] = gate_level
                parts.append(
                    pack_stage(
                        codes_np,
                        out_np,
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
        pairs = torch.stack(gates_t).cpu().tolist() if gates_t else []
        if any(int(pair[0]) & 3 for pair in pairs):
            gates = []
            for control, scale_level in pairs:
                gates.append(int(control))
                if int(control) & 3:
                    gates.append(int(scale_level))
        if gates is not None and any(g < 0 or g > 255 for g in gates):
            raise AssertionError("packed gate decision does not fit in one byte")
    return b"".join(parts), gates


def _chunk_device_plan(
    torch, dev, cshape, full_shape, levels, stride, block, low_axes=()
):
    """Per-chunk-shape integer-index schedule for the device wave inner loop.

    ``pos_dev`` addresses a contiguous flattened chunk value block.  The
    corresponding ``recon_off_dev`` uses full-tensor strides, so adding a
    chunk's global-flat origin addresses the device-resident reconstruction
    without scanning a full chunk-sized boolean mask at every stage.  Coordinate
    tuples are retained only for interpolation.  The plan is cached per
    (chunk shape, ``low_axes``) within one tensor encode/decode.

    ``low_axes`` (grow mode): axes whose local coord-0 hyperplane is an inherited
    high face of the up/left neighbour -- already decoded, so it is *column-split*
    out of this chunk's coded set. ``pos_dev``/``recon_off_dev``/``interp_dev``
    hold only the kept (coded) rows; ``full_counts[s]`` is the unsplit stage size,
    which still drives ``predict_wave_stage`` (context is built over the whole
    extended block), and ``pred_idx_dev[s]`` gathers the kept rows out of the
    full-stage prediction (``None`` when nothing is split).
    """
    plan = stage_plan(cshape, levels, stride, block)
    full_strides = np.cumprod((1,) + tuple(full_shape)[:0:-1])[::-1].astype(np.int64)
    full_counts: list[int] = []
    counts: list[int] = []
    pos_dev: list = []
    recon_off_dev: list = []
    interp_dev: list = []
    pred_idx_dev: list = []
    for m, st, ax in plan:
        pos = np.flatnonzero(m).astype(np.int64, copy=False)
        coords_np = np.unravel_index(pos, cshape)
        full_counts.append(int(pos.size))
        if low_axes and pos.size:
            drop = np.zeros(pos.shape, bool)
            for a in low_axes:
                drop |= coords_np[a] == 0
            keep = ~drop
        else:
            keep = np.ones(pos.shape, bool)
        keep_idx = np.flatnonzero(keep)
        pred_idx_dev.append(
            None
            if keep_idx.size == pos.size
            else torch.from_numpy(np.ascontiguousarray(keep_idx)).to(dev)
        )
        pos = pos[keep]
        coords_np = tuple(cc[keep] for cc in coords_np)
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
    return full_counts, counts, pos_dev, recon_off_dev, interp_dev, pred_idx_dev, center


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
    torch = predictor._torch
    dev = predictor.device
    recon_t = torch.from_numpy(recon).to(dev)  # device-resident, same as encode
    waves = [list(range(predictor.n_chunks))]  # raster order, mirrors encode
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
            low_ax = predictor.low_axes(ids[0])
            key = (cshape, low_ax)
            if key not in index_cache:
                index_cache[key] = _chunk_device_plan(
                    torch, dev, cshape, shape, predictor.levels, stride, block, low_ax
                )
            full_counts, counts, _, recon_off_dev, interp_dev, pred_idx_dev, center = (
                index_cache[key]
            )
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
            for s in range(1, len(full_counts)):
                tables = stage_tables[s]
                if full_counts[s] == 0:
                    for _ in ids:
                        _c, _o, off = unpack_stage(
                            payload,
                            off,
                            rans_levels=np.zeros(0, np.uint8),
                            rans_tables=tables,
                        )
                    continue
                pred, scale = predictor.predict_wave_stage(s, recon_t, ebs[s])
                pi = pred_idx_dev[s]
                if pi is not None:  # column-split: keep only the coded rows
                    pred = pred.index_select(1, pi)
                    scale = scale.index_select(1, pi)
                if counts[s] == 0:  # every cell here is an inherited low face
                    for _ in ids:
                        _c, _o, off = unpack_stage(
                            payload,
                            off,
                            rans_levels=np.zeros(0, np.uint8),
                            rans_tables=tables,
                        )
                    continue
                for bi in range(len(ids)):
                    sls = predictor.chunk_slices(ids[bi])
                    p = pred[bi][None, :]
                    sc = scale[bi]
                    gate_kind = 0
                    gate_threshold = 0
                    gate_level = 0
                    gate_low_side = 0
                    if gates is not None:
                        if gi >= len(gates):
                            raise ValueError("truncated implicit gate list")
                        packed_gate = torch.tensor(
                            gates[gi], dtype=torch.int64, device=dev
                        )
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
                        ip_linear_last = _interp_stage_pred_t(
                            torch,
                            recon_t,
                            sls,
                            coords_t,
                            st_i,
                            ax_i,
                            2,
                            cubic=False,
                        )
                        gk, gt = _gate_unpack_t(packed_gate)
                        if int(gk):
                            if gi >= len(gates):
                                raise ValueError("truncated implicit gate parameters")
                            gate_param = int(gates[gi])
                            gi += 1
                            if gate_param >= 128:
                                raise ValueError("invalid implicit gate parameter")
                            gl = torch.tensor(
                                gate_param & 63, dtype=torch.int64, device=dev
                            )
                            gd = torch.tensor(
                                (gate_param >> 6) & 1,
                                dtype=torch.int64,
                                device=dev,
                            )
                        else:
                            gl = torch.zeros((), dtype=torch.int64, device=dev)
                            gd = torch.zeros((), dtype=torch.int64, device=dev)
                        ip = torch.where(
                            gk == 2,
                            ip_linear,
                            torch.where(gk == 3, ip_linear_last, ip_cubic),
                        )
                        p = _gate_apply_t(
                            torch, pred[bi], ip, sc, ebs[s], gk, gt, gd
                        )
                        gate_kind = int(gk)
                        gate_threshold = int(gt)
                        gate_level = int(gl)
                        gate_low_side = int(gd)
                    levels64 = scale_to_level(
                        sc.cpu().numpy()[None, :], ebs[s]
                    ).reshape(-1)
                    if gate_kind:
                        mask = (
                            levels64 <= gate_threshold
                            if gate_low_side
                            else levels64 >= gate_threshold
                        )
                        levels64[mask] = gate_level
                    codes, outliers, off = unpack_stage(
                        payload,
                        off,
                        rans_levels=levels64,
                        rans_tables=tables,
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
        classical_fallback: bool = True,
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
        # gate: hybrid classical fallback. Applies to both the chunked and
        # whole-tensor path. Each chunk-stage rate-selects GNN, cubic/averaged
        # interpolation, or first/last-axis linear interpolation.
        self.gate = bool(gate)
        # Per-chunk safety net: independently encode each outer block with the
        # implicit GNN hybrid and tuned standard interpolation, then retain the
        # smaller bounded stream. gate=False remains the pure-GNN control.
        self.classical_fallback = bool(classical_fallback)
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

    def _compress_gnn_block(
        self,
        block: np.ndarray,
        absolute_eb: float,
        levels: int,
    ) -> bytes:
        """Compress one independent block with the implicit GNN hybrid."""
        block32 = block.astype(np.float32, copy=False)
        bmin = float(block32.min())
        bmax = float(block32.max())
        block_span = bmax - bmin if bmax > bmin else 1.0
        compile_mode: bool | str = "auto" if self.auto_compile else self.compile
        codec = GNNCompressorCodec(
            self.checkpoint_path,
            error_bound=absolute_eb / block_span,
            levels=levels,
            radius=self.radius,
            device=self.device,
            zstd_level=self.zstd_level,
            eb_ratio=self.eb_ratio,
            tune=self.tune,
            strict_checkpoint=self.strict_checkpoint,
            chunk_size=self.chunk_size,
            fp16=self.fp16,
            compile=compile_mode,
            gate=self.gate,
            classical_fallback=False,
        )
        return codec.compress(block)

    def _compress_interp_block(
        self,
        block: np.ndarray,
        absolute_eb: float,
        levels: int,
        anchor_stride: int,
    ) -> bytes | None:
        """Compress one independent block with tuned standard interpolation."""
        from .codec import compress as classic_compress
        from .predictor import InterpPredictor

        if block.dtype.kind in "biu" and block.dtype != np.dtype(np.uint8):
            # The standard stream currently stores only uint8 or float32. Do
            # not introduce an unchecked float-to-integer rounding path.
            return None
        classic_block = (
            block
            if block.dtype == np.dtype(np.uint8)
            else block.astype(np.float32, copy=False)
        )
        predictor = InterpPredictor(
            order="cubic",
            levels=levels,
            anchor_stride=anchor_stride,
            anchor_block=_ANCHOR_BLOCK,
        )
        stream, _ = classic_compress(
            np.ascontiguousarray(classic_block),
            absolute_eb,
            predictor,
            levels=levels,
            anchor_stride=anchor_stride,
            anchor_block=_ANCHOR_BLOCK,
            radius=self.radius,
            zstd_level=self.zstd_level,
            tune="size",
        )
        return stream

    def _compress_rate_selected_chunks(
        self,
        raw: np.ndarray,
        absolute_eb: float,
        relative_eb: float,
        levels: int,
        anchor_stride: int,
        edges: tuple[int, ...],
        original_shape: tuple[int, ...],
        dtype: np.dtype,
        vmin: float,
        vmax: float,
    ) -> bytes:
        """Choose the bounded GNN or interpolation stream independently per block."""
        parts = []
        choices = []
        for sls in _chunk_slices(raw.shape, edges):
            block = np.ascontiguousarray(raw[sls])
            gnn_stream = self._compress_gnn_block(block, absolute_eb, levels)
            interp_stream = self._compress_interp_block(
                block, absolute_eb, levels, anchor_stride
            )
            choices_for_block = [(1, gnn_stream)]
            if interp_stream is not None:
                choices_for_block.append((0, interp_stream))
            mode, stream = min(choices_for_block, key=lambda item: len(item[1]))
            choices.append((mode, stream))
            parts.append(_pack_rate_selected_chunk(mode, stream))

        if len(choices) == 1:
            return choices[0][1]
        meta = {
            "shape": list(original_shape),
            "dtype": dtype.str,
            "error_bound": relative_eb,
            "levels": levels,
            "radius": self.radius,
            "vmin": vmin,
            "vmax": vmax,
            "checkpoint_hash": self.checkpoint_hash.hex(),
            "fallback": "rate-selected-chunks",
            "select_chunks": list(edges),
        }
        return _write_stream(meta, b"".join(parts), self.zstd_level)

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
        if self.auto_levels:
            import torch

            agg_level = _gp._load_inference_model(
                self.checkpoint_path, torch, self.device
            )[3]
            levels = _auto_levels(shape, agg_level)
        else:
            levels = self.levels
        anchor_stride = 1 << levels
        ratio_candidates = sorted(
            {_per_step_eb_ratio(c, levels) for c in coarse_candidates}
        )
        edges = self._chunk_edges(shape, anchor_stride)
        if edges is None and self.gate:
            # The per-stage hybrid selector lives in the device chunked inner loop;
            # the numpy whole-tensor path has no selector. So when it is enabled,
            # realize a hybrid whole-tensor encode as one chunk covering the shape
            # (edges rounded up to the anchor stride). gate=False keeps the plain
            # numpy whole path as an explicit pure-GNN control.
            edges = tuple(-(-n // anchor_stride) * anchor_stride for n in shape)
        if self.gate and self.classical_fallback:
            return self._compress_rate_selected_chunks(
                arr.reshape(shape),
                eb * (vmax - vmin),
                eb,
                levels,
                anchor_stride,
                edges,
                original_shape,
                dtype,
                vmin,
                vmax,
            )
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
                meta["gate_count"] = len(gates)
                payload = bytes(gates) + payload
            stream = _write_stream(meta, payload, self.zstd_level)
            candidates.append((len(stream), stream))
        return min(candidates, key=lambda item: item[0])[1]

    def uncompress(self, stream: bytes | bytearray | memoryview):
        """Decompress bytes from ``compress`` and return a torch tensor."""
        import torch

        stream = bytes(stream)
        from .bitstream import MAGIC as CLASSIC_MAGIC

        if stream.startswith(CLASSIC_MAGIC):
            from .codec import decompress as classic_decompress

            return torch.as_tensor(classic_decompress(stream))

        meta, payload = _read_stream(stream)
        gates, payload = _split_gate_payload(meta, payload)
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

        if meta.get("fallback") == "rate-selected-chunks":
            from .codec import decompress as classic_decompress

            edges = tuple(int(e) for e in meta["select_chunks"])
            if len(edges) != len(shape) or any(e <= 0 for e in edges):
                raise ValueError("invalid rate-selected chunk edges")
            values = np.empty(shape, dtype=dtype)
            off = 0
            for sls in _chunk_slices(shape, edges):
                mode, chunk_stream, off = _unpack_rate_selected_chunk(payload, off)
                block = (
                    classic_decompress(chunk_stream)
                    if mode == 0
                    else self.uncompress(chunk_stream).numpy()
                )
                expected = tuple(values[sls].shape)
                if tuple(block.shape) != expected:
                    raise ValueError("rate-selected chunk shape mismatch")
                values[sls] = block
            if off != len(payload):
                raise ValueError("trailing bytes in rate-selected chunk payload")
            return torch.as_tensor(values.reshape(original_shape))

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
                    gates=gates,
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
