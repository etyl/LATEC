"""Checkpoint-backed, tensor-shaped GNN compressor codec."""

from __future__ import annotations

import base64
import json
import os
import struct
import sys
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import zstandard

from . import gnn_predictor as _gp
from .gnn_predictor import ChunkedGNNPredictor
from .levels import stage_ebs, stage_plan, stage_strides
from .predictor import END_EXTRAP, END_QUAD, default_interp_center
from .quantizer import dequantize, quantize
from .bitstream import pack_stage, unpack_stage
from .rans import SCALE_HI_MULT, SCALE_LO_DIV, build_laplace_tables, scale_to_level


_MAGIC = b"LATECGNN"
_VERSION = 13
_PREFIX = "<8sII"
_PREFIX_SIZE = struct.calcsize(_PREFIX)
_ANCHOR_BLOCK = 1

# auto mode: a single chunk covering the tensor below this many points, a grid
# of chunks above (a chunk's transients are ~30*L*K*d bytes/point — ~2^21 points
# is a few GB)
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
    # ponytail: env-gated so tests/CLI stay quiet; set LATEC_PROGRESS=1 to see it
    if not os.environ.get("LATEC_PROGRESS"):
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
    # env-gated (LATEC_PROGRESS) so tests/CLI stay quiet; disabled bar is a no-op.
    from tqdm import tqdm

    return tqdm(
        total=n,
        desc=tag,
        unit=unit,
        file=sys.stderr,
        disable=not os.environ.get("LATEC_PROGRESS"),
    )


def _geometry_stages(ndim: int, levels: int) -> int:
    """Number of masks/geometries in the dimension-generic stage schedule."""
    return 1 + levels * (2 * ndim - 1)


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


# float32 unit roundoff (half an eps): the largest relative error of one
# correctly-rounded float32 operation.
_F32_U = float(np.finfo(np.float32).eps) / 2


def roundtrip_slack(vmin: float, vmax: float) -> float:
    """Float32 normalize/denormalize slack, in the tensor's ORIGINAL units.

    ``error_bound`` is relative to the data range and is enforced on the [0, 1]
    normalized tensor: the codec guarantees ``|x_norm - recon_norm| <= eb``. The
    caller, though, sees ``recon_norm * S + vmin`` (``S = vmax - vmin``) compared
    against the untouched input, and both the normalize and the denormalize are
    float32 -- so the round trip adds representation error the quantizer never
    saw. Checking ``|v - out| <= eb * S`` as an exact inequality therefore asks
    for more precision than float32 can carry; the correct comparison allows
    this slack. (Measured on rti_normal.npy at eb=1e-3: the internal normalized
    bound held on every one of 1048576 points, yet 2 points sat 4.2e-8 over
    ``eb * S`` purely from the round trip -- well inside the slack below.)

    Budget, with ``u = _F32_U`` and each float32 op contributing ``(1 + e)``,
    ``|e| <= u``:

    * normalize ``xn = fl(fl(v - vmin) / S)`` gives
      ``|xn * S - (v - vmin)| <= 2u * S``;
    * denormalize ``out = fl(fl(rn * S) + vmin)`` gives
      ``|out - (rn * S + vmin)| <= u * S + u * max|v|``.

    Summing the two, ``|v - out| <= S * eb + u * (3S + max|v|)``, so the slack is
    ``u * (3S + max|v|)``. This replaces the ad-hoc one-output-ULP allowance the
    codec's tests used to grant, with a bound derived from the arithmetic.

    Note this is a float32 representation allowance, not slack in the codec's
    guarantee: the quantizer still enforces ``eb`` exactly in normalized units,
    so the excess over ``eb * S`` stays a few ULPs and never grows with eb.
    """
    span = float(vmax) - float(vmin)
    if span <= 0:
        return 0.0
    return _F32_U * (3.0 * span + max(abs(float(vmin)), abs(float(vmax))))


def _write_stream(meta: dict[str, Any], payload: bytes, zstd_level: int) -> bytes:
    header = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body = zstandard.ZstdCompressor(level=zstd_level).compress(payload)
    return struct.pack(_PREFIX, _MAGIC, _VERSION, len(header)) + header + body


def _read_stream(stream: bytes) -> tuple[dict[str, Any], bytes]:
    if len(stream) < _PREFIX_SIZE:
        raise ValueError("not a LATEC GNN stream")
    magic, version, header_len = struct.unpack_from(_PREFIX, stream, 0)
    if magic != _MAGIC:
        raise ValueError(f"not a LATEC GNN stream (bad magic {magic!r})")
    if version != _VERSION:
        raise ValueError(f"unsupported LATEC GNN stream version {version}")
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


# Scale-gated classical-predictor fallback: where the model's predicted scale b
# sits on one side of eb*2^T, its confidence is either genuine (high eb: the
# point is ~free), the learned-predictor precision floor talking (low eb), or the
# model simply wandering off a locally smooth field (large b) — only measuring
# can tell. The encoder sweeps (predictor, direction, T, shift) per chunk-stage
# against the true residuals. The candidates are the original cubic/axis-average
# interpolation and anisotropic linear interpolation along either the first or
# last active axis. Direction picks whether the fallback fires on the LOW side
# (b < eb*2^T -- confident/precision-floor region) or the HIGH side (b >= eb*2^T
# -- the uncertain region where a smooth interp often beats a flailing model).
# The winner travels in the header, so the gate self-disables (T=0) wherever
# neither classical predictor pays.
_GATE_T = np.arange(2, 14, dtype=np.float64)
_GATE_SHIFTS = (0, 2, 4, 6)

# Gate byte layout: dir(bit10) | kind(bits8-9) | T(bits4-7) | shift(bits0-3).
# kind 0 == off; kind 1/2/3 == cubic / first-axis-linear / last-axis-linear.
_GATE_DIR_SHIFT = 10

# Line-end handling for the gate's interp candidates (see ``_interp_axis_at_t``).
# Fixed, not stored: it is a property of this codec's fallback rather than of the
# stream, so encoder and decoder simply share the constant.
#
# The interp predictor has to *sweep* its ``end_mode`` because the extrapolated
# ends are bimodal across fields there (see ``default_interp_end_mode``). Here
# they are not, and the reason is the chunking: ``W`` is the chunk-local recon,
# so a line end is a chunk *seam* -- where continuing the field's slope is right
# -- far more often than it is a domain boundary. Measured, tune=fast, stream
# bytes, v6.1-d64-1agg:
#
#   field / rel eb              legacy    QUAD   QUAD|EXTRAP
#   s3d 160^3        1e-2        13859   11959         11204   -19%
#   s3d 160^3        1e-3        54292   46086         30346   -44%
#   CLOUDf48 96x160^2 1e-2      159945  159042        159308   -0.4%
#   CLOUDf48 96x160^2 1e-3      440009  440157        440105   +0.02%
#   rti 32^4         1e-2       246898  247714        247106   +0.08%
#   rti 32^4         1e-3       603216  603056        602140   -0.18%
#
# No field prefers the legacy copy, and the two that look flat are fields where
# the gate rarely fires at all -- so unlike the interp side there is nothing to
# earn and the strongest mode is the default.
#
# It is not free, but it is much cheaper than it looks: the end rules apply to the
# ~3% of points that are line ends, so ``_end_plan`` compacts them once per cached
# chunk plan and the device evaluates them on that subset instead of masking
# full-length tensors. This needs no ``nonzero`` (which would sync the launch-bound
# critical path) because *which* points are ends is pure geometry of the chunk
# shape and sub-stage. All three candidates, one 64^3 stride-1 stage, V100, min of
# 7 interleaved rounds, outputs ``torch.equal`` between the two implementations:
#
#   end mode   full-length masks   _end_plan subset
#   legacy               3.61 ms      2.96 ms  -18%
#   QUAD                 4.45         3.83     -14%
#   EXTRAP               4.89         3.60     -26%
#   both (deployed)      5.75         4.46     -22%
#
# EXTRAP gains most because the two *linear* candidates need +-3s only at the ends
# (the cubic candidate already holds those samples for its own stencil, and
# END_QUAD is pure arithmetic on them). Legacy gains too: the full-length +-s
# validity masks and the final selecting ``where``s are gone entirely.
# End to end the win is inside run-to-run noise on s3d (encode is dominated by the
# model forward, and the same config varies 33-66s on a shared box) -- what this
# buys is that the strongest end mode is no longer something a field with a rarely
# firing gate pays much for.
_GATE_END_MODE = END_QUAD | END_EXTRAP

# Adaptive finest-level gate. The finest level (stride-1 sub-stages) carries the
# large majority of the bits, and on smooth fields interpolation beats the model
# there almost everywhere. Spending one swept gate word per finest sub-stage is
# accurate but the finest level has 2*ndim - 1 sub-stages per chunk. gate_fine
# instead chooses a *single* descriptor per chunk (predictor, direction, T,
# shift) from the chunk's first finest sub-stage and reuses it across the rest,
# storing one word per chunk. The threshold is still data-chosen (adaptive) --
# unlike a hardcoded rule -- but the per-chunk metadata drops accordingly. The
# error bound is enforced by quantization regardless of predictor, so this only
# ever trades rate, never correctness; disable per stream with gate_fine=False.


def _pack_gates(gates: list[int]) -> str:
    """Pack the per-chunk-stage gate words into a compact base64 blob.

    Each word fits in a uint16 (see the layout above); the array is zstd-packed
    before base64 so the long runs of repeated / zero words cost almost nothing.
    The JSON header is not otherwise compressed, so this is far smaller than a
    JSON int list once there are many chunks."""
    raw = np.asarray(gates, dtype=np.uint16).tobytes()
    comp = zstandard.ZstdCompressor(level=19).compress(raw)
    return base64.b64encode(comp).decode("ascii")


def _unpack_gates(blob: str) -> list[int]:
    raw = zstandard.ZstdDecompressor().decompress(base64.b64decode(blob))
    return np.frombuffer(raw, dtype=np.uint16).tolist()


# --- device inner-loop primitives -----------------------------------------
# The chunked encode/decode inner loop runs entirely on the GPU: recon stays
# resident on the device and quantize / dequantize / gate all execute in torch,
# so the per-stage critical path never syncs to host. Only the rANS pack/unpack
# (constriction, host-only) crosses over -- and it does not feed recon, so on
# encode it is deferred off the critical path. Encoder and decoder call the
# identical functions, so their reconstructions match bit for bit even where a
# torch transcendental (exp2/log2) differs from numpy by a ULP; correctness is
# anchored on enc/dec agreement, not on matching the old numpy stream.


def _end_plan(torch, dev, coords_np, axis, s, shape):
    """Static geometry of one axis's line ends, or ``None`` if it has none.

    A point whose ``+-s`` neighbours are both in bounds is interpolated by the
    ordinary stencil; the rest -- the line ends, ~2.5% of a 160^3 chunk's points --
    take a one-sided rule that reads only the two samples *behind* them. Which
    points those are, which side they lean to, and where their behind-samples live
    depend on the chunk shape and sub-stage alone, so they are resolved on the host
    once per cached chunk plan and the device just applies them to a compacted
    subset. Returns ``(idx, only_left, far_idx, far_valid)``: positions within the
    stage, side, the gather index of the ``-+3s`` sample, and its validity.
    """
    c, L = coords_np[axis], shape[axis]
    vm1, vp1 = c - s >= 0, c + s < L
    idx = np.flatnonzero(~(vm1 & vp1))
    if idx.size == 0:
        return None
    only_left = (vm1 & ~vp1)[idx]
    cfar = c[idx] + np.where(only_left, -3 * s, 3 * s)
    far = list(cc[idx] for cc in coords_np)
    far[axis] = np.clip(cfar, 0, L - 1)

    def to_dev(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

    return (
        to_dev(idx.astype(np.int64)),
        to_dev(only_left),
        tuple(to_dev(f.astype(np.int64)) for f in far),
        to_dev((cfar >= 0) & (cfar < L)),
    )


def _interp_axis_at_t(torch, W, coords, axis, s, shape, cubic, end_mode, ends):
    """Linear or cubic interpolation on device.

    ``W`` is ``(C, *S)`` float64 and ``coords`` is a tuple of ``(M,)`` long
    tensors. ``end_mode`` is ``predictor.END_QUAD | END_EXTRAP`` and must match
    the numpy ``_interp_axis_at`` branch for branch: here ``W`` is the *chunk-local*
    reconstruction, so a line end is hit at every chunk face rather than only at
    the domain face, and the legacy nearest-neighbour copy is correspondingly
    more expensive than it is for the (unchunked) numpy interp predictor.

    Line ends are handled on the ``_end_plan`` subset rather than by masking
    full-length tensors, which is what keeps ``END_EXTRAP`` from costing the
    *linear* candidates a pair of full ``+-3s`` gathers they use nowhere else.
    """

    def at(off):  # neighbour values, edge-clamped (validity is handled per branch)
        idx = list(coords)
        idx[axis] = (coords[axis] + off).clamp(0, shape[axis] - 1)
        return W[(slice(None), *idx)]

    Lm1, Lp1 = at(-s), at(+s)
    out = 0.5 * (Lm1 + Lp1)
    if cubic:
        ca = coords[axis]
        vm3, vp3 = ca - 3 * s >= 0, ca + 3 * s < shape[axis]
        Lm3, Lp3 = at(-3 * s), at(+3 * s)
        cub = (-Lm3 + 9 * Lm1 + 9 * Lp1 - Lp3) / 16.0
        one_far = out
        if end_mode & END_QUAD:
            one_far = torch.where(
                (vp3 & ~vm3).unsqueeze(0),
                (3 * Lm1 + 6 * Lp1 - Lp3) / 8.0,  # interp_quad_1 (leading end)
                torch.where(
                    (vm3 & ~vp3).unsqueeze(0),
                    (-Lm3 + 6 * Lm1 + 3 * Lp1) / 8.0,  # interp_quad_2 (trailing)
                    out,  # both far neighbours missing: nothing better than linear
                ),
            )
        out = torch.where((vm3 & vp3).unsqueeze(0), cub, one_far)
    if ends is None:  # this axis is bracketed everywhere: no ends to overwrite
        return out
    eidx, only_left, far_idx, far_valid = ends
    side = only_left.unsqueeze(0)
    near = torch.where(side, Lm1.index_select(1, eidx), Lp1.index_select(1, eidx))
    if end_mode & END_EXTRAP:
        # interp_linear1: continue the last known slope, falling back to the copy
        # when even the second sample behind is off the line.
        far = W[(slice(None), *far_idx)]
        near = torch.where(far_valid.unsqueeze(0), 1.5 * near - 0.5 * far, near)
    return out.index_copy_(1, eidx, near)


def _axis_preds_t(torch, W, coords_t, stride, axes, cubic, end_mode, ends):
    """Per-axis interpolated value for every candidate axis, ``{a: (C, n) f64}``.

    ``axes`` is the sub-stage's candidate axis set (see ``levels.stage_plan``):
    a single axis for a weight-1 sub-stage, or the full range for a fused
    weight >= 2 sub-stage whose points' actual odd axes vary per point.
    """
    shape = tuple(W.shape[1:])
    return {
        a: _interp_axis_at_t(
            torch, W, coords_t, a, stride, shape, cubic, end_mode, ends[a]
        )
        for a in axes
    }


def _interp_axis_pair_t(torch, W, coords, axis, s, shape, end_mode, ends):
    """Return cubic and linear predictions while sharing their neighbour reads."""

    def at(off):
        idx = list(coords)
        idx[axis] = (coords[axis] + off).clamp(0, shape[axis] - 1)
        return W[(slice(None), *idx)]

    Lm1, Lp1 = at(-s), at(+s)
    linear = 0.5 * (Lm1 + Lp1)
    ca = coords[axis]
    vm3, vp3 = ca - 3 * s >= 0, ca + 3 * s < shape[axis]
    Lm3, Lp3 = at(-3 * s), at(+3 * s)
    cubic = (-Lm3 + 9 * Lm1 + 9 * Lp1 - Lp3) / 16.0
    one_far = linear
    if end_mode & END_QUAD:
        one_far = torch.where(
            (vp3 & ~vm3).unsqueeze(0),
            (3 * Lm1 + 6 * Lp1 - Lp3) / 8.0,
            torch.where(
                (vm3 & ~vp3).unsqueeze(0),
                (-Lm3 + 6 * Lm1 + 3 * Lp1) / 8.0,
                linear,
            ),
        )
    cubic = torch.where((vm3 & vp3).unsqueeze(0), cubic, one_far)
    if ends is None:
        return cubic, linear
    eidx, only_left, far_idx, far_valid = ends
    side = only_left.unsqueeze(0)
    near = torch.where(side, Lm1.index_select(1, eidx), Lp1.index_select(1, eidx))
    if end_mode & END_EXTRAP:
        far = W[(slice(None), *far_idx)]
        near = torch.where(far_valid.unsqueeze(0), 1.5 * near - 0.5 * far, near)
    return cubic.index_copy_(1, eidx, near), linear.index_copy_(1, eidx, near)


def _axis_pred_pairs_t(torch, W, coords_t, stride, axes, end_mode, ends):
    """Cubic and linear per-axis predictions with shared gathers."""
    shape = tuple(W.shape[1:])
    pairs = {
        a: _interp_axis_pair_t(
            torch, W, coords_t, a, stride, shape, end_mode, ends[a]
        )
        for a in axes
    }
    return (
        {a: pair[0] for a, pair in pairs.items()},
        {a: pair[1] for a, pair in pairs.items()},
    )


def _combine_axis_preds(torch, preds, coords_t, stride, axes, center):
    """Reduce per-axis predictions to one per point, following ``center``.

    Each point's own odd-axis membership is re-derived here from its coordinates
    rather than assumed uniform across the stage. Combined with a plain
    sequential ``where`` walk (never ``argmax``/``max`` over a tie) so ties
    among several True axes resolve the same way every call -- required for
    the encoder and decoder to land on bit-identical gate candidates."""

    def is_odd(a):
        return (coords_t[a] // stride % 2 == 1).unsqueeze(0)

    if center == 0 or len(axes) == 1:
        total = count = None
        for a in axes:
            m = is_odd(a).double()
            v = preds[a] * m
            total = v if total is None else total + v
            count = m if count is None else count + m
        ip = total / count
    else:
        ip = None
        order = reversed(axes) if center == 1 else axes  # first-/last-wins
        for a in order:
            ip = preds[a] if ip is None else torch.where(is_odd(a), preds[a], ip)
    return ip.to(torch.float32)  # (C, n)


def _gate_interps(torch, recon_t, sls, interp_dev, s, center, end_mode):
    """The three gate interp candidates for one stage: (cubic, first-axis linear,
    last-axis linear). Shared by encoder and decoder so both see identical
    fallbacks; the caller selects among them by the stored/derived gate kind.

    The two linear candidates differ only in how per-axis predictions are
    *combined* (first-wins vs last-wins), so the per-axis interpolation is done
    once and reduced twice; likewise the float64 view of the chunk is taken once
    for all three. Same arithmetic per candidate, a third fewer gathers."""
    coords_t, st_i, ax_i, ends_i = interp_dev[s]
    W = recon_t[(slice(None), *sls)].double()
    cub, lin = _axis_pred_pairs_t(
        torch, W, coords_t, st_i, ax_i, end_mode, ends_i
    )
    return (
        _combine_axis_preds(torch, cub, coords_t, st_i, ax_i, center),
        _combine_axis_preds(torch, lin, coords_t, st_i, ax_i, 1),
        _combine_axis_preds(torch, lin, coords_t, st_i, ax_i, 2),
    )


def _stored_gate_interp(torch, recon_t, sls, interp_dev, s, center, end_mode, kind):
    """Compute only the interpolation selected by a stored decoder gate.

    The encoder must evaluate all three candidates to choose a gate.  Decode
    already knows that choice from the header, so evaluating the two discarded
    candidates wastes gathers and arithmetic.  This follows the corresponding
    branch of :func:`_gate_interps` exactly.
    """
    coords_t, st_i, ax_i, ends_i = interp_dev[s]
    W = recon_t[(slice(None), *sls)].double()
    cubic = kind == 1
    preds = _axis_preds_t(
        torch, W, coords_t, st_i, ax_i, cubic, end_mode, ends_i
    )
    combine = center if cubic else kind - 1
    return _combine_axis_preds(torch, preds, coords_t, st_i, ax_i, combine)


def _laplace_bits_t(torch, absr, b, eb, radius):
    """rANS-aligned rate estimate used to rank gate settings.

    Scales are snapped to the same 64-level grid as ``scale_to_level``. Regular
    symbols are capped at the rANS table precision (24 bits), while outliers pay
    that marker cost plus the raw float32 stored by ``pack_stage``.

    ``absr`` and ``b`` may carry a leading candidate axis, so ``_gate_select_t``
    can score every gate setting in one pass. Dropping to float32 was measured
    and is NOT worth it (<1% either way): this is launch-bound, not ALU-bound,
    so the win comes from batching the candidates, not from cheaper arithmetic.
    """
    b = _gate_quantized_scales(torch, b, eb)
    k = torch.round(absr.double().abs() / (2 * eb))
    p = torch.where(
        k == 0,
        -torch.expm1(-eb / b),
        0.5 * torch.exp(-((2 * k - 1) * eb / b)) * -torch.expm1(-2 * eb / b),
    )
    regular = (-torch.log2(p)).clamp_max(24.0)
    return torch.where(k >= radius, torch.full_like(regular, 56.0), regular)


_GATE_CONST_CACHE: dict = {}
_SCALE_THRESHOLD_CACHE: dict = {}
_GATE_SCALE_CACHE: dict = {}


def _gate_consts(torch, dev):
    """Device copies of the gate's fixed tables: ``(T, shifts, 2**T, arange(nb))``.

    These depend on nothing but the module constants, yet a per-stage
    ``torch.tensor(..., device=cuda)`` is a *pageable* host->device copy that
    blocks until the stream drains -- measured at ~2.4ms a call behind queued
    work, which for two calls per stage was 11% of a 64^4 encode. Building them
    once per device removes the copy from the critical path entirely.
    """
    c = _GATE_CONST_CACHE.get(dev)
    if c is None:
        GTf = torch.tensor(_GATE_T, dtype=torch.float64, device=dev)
        c = (
            GTf.to(torch.int64),
            torch.tensor(_GATE_SHIFTS, dtype=torch.int64, device=dev),
            torch.exp2(GTf),
            torch.arange(_GATE_T.size + 1, device=dev),
        )
        _GATE_CONST_CACHE[dev] = c
    return c


def _scale_level_thresholds(eb: float) -> np.ndarray:
    """Float32 boundaries exactly equivalent to :func:`scale_to_level`.

    Positive float32 bit patterns are ordered like their unsigned integers.
    Binary-search the first pattern assigned to each level by the authoritative
    NumPy implementation; device bucketization against those patterns then
    reproduces all 64 levels using comparisons only (no backend-dependent log).
    """
    key = ("host", float(eb))
    hit = _SCALE_THRESHOLD_CACHE.get(key)
    if hit is not None:
        return hit
    targets = np.arange(1, 64, dtype=np.uint8)
    lo = np.zeros(63, dtype=np.uint32)
    hi_bits = np.asarray(np.float32(eb * SCALE_HI_MULT)).view(np.uint32)
    hi = np.full(63, hi_bits, dtype=np.uint32)
    while np.any(lo < hi):
        mid = lo + (hi - lo) // 2
        values = mid.view(np.float32)
        upper = scale_to_level(values, eb) >= targets
        hi = np.where(upper, mid, hi).astype(np.uint32)
        lo = np.where(upper, lo, mid + 1).astype(np.uint32)
    out = np.ascontiguousarray(lo.view(np.float32))
    _SCALE_THRESHOLD_CACHE[key] = out
    return out


def _scale_to_level_t(torch, scale, eb: float):
    """Device scale quantization bit-identical to NumPy ``scale_to_level``."""
    dev_key = (str(scale.device), float(eb))
    thresholds = _SCALE_THRESHOLD_CACHE.get(dev_key)
    if thresholds is None:
        thresholds = torch.from_numpy(_scale_level_thresholds(eb)).to(scale.device)
        _SCALE_THRESHOLD_CACHE[dev_key] = thresholds
    return torch.bucketize(scale.float(), thresholds, right=True).to(torch.uint8)


def _gate_quantized_scales(torch, scale, eb: float):
    """Gate scale snapping equivalent to its former double log/round/exp path."""
    key = (str(scale.device), float(eb))
    cached = _GATE_SCALE_CACHE.get(key)
    if cached is None:
        _prepare_gate_scale_tables(torch, scale.device, [eb])
        cached = _GATE_SCALE_CACHE[key]
    thresholds, grid = cached
    level = torch.bucketize(scale.float(), thresholds, right=True)
    return grid[level]


def _gate_scale_thresholds(eb: float) -> tuple[np.ndarray, float, float]:
    """Host boundaries for the gate's double log/round scale assignment."""
    lo_f = float(eb) / SCALE_LO_DIV
    hi_f = float(eb) * SCALE_HI_MULT
    log_lo = np.log(lo_f)
    log_step = (np.log(hi_f) - log_lo) / 63.0
    targets = np.arange(1, 64)
    lo_bits = np.zeros(63, dtype=np.uint32)
    hi_bit = np.asarray(np.float32(hi_f)).view(np.uint32)
    hi_bits = np.full(63, hi_bit, dtype=np.uint32)
    while np.any(lo_bits < hi_bits):
        mid = lo_bits + (hi_bits - lo_bits) // 2
        values = mid.view(np.float32).astype(np.float64)
        level = np.rint(
            (np.log(np.clip(values, lo_f, hi_f)) - log_lo) / log_step
        )
        upper = level >= targets
        hi_bits = np.where(upper, mid, hi_bits).astype(np.uint32)
        lo_bits = np.where(upper, lo_bits, mid + 1).astype(np.uint32)
    return np.ascontiguousarray(lo_bits.view(np.float32)), log_lo, log_step


def _prepare_gate_scale_tables(torch, device, ebs) -> None:
    """Batch all missing gate threshold/grid tables into one device upload."""
    missing = []
    seen = set()
    for eb in ebs:
        eb = float(eb)
        key = (str(device), eb)
        if key not in _GATE_SCALE_CACHE and eb not in seen:
            missing.append(eb)
            seen.add(eb)
    if not missing:
        return
    host = [_gate_scale_thresholds(eb) for eb in missing]
    thresholds = torch.from_numpy(np.stack([x[0] for x in host])).to(device)
    log_lo = torch.tensor([x[1] for x in host], dtype=torch.float64, device=device)
    log_step = torch.tensor([x[2] for x in host], dtype=torch.float64, device=device)
    levels = torch.arange(64, dtype=torch.float64, device=device)
    grids = torch.exp(log_lo[:, None] + log_step[:, None] * levels)
    for i, eb in enumerate(missing):
        _GATE_SCALE_CACHE[(str(device), eb)] = (thresholds[i], grids[i])


def _gate_select_t(torch, r_g, r_is, b, eb, radius=1 << 15):
    """Choose the best (classical predictor, direction, T, shift) for one stage.

    Returns device scalar ``(kind, dir, T, shift)`` tensors, with kind 0 meaning
    off and positive kinds indexing ``r_is`` from one; ``dir`` is 0 for a low-side
    split (fall back where b < eb*2^T) and 1 for a high-side split (fall back where
    b >= eb*2^T). The selection is branch-free so it issues no host sync.
    """
    dev = b.device
    GTi, SHi, GT2, buckets = _gate_consts(torch, dev)
    nb = _GATE_T.size + 1
    bucket = torch.bucketize(b.double(), eb * GT2)
    # one-hot bucket membership: per-bucket sums via a plain (deterministic)
    # reduction, not bincount/scatter_add -- those use atomic float adds on CUDA
    # and would make the gate choice, and thus the stream, non-deterministic.
    onehot = (bucket.unsqueeze(-1) == buckets).double()  # (n, nb)

    # Every candidate rate estimate (the ungated baseline, plus each classical
    # predictor at each shift) is evaluated in ONE batched pass. Scored one at
    # a time they issued ~13x the kernel launches for identical arithmetic, and
    # this encode is launch-bound. The per-bucket sum is a matmul against the
    # one-hot rather than an (n, nb) product per candidate, so the big
    # transient never materialises either.
    nsh = len(_GATE_SHIFTS)
    A = torch.stack([r_g] + [r for r in r_is for _ in _GATE_SHIFTS])  # (R, C, n)
    Bb = torch.stack(
        [b] + [b * 2.0**-sh for _ in r_is for sh in _GATE_SHIFTS]
    ).unsqueeze(1)  # (R, 1, n)
    W = _laplace_bits_t(torch, A, Bb, eb, radius).sum(1)  # (R, n)
    cums = torch.cumsum(W @ onehot, -1)  # (R, nb)

    base = cums[0]
    tot = base[-1]
    # For a boundary at bucket k (split index k in [0, nb-2]):
    #   low-side  cost = interp(buckets<=k)  + gnn(buckets>k)  = ci[k] + (tot-base[k])
    #   high-side cost = gnn(buckets<=k)     + interp(buckets>k) = base[k] + (ci[-1]-ci[k])
    ci = cums[1:]  # (n_predictor * n_shift, nb), same order as before
    low = tot - base[:-1] + ci[:, :-1]
    high = base[:-1] + (ci[:, -1:] - ci[:, :-1])
    costs = torch.cat([low, high])  # (2 * n_predictor * n_shift, nb-1)
    mv, fi = costs.reshape(-1).min(0)
    fired = mv < tot
    ncol = nb - 1
    n_side = len(r_is) * nsh
    z = torch.zeros((), dtype=torch.int64, device=dev)
    row = fi // ncol  # 0..2*n_side-1; first n_side rows are low-side
    gate_dir = torch.where(fired, (row >= n_side).to(torch.int64), z)
    side_row = row % n_side
    gate_kind = torch.where(fired, side_row // nsh + 1, z)
    gate_t = torch.where(fired, GTi[fi % ncol], z)
    gate_sh = torch.where(fired, SHi[side_row % nsh], z)
    return gate_kind, gate_dir, gate_t, gate_sh


def _gate_apply_t(torch, pred_bi, scale_bi, ip, eb, gate_t, gate_sh, gate_dir):
    """Device twin of ``_gate_apply``, called unconditionally on both sides: a
    ``gate_t`` of 0 is a no-op, so encoder and decoder stay symmetric without a
    host-sync branch. ``gate_dir`` picks the side of the eb*2^T threshold on which
    the fallback fires (0 = low, b < thr; 1 = high, b >= thr). (pred, coded scale).
    """
    if isinstance(gate_t, int):
        if gate_t <= 0:
            return pred_bi.unsqueeze(0).to(torch.float32), scale_bi.to(torch.float32)
        below = scale_bi < eb * (2.0**gate_t)
        m = ~below if gate_dir > 0 else below
        p = torch.where(m.unsqueeze(0), ip, pred_bi.unsqueeze(0))
        sc = torch.where(m, scale_bi * (2.0 ** -gate_sh), scale_bi)
        return p.to(torch.float32), sc.to(torch.float32)
    active = gate_t > 0
    below = scale_bi < eb * torch.exp2(gate_t.double())
    m = active & torch.where(gate_dir > 0, ~below, below)
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
    recon_candidate = (
        pred.double() + w * (codes - radius).double()
    ).to(torch.float32)
    if round_output is True:
        recon_chk = torch.round(recon_candidate)
    elif round_output:
        span, offset = round_output
        recon_chk = (
            (torch.round(recon_candidate.double() * span + offset) - offset) / span
        ).to(torch.float32)
    else:
        recon_chk = recon_candidate
    ok = in_range & ((x - recon_chk).abs() <= float(np.float32(eb)))
    codes = torch.where(ok, codes, z)
    is_out = codes == 0
    # A surviving regular code is unchanged from the one used to construct
    # recon_candidate; a rejected/out-of-range code reconstructs as its exact
    # outlier value. Avoid repeating the full float64 dequantization here.
    recon = torch.where(is_out, x, recon_candidate)
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
    gate_fine: bool = True,
) -> tuple[bytes, list[int] | None, list[int] | None]:
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
    geom_bar = _progress_bar(
        "encode geometry", _geometry_stages(len(shape), predictor.levels), unit="stage"
    )
    if predictor.geometry_cached(shape, edges):
        parts = [
            _code_anchor_stage(values, recon, axes, ebs[0], radius, round_output)
        ]
        anchor_bar.update(1)
        anchor_bar.close()
        predictor.begin(shape, edges, channels=c, geometry_progress=geom_bar.update)
    else:
        # Anchor entropy coding and schedule construction are independent. The
        # cold rank-4 path spends substantial time in both, mostly inside NumPy,
        # constriction and device copies that release the GIL, so overlap them.
        with ThreadPoolExecutor(max_workers=1) as pool:
            geometry = pool.submit(
                predictor.begin,
                shape,
                edges,
                channels=c,
                geometry_progress=geom_bar.update,
            )
            parts = [
                _code_anchor_stage(values, recon, axes, ebs[0], radius, round_output)
            ]
            anchor_bar.update(1)
            anchor_bar.close()
            geometry.result()
    geom_bar.close()
    # hand recon to the GPU for the wave loop; it never returns to host on encode
    recon_t = torch.from_numpy(recon).to(dev)
    del recon  # host copy is dead once uploaded; keeping it costs a full field
    recon_flat = recon_t.reshape(c, -1)
    waves = [list(range(predictor.n_chunks))]  # raster order (see docstring)
    _log(
        f"encode: anchors done, {predictor.n_chunks} chunks, "
        f"{predictor.n_chunks} model passes"
    )
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    if gate:
        _prepare_gate_scale_tables(torch, dev, ebs)
    index_cache: OrderedDict = OrderedDict()  # bounded; see _cached_chunk_plan
    full_strides = np.cumprod((1,) + shape[:0:-1])[::-1].astype(np.int64)
    gates_t: list = [] if gate else None  # per stage-chunk gate byte, device scalars
    # Finest level = stride-1 sub-stages; with gate_fine one descriptor per chunk
    # covers them all (chosen at the chunk's first finest sub-stage). is_finest
    # depends only on the schedule shape, so compute once for the whole encode.
    fine_gates_t: list = [] if (gate and gate_fine) else None  # one word per chunk
    is_finest = [st == 1 for st in stage_strides(len(shape), predictor.levels, stride)]
    bar = _progress_bar("encode", predictor.n_chunks)
    pack_pool = (
        ThreadPoolExecutor(max_workers=1) if predictor.n_chunks > 1 else None
    )
    pack_futures = deque()

    def pack_wave(items):
        packed = []
        for item in items:
            if len(item) == 2:  # empty stage
                packed.append(
                    pack_stage(
                        np.zeros(0, np.uint32),
                        np.zeros(0, np.float32),
                        rans_levels=np.zeros(0, np.uint8),
                        rans_tables=item[1],
                    )
                )
                continue
            codes_c, out_c, levels_c, tables = item
            packed.append(
                pack_stage(
                    codes_c.numpy().astype(np.uint32),
                    out_c.numpy().astype(np.float32),
                    rans_levels=levels_c.numpy().reshape(-1),
                    rans_tables=tables,
                )
            )
        return packed

    for group in waves:
        for ci in group:
            ids = [ci]
            cshape = tuple(sl.stop - sl.start for sl in predictor.chunk_slices(ids[0]))
            low_ax = predictor.low_axes(ids[0])
            (
                full_counts, counts, pos_dev, recon_off_dev, interp_dev,
                pred_idx_dev, center, end_mode,
            ) = _cached_chunk_plan(
                index_cache, torch, dev, cshape, shape, predictor.levels,
                stride, block, low_ax,
            )
            fine_desc = None  # (kind, dir, T, shift) chosen once per chunk (gate_fine)
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
                    if gate and gate_fine and is_finest[s]:
                        # Adaptive finest gate: pick one descriptor per chunk at
                        # its first finest sub-stage, then reuse it (one stored
                        # word per chunk, see the "Adaptive finest-level gate" note).
                        ip_cubic, ip_linear, ip_linear_last = _gate_interps(
                            torch, recon_t, sls, interp_dev, s, center, end_mode
                        )
                        if fine_desc is None:
                            gk, gd, gt, gs = _gate_select_t(
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
                            fine_desc = (gk, gd, gt, gs)
                            fine_gates_t.append(
                                (gd << _GATE_DIR_SHIFT) + gk * 256 + gt * 16 + gs
                            )
                        else:
                            gk, gd, gt, gs = fine_desc
                        ip = torch.where(
                            gk == 2,
                            ip_linear,
                            torch.where(gk == 3, ip_linear_last, ip_cubic),
                        )
                        p, sc = _gate_apply_t(
                            torch, pred[bi], sc, ip, ebs[s], gt, gs, gd
                        )
                    elif gate:
                        ip_cubic, ip_linear, ip_linear_last = _gate_interps(
                            torch, recon_t, sls, interp_dev, s, center, end_mode
                        )
                        gk, gd, gt, gs = _gate_select_t(
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
                        gates_t.append(
                            (gd << _GATE_DIR_SHIFT) + gk * 256 + gt * 16 + gs
                        )
                        ip = torch.where(
                            gk == 2,
                            ip_linear,
                            torch.where(gk == 3, ip_linear_last, ip_cubic),
                        )
                        p, sc = _gate_apply_t(
                            torch, pred[bi], sc, ip, ebs[s], gt, gs, gd
                        )
                    codes, recon_stage, outliers = _quantize_t(
                        torch, cvals, p, ebs[s], radius, round_output
                    )
                    gpos = recon_off_dev[s] + origin_bases[bi]
                    recon_flat.index_copy_(1, gpos, recon_stage.reshape(c, -1))
                    levels = _scale_to_level_t(torch, sc, ebs[s])
                    wave_pending.append(
                        (
                            codes.to("cpu", non_blocking=True),
                            outliers.to("cpu", non_blocking=True),
                            levels.to("cpu", non_blocking=True),
                            tables,
                        )
                    )
            predictor.finish_wave(recon_t)
            # deferred rANS: the codes streamed off the GPU during the wave; sync
            # once, then pack in stream order (host-only, off the recon path).
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            if pack_pool is None:
                parts.extend(pack_wave(wave_pending))
            else:
                pack_futures.append(pack_pool.submit(pack_wave, wave_pending))
                # Keep at most one CPU wave behind the GPU. This bounds the host
                # buffers while letting wave N entropy-code during wave N+1.
                if len(pack_futures) >= 2:
                    parts.extend(pack_futures.popleft().result())
            peak = _cuda_peak(predictor)
            if peak:
                bar.set_postfix_str(f"peak {peak / 1e9:.2f}GB")
            bar.update(1)
    while pack_futures:
        parts.extend(pack_futures.popleft().result())
    if pack_pool is not None:
        pack_pool.shutdown()
    bar.close()
    gates = None
    if gate:
        gates = [int(x) for x in torch.stack(gates_t).cpu().tolist()] if gates_t else []
    fine_gates = None
    if gate and gate_fine:
        fine_gates = (
            [int(x) for x in torch.stack(fine_gates_t).cpu().tolist()]
            if fine_gates_t
            else []
        )
    return b"".join(parts), gates, fine_gates


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
            ends = {a: _end_plan(torch, dev, coords_np, a, st, cshape) for a in ax}
            interp_dev.append((coords, st, ax, ends))
        else:
            interp_dev.append(None)
    center = default_interp_center(len(cshape))
    end_mode = _GATE_END_MODE
    return (
        full_counts, counts, pos_dev, recon_off_dev, interp_dev, pred_idx_dev,
        center, end_mode,
    )


# Chunk plans are keyed by (chunk shape, low_axes), and in grow mode each axis is
# independently first / middle / last, so the key space is 3 ** ndim -- 81 at rank
# 4. Caching all of them defeats the point of chunking: device memory climbs with
# every newly seen combination instead of staying bounded, reaching ~4.7 GB of
# plans by the last chunk of a 128^4 encode.
#
# The budget is in *bytes*, not entries, because a plan is chunk-sized: a fixed
# entry count would be far too much memory for large chunks and needless eviction
# for small ones. Raster order varies the last axis fastest, so even a modest
# budget retains the immediately reusable shapes. Large one-off plans are not
# cached: keeping one alive while constructing its successor raises the peak
# without buying a cache hit.
_PLAN_CACHE_MIB = float(os.environ.get("LATEC_PLAN_CACHE_MIB", 256))


def _cached_chunk_plan(cache, torch, dev, cshape, shape, levels, stride, block, low_ax):
    """``_chunk_device_plan`` behind a byte-budgeted LRU, shared by encode/decode.

    Eviction is safe against the in-flight plan: the entry just used is moved to
    the most-recent end before anything is dropped. Oversized plans are returned
    to the caller but not retained after the current wave."""
    key = (cshape, low_ax)
    hit = cache.get(key)
    if hit is not None:
        cache.move_to_end(key)
        return hit[0]
    plan = _chunk_device_plan(torch, dev, cshape, shape, levels, stride, block, low_ax)
    nbytes = _gp._storage_bytes(plan)
    budget = _PLAN_CACHE_MIB * 2**20
    total = sum(nbytes for _, nbytes in cache.values())
    while cache and total + nbytes > budget:
        _, (_, dropped) = cache.popitem(last=False)
        total -= dropped
    if nbytes <= budget:
        cache[key] = (plan, nbytes)
    return plan


def _decompress_chunked(
    payload: bytes,
    shape: tuple[int, ...],
    ebs: list[float],
    radius: int,
    predictor: ChunkedGNNPredictor,
    edges: tuple[int, ...],
    gates: list[int] | None = None,
    fine_gates: list[int] | None = None,
) -> np.ndarray:
    c = 1
    stride, block = predictor.anchor_stride, predictor.anchor_block
    recon = np.zeros((c, *shape), np.float32)
    axes = _anchor_axes(shape, stride, block)
    _log(f"decode: shape={shape} edges={edges} decoding anchors...")
    anchor_bar = _progress_bar("decode anchors", 1, unit="stage")
    geom_bar = _progress_bar(
        "decode geometry", _geometry_stages(len(shape), predictor.levels), unit="stage"
    )
    if predictor.geometry_cached(shape, edges):
        off = _decode_anchor_stage(payload, 0, recon, axes, ebs[0], radius)
        anchor_bar.update(1)
        anchor_bar.close()
        predictor.begin(shape, edges, channels=c, geometry_progress=geom_bar.update)
    else:
        with ThreadPoolExecutor(max_workers=1) as pool:
            geometry = pool.submit(
                predictor.begin,
                shape,
                edges,
                channels=c,
                geometry_progress=geom_bar.update,
            )
            off = _decode_anchor_stage(payload, 0, recon, axes, ebs[0], radius)
            anchor_bar.update(1)
            anchor_bar.close()
            geometry.result()
    geom_bar.close()
    torch = predictor._torch
    dev = predictor.device
    recon_t = torch.from_numpy(recon).to(dev)  # device-resident, same as encode
    del recon  # host copy is dead once uploaded; keeping it costs a full field
    waves = [list(range(predictor.n_chunks))]  # raster order, mirrors encode
    _log(f"decode: anchors done, {predictor.n_chunks} chunks/model passes")
    stage_tables = [build_laplace_tables(e, radius) for e in ebs]
    index_cache: OrderedDict = OrderedDict()  # bounded; see _cached_chunk_plan
    full_strides = np.cumprod((1,) + shape[:0:-1])[::-1].astype(np.int64)
    recon_flat = recon_t.reshape(c, -1)
    # Mirror the encoder's adaptive finest gate: one descriptor per chunk covering
    # all its stride-1 sub-stages, read at the chunk's first finest sub-stage.
    is_finest = [st == 1 for st in stage_strides(len(shape), predictor.levels, stride)]
    gi = 0
    fgi = 0
    bar = _progress_bar("decode", predictor.n_chunks)
    for group in waves:
        for ci in group:
            ids = [ci]
            cshape = tuple(sl.stop - sl.start for sl in predictor.chunk_slices(ids[0]))
            low_ax = predictor.low_axes(ids[0])
            (
                full_counts, counts, _, recon_off_dev, interp_dev,
                pred_idx_dev, center, end_mode,
            ) = _cached_chunk_plan(
                index_cache, torch, dev, cshape, shape, predictor.levels,
                stride, block, low_ax,
            )
            fine_desc = None  # (kind, dir, T, shift) read once per chunk (gate_fine)
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
                    if fine_gates is not None and is_finest[s]:
                        # Adaptive finest gate: read one descriptor per chunk at
                        # its first finest sub-stage, then reuse it.
                        if fine_desc is None:
                            g = int(fine_gates[fgi])
                            fgi += 1
                            fine_desc = (
                                (g >> 8) & 3,
                                (g >> 4) & 15,
                                g & 15,
                                (g >> _GATE_DIR_SHIFT) & 1,
                            )
                        gk, gt, gs, gd = fine_desc
                        if gt > 0:
                            ip = _stored_gate_interp(
                                torch,
                                recon_t,
                                sls,
                                interp_dev,
                                s,
                                center,
                                end_mode,
                                gk,
                            )
                            p, sc = _gate_apply_t(
                                torch, pred[bi], sc, ip, ebs[s], gt, gs, gd
                            )
                    elif gates is not None:
                        g = int(gates[gi])
                        gi += 1
                        gk = (g >> 8) & 3
                        gt = (g >> 4) & 15
                        if gt > 0:
                            ip = _stored_gate_interp(
                                torch,
                                recon_t,
                                sls,
                                interp_dev,
                                s,
                                center,
                                end_mode,
                                gk,
                            )
                            p, sc = _gate_apply_t(
                                torch,
                                pred[bi],
                                sc,
                                ip,
                                ebs[s],
                                gt,
                                g & 15,
                                (g >> _GATE_DIR_SHIFT) & 1,
                            )
                    levels64 = _scale_to_level_t(torch, sc, ebs[s]).cpu().numpy()
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
        raise ValueError("trailing bytes in LATEC GNN payload")
    if gates is not None and gi != len(gates):
        raise ValueError("gate list length does not match the stream")
    out = recon_t[0].cpu().numpy()
    predictor.clear_runtime_cache()
    return out


class GNNCompressorCodec:
    """Usable Python codec for GNN-backed LATEC tensor compression.

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
        device: str | None = None,  # None -> cuda if available, else cpu
        *,
        strict_checkpoint: bool = True,  # decode-side: reject foreign-checkpoint streams
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"GNN checkpoint not found: {self.checkpoint_path}")
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.strict_checkpoint = bool(strict_checkpoint)
        self.checkpoint_hash = self._checkpoint_hash()
        # Keep the loaded model and base geometry across an encode/decode pair,
        # then release tensor-shaped runtime geometry after decode. Compact
        # frames are not retained by default because they dwarf the base data.
        self._chunked_predictors: OrderedDict[
            tuple[int, bool, bool], ChunkedGNNPredictor
        ] = OrderedDict()

    @staticmethod
    def _chunk_edges(
        shape: tuple[int, ...],
        anchor_stride: int,
        cs: int | tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        """Chunk edges for this shape. ``None`` = auto: one chunk covering the
        whole tensor when it is small, otherwise a grid of auto-sized chunks."""
        if cs is None:
            if int(np.prod(shape)) <= _AUTO_CHUNK_THRESHOLD:
                # Single chunk over the whole shape (edges rounded up to the
                # anchor stride): one GPU pass, and the gate still applies.
                return tuple(
                    -(-n // anchor_stride) * anchor_stride for n in shape
                )
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

    def compress(
        self,
        x: Any,
        error_bound: float = 1e-2,  # relative to the tensor's (max - min); see class docstring
        *,
        levels: int | str = "auto",
        radius: int = 1 << 15,
        zstd_level: int = 9,
        eb_ratio: float | None = None,  # coarsest-level factor; None=auto (fast/sweep)
        tune: str = "fast",
        chunk_size: int | tuple[int, ...] | None = None,
        fp16: bool = True,
        compile: bool | str = "auto",
        gate: bool = True,
        gate_fine: bool = False,
    ) -> bytes:
        """Compress a numpy array or torch tensor of any rank into bytes.

        ``levels``: an explicit int fixes the dyadic schedule depth; "auto" (the
        default) picks it from the input shape (see _auto_levels), since
        anchor_stride = 2**levels is capped by the smallest axis. Decode always
        reads the resolved levels back from the stream.

        ``eb_ratio``: coarsest-level error-bound factor (the finest level always
        keeps full eb); None -> auto (``tune="fast"``: _GNN_EB_COARSE_FACTOR,
        ``tune="size"``: sweep). Depth-normalised per resolved levels so it is
        rank-invariant.

        ``chunk_size``: None = auto (a single chunk covering small inputs,
        otherwise the largest near-isotropic chunk within
        _AUTO_CHUNK_THRESHOLD points); an int or per-axis tuple forces those
        edges (multiples of anchor_stride).

        ``fp16``: run the message-pass matmuls in fp16 (cuda only; the readout
        stays fp32). ~2x on the GNN forward, may cost a little ratio at small eb.
        Stored in meta so decode uses the same float path.

        ``compile``: torch.compile the message-pass embed. "auto" (default)
        decides per chunk count from a benchmark-backed crossover
        (_COMPILE_AUTO_CROSSOVER, currently None -> off: compile never beat eager
        here). True forces it (still gated by _COMPILE_MIN_CHUNKS); False
        disables it. The resolved decision is stored in meta so decode replays
        the same float path. First encode pays a one-off compilation cost.

        ``gate``: scale-gated classical fallback. The encoder rate-selects GNN, cubic/averaged interpolation, or
        first/last-axis linear interpolation per chunk-stage and stores the
        winner in the header. It self-disables wherever no fallback pays.

        ``gate_fine`` (opt-in, default off): at the finest (stride-1) level store
        one adaptively-chosen gate descriptor per chunk instead of one swept word
        per finest sub-stage. The threshold is still data-chosen (picked from the
        chunk's first finest sub-stage and reused across its 2*ndim - 1 finest
        sub-stages), so it keeps per-chunk adaptivity while cutting the finest
        level's metadata accordingly. It reuses one descriptor across sub-stages,
        so its rate is >= the fully per-sub-stage gate (gate_fine=False); since
        packing already makes the per-sub-stage words cheap, that default is
        usually as good or better. gate_fine helps most when finest metadata, not
        payload, dominates. Only active when gate is on; the error bound holds
        regardless (quantization enforces it).
        """
        if tune not in ("fast", "size"):
            raise ValueError("tune must be 'fast' or 'size'")
        if isinstance(levels, str) and levels != "auto":
            raise ValueError("levels must be a positive int or 'auto'")
        if isinstance(compile, str) and compile != "auto":
            raise ValueError("compile must be a bool or 'auto'")
        if not isinstance(levels, str) and int(levels) < 1:
            raise ValueError("levels must be >= 1")
        radius = int(radius)
        gate = bool(gate)
        gate_fine = bool(gate_fine) and gate

        arr = np.asarray(_as_numpy(x))
        if arr.size == 0:
            raise ValueError("cannot compress an empty tensor")
        if arr.dtype.kind not in "biuf":
            raise TypeError(f"unsupported dtype {arr.dtype}; expected numeric data")

        dtype = np.dtype(arr.dtype)
        original_shape = tuple(int(n) for n in arr.shape)
        shape = original_shape if original_shape else (1,)
        # copy=True (the default): we own the buffer, so the normalize below can
        # run in place. astype(copy=False) + out-of-place arithmetic would peak
        # at three full-size arrays on a large field.
        values = arr.reshape(shape).astype(np.float32)
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
        # Normalize to [0, 1]: the GNN always operates in that range, and it
        # makes error_bound naturally relative -- applied directly below, with
        # no separate rescale, it means a fraction of (vmax - vmin).
        values -= vmin
        values /= vmax - vmin
        # Integer sources: the final decompressed value is rounded to the
        # nearest raw integer (_restore_dtype), so the quantizer must verify
        # the bound against that rounded value, not the normalized one -- see
        # quantize()'s round_output=(span, offset) contract.
        round_output = (vmax - vmin, vmin) if dtype.kind in "bi" else False
        eb = float(error_bound)
        if eb <= 0:
            raise ValueError("error_bound must be > 0")

        # eb_ratio is a coarsest-level factor (see _GNN_EB_COARSE_FACTOR); depth-
        # normalise it against the resolved schedule depth so the coarse/fine
        # spread is the same across ranks. The set dedups the levels==1 case
        # where every factor collapses to a flat 1.0 (one encode, not four).
        coarse_candidates = (
            [float(eb_ratio)]
            if eb_ratio is not None
            else (
                list(_GNN_EB_COARSE_SWEEP)
                if tune == "size"
                else [_GNN_EB_COARSE_FACTOR]
            )
        )
        if isinstance(levels, str):
            import torch

            agg_level = _gp._load_inference_model(
                self.checkpoint_path, torch, self.device
            )[3]
            levels = _auto_levels(shape, agg_level)
        else:
            levels = int(levels)
        anchor_stride = 1 << levels
        ratio_candidates = sorted(
            {_per_step_eb_ratio(c, levels) for c in coarse_candidates}
        )
        edges = self._chunk_edges(shape, anchor_stride, chunk_size)
        # torch.compile costs seconds of dynamo warmup per process; only worth
        # it when there are enough chunk waves to amortize. "auto" defers to the
        # benchmark-backed crossover; explicit True is honored past a floor.
        # Frozen into the stream meta so decode replays the same float path.
        nchunks = int(np.prod([-(-n // e) for n, e in zip(shape, edges)]))
        if isinstance(compile, str):
            want_compile = (
                _COMPILE_AUTO_CROSSOVER is not None
                and nchunks >= _COMPILE_AUTO_CROSSOVER
            )
        else:
            want_compile = bool(compile) and nchunks >= _COMPILE_MIN_CHUNKS
        use_compile = want_compile
        candidates: list[tuple[int, bytes]] = []
        for ratio in ratio_candidates:
            payload, gates, fine_gates = self._compress_chunked_payload(
                values, round_output, eb, ratio, edges, use_compile,
                levels, anchor_stride, radius, fp16, gate, gate_fine,
            )
            if gates is not None and not any(gates):
                gates = None  # gate never fired -> plain ungated stream
            if fine_gates is not None and not any(fine_gates):
                fine_gates = None  # finest gate never fired -> nothing to store
            meta = {
                "shape": list(original_shape),
                "dtype": dtype.str,
                "error_bound": eb,
                "levels": levels,
                "radius": radius,
                "vmin": vmin,
                "vmax": vmax,
                "eb_ratio": ratio,
                "checkpoint_hash": self.checkpoint_hash.hex(),
                "chunks": list(edges),
                "m_tile": int(_gp._M_TILE),  # replay the exact float path
                "fp16": bool(fp16),
                "compiled": bool(use_compile),
            }
            if gates is not None:
                # Per-chunk-stage coarse gate words, zstd+base64 packed (the JSON
                # header is otherwise uncompressed; repeated/zero words collapse).
                meta["gates"] = _pack_gates(gates)
            if fine_gates is not None:
                # One adaptive finest-gate descriptor per chunk (gate_fine),
                # packed the same way; decode reads one per chunk.
                meta["fine_gates"] = _pack_gates(fine_gates)
            stream = _write_stream(meta, payload, int(zstd_level))
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
            packed_gates = meta.get("gates")
            packed_fine = meta.get("fine_gates")
            values = _decompress_chunked(
                payload,
                shape,
                ebs,
                int(meta["radius"]),
                predictor,
                edges,
                gates=None if packed_gates is None else _unpack_gates(packed_gates),
                fine_gates=None if packed_fine is None else _unpack_gates(packed_fine),
            )
        finally:
            _gp._M_TILE = saved_tile
        # In place: `values` is a fresh float32 buffer from the decode, and
        # out-of-place would hold two full-size copies of the field.
        values *= vmax - vmin  # undo compress()'s [0, 1] normalize
        values += vmin
        out = _restore_dtype(values.reshape(original_shape), dtype)
        return torch.as_tensor(out)

    decompress = uncompress

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
        radius: int,
        fp16: bool,
        gate: bool,
        gate_fine: bool,
    ) -> tuple[bytes, list[int] | None, list[int] | None]:
        predictor = self._chunked_predictor(
            levels, fp16=fp16, compile=bool(use_compile)
        )
        ebs = _chunk_stage_ebs(
            values.shape,
            levels,
            anchor_stride,
            _ANCHOR_BLOCK,
            eb,
            eb_ratio,
        )
        payload, gates, fine_gates = _compress_chunked(
            values[None, ...],
            ebs,
            radius,
            round_output,
            predictor,
            edges,
            gate=gate,
            gate_fine=gate_fine,
        )
        return payload, gates, fine_gates

    def _chunked_predictor(
        self,
        levels: int,
        meta: dict[str, Any] | None = None,
        *,
        fp16: bool = True,
        compile: bool = False,
    ) -> ChunkedGNNPredictor:
        # vmin/vmax are always 0.0/1.0: compress() normalizes the tensor to
        # [0, 1] up front, so the predictor never sees raw-scale values.
        anchor_stride = 1 << levels
        use_fp16 = bool(fp16 if meta is None else meta["fp16"])
        use_compile = bool(compile if meta is None else meta["compiled"])
        key = (int(levels), use_fp16, use_compile)
        predictor = self._chunked_predictors.get(key)
        if predictor is None:
            predictor = ChunkedGNNPredictor(
                self.checkpoint_path,
                0.0,
                1.0,
                device=self.device,
                levels=levels,
                anchor_stride=anchor_stride,
                anchor_block=_ANCHOR_BLOCK,
            )
            predictor.fp16 = use_fp16
            predictor.compile = use_compile
            self._chunked_predictors[key] = predictor
            # Runtime geometry dwarfs the model weights. A different schedule or
            # precision mode is unlikely to be reused before the current one, so
            # do not let several complete geometry caches accumulate on a codec.
            while len(self._chunked_predictors) > 1:
                self._chunked_predictors.popitem(last=False)
        else:
            self._chunked_predictors.move_to_end(key)
        return predictor

    def _checkpoint_hash(self) -> bytes:
        # Streamed: this runs on every codec construction, and read_bytes()
        # would pull the whole checkpoint into RAM each time.
        return _gp.file_sha256(self.checkpoint_path)[:16]


GNNCodec = GNNCompressorCodec
