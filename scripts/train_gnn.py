"""Train the lightweight GNN predictor (deepsz/gnn_predictor.py) purely on
synthetic scientific-like fields: 2-D fields mixed with anisotropic 4-D fields,
both from the same generator (smooth Gaussian random fields *and* power-law
turbulence, with random-hyperplane discontinuities). No natural images or
external data are needed. CPU-friendly: the model is ~20k params, a few thousand
steps suffice.

    conda run -n nf python scripts/train_gnn.py

Each step generates fresh 2-D fields (shape --crop x --crop) and/or 4-D fields
(--synthetic-shape) directly on the device, teacher-forces the hierarchical
stage schedule (levels.py) with true values (+ quantization-like noise at the
sampled error bound) and minimises discretized-Laplacian NLL over the hole
positions of every refinement stage. Checkpoint -> data/gnn_predictor.pt.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import threading
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepsz.gnn_predictor import (
    CKPT_VERSION,
    _CompactFrame,
    build_chunk_geoms,
    build_model,
    build_stage_geoms,
    stage_forward,
    value_halo_embed,
)
from deepsz.levels import stage_masks
from deepsz.synthetic_data import (
    DEFAULT_SYNTHETIC_DIST as _SYN,
    make_eval_field,
    mixed_batch_sizes,
    sample_synthetic_batch,
)


def load_tensor(path: str) -> np.ndarray:
    """Load a raw n-D tensor from .npy or a torch .pt/.pth for the codec eval."""
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith((".pt", ".pth")):
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(obj, torch.Tensor):
            raise ValueError(f"{path} holds {type(obj).__name__}, expected a tensor")
        return obj.detach().cpu().numpy()
    raise ValueError(f"unsupported extension: {path} (use .npy/.pt/.pth)")


def normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    """Min-max normalize a finite tensor to the closed interval ``[0, 1]``."""
    tensor = np.asarray(tensor, dtype=np.float32)
    if not np.isfinite(tensor).all():
        raise ValueError("cannot normalize an eval tensor containing NaN or infinity")
    lo = float(tensor.min())
    hi = float(tensor.max())
    if hi == lo:
        return np.zeros_like(tensor)
    return (tensor - lo) / (hi - lo)


def qz(pred, truth, eb):
    """Linear-quantised reconstruction (deepsz.quantizer): pred + the residual
    snapped to the 2*eb grid, so |recon - truth| <= eb. Works on np or torch."""
    round_ = torch.round if torch.is_tensor(truth) else np.round
    if torch.is_tensor(eb) and torch.is_tensor(truth) and eb.ndim == 1:
        eb = eb.reshape(-1, *([1] * (truth.ndim - 1)))
    return pred + 2 * eb * round_((truth - pred) / (2 * eb))


def interp_eval(x, masks, eb, order):
    """Real interp-predictor closed loop with linear quantisation, batch 1 on
    CPU. At each stage the interp predictor fills the holes from the *quantised*
    reconstruction of earlier stages (never the truth), exactly as the codec's
    InterpPredictor does; returns (sum |interp - truth|, n_holes) — the baseline
    prediction MAE the GNN must beat."""
    from scipy.interpolate import griddata

    h, w = masks[0].shape
    truth = x.detach().cpu().numpy().reshape(h, w)
    recon = np.zeros((h, w), np.float32)
    known = masks[0].copy()
    recon[known] = qz(0.0, truth[known], eb)  # anchors: quantised against pred 0
    abs_err = 0.0
    npix = 0
    for pos in masks[1:]:
        if not pos.any():
            continue
        pts = np.column_stack(np.nonzero(known)).astype(np.float64)
        q = np.column_stack(np.nonzero(pos)).astype(np.float64)
        z = griddata(pts, recon[known], q, method=order)
        nan = np.isnan(z)  # holes outside the known convex hull
        if nan.any():
            z[nan] = griddata(pts, recon[known], q, method="nearest")[nan]
        tg = truth[pos]
        abs_err += float(np.abs(z - tg).sum())
        npix += tg.size
        recon[pos] = qz(z, tg, eb)  # feed back the quantised interp recon
        known = known | pos
    return abs_err, npix


def residual_rgb(res):
    """Signed residual -> RGB via a diverging colormap symmetric about zero
    (blue = under-predicted, red = over-predicted, white = exact)."""
    import matplotlib

    m = float(np.abs(res).max()) or 1.0
    return matplotlib.colormaps["seismic"]((res / m + 1) / 2)[..., :3]


def _batch_scalar(value, batch, device):
    if torch.is_tensor(value):
        value = value.to(device=device, dtype=torch.float32).reshape(-1)
        if value.numel() == 1:
            return value.expand(batch)
        return value.reshape(batch)
    return torch.full((batch,), float(value), dtype=torch.float32, device=device)


def sample_noise(batch, args, device):
    """Per-sample normalized teacher-forcing noise range."""
    if args.noise_range is None:
        return _batch_scalar(args.noise, batch, device)
    lo, hi = args.noise_range
    log_lo, log_hi = np.log(float(lo)), np.log(float(hi))
    return torch.empty(batch, device=device).uniform_(log_lo, log_hi).exp()


def training_autocast(fp16, device):
    """CUDA FP16 autocast matching the codec's message-pass execution mode."""
    if fp16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class ModelEMA:
    """Exponential moving average of the training weights, evaluated and saved
    instead of the live weights.

    The synthetic eval is deterministic (fixed-seed field, fixed eb, fp32), so
    its step-to-step swings are entirely the live weights orbiting the noisy
    optimum. Averaging the iterate (Polyak/EMA) sits nearer the centre of that
    orbit -> a smoother, usually-lower eval curve, at no cost to the live
    optimisation, which proceeds on the raw weights.

    The shadow is kept in float32 for a numerically stable average and cast back
    to each destination dtype on ``copy_to``. Non-floating buffers (e.g. integer
    schedule state) are copied through, never averaged. ``decay`` is warmed up as
    ``min(decay, (1 + step) / (10 + step))`` so the average tracks quickly while
    the model is still moving fast and only smooths once it settles -- the fix
    for the usual "EMA lags early training" failure mode.
    """

    def __init__(self, model, decay):
        self.decay = decay
        sd = model.state_dict()
        self.shadow = {k: v.detach().clone().float() for k, v in sd.items()}
        self._is_float = {k: v.is_floating_point() for k, v in sd.items()}

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if self._is_float[k]:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def copy_to(self, model):
        dst = model.state_dict()
        model.load_state_dict(
            {k: self.shadow[k].to(dst[k].dtype) for k in self.shadow}
        )


def discretized_laplace_nll(mu, log_b, target, eb):
    """Mean code length in bits for target under a discretized Laplace cell."""
    eb = torch.as_tensor(eb, device=target.device, dtype=target.dtype).reshape(-1, 1)
    b = torch.exp2(log_b).clamp_min(torch.finfo(target.dtype).tiny)
    log_half = -math.log(2.0)
    log2e = 1.0 / math.log(2.0)

    abs_r = (target - mu).abs()
    rho = abs_r / b
    e = eb / b

    # |residual| >= eb: quantization cell is wholly on one side of zero.
    tail_mass = torch.log1p(
        -torch.exp((-2.0 * e).clamp_max(-torch.finfo(target.dtype).eps))
    )
    # Past ~32 bits (the codec stores such points as raw f32 outliers anyway)
    # compress the linear tail logarithmically: keeps a scale-free ~1/|r|
    # gradient instead of ~1/b, which hits 1e7 at eb=1e-6 (b init ~= eb) and
    # poisons Adam's moments (flat loss) then NaNs. clamp_min guards the
    # eagerly-evaluated where branch from log1p(<=-1).
    t = rho - e
    t_max = 32.0 * math.log(2.0)
    t = torch.where(t > t_max, t_max + torch.log1p((t - t_max).clamp_min(0.0)), t)
    logp_tail = log_half - t + tail_mass

    # |residual| < eb: quantization cell straddles zero. Clamp rho to this
    # branch's domain before evaluating so torch.where's eager branch execution
    # cannot overflow on far-tail residuals.
    rho_cross = torch.minimum(rho, e)
    cross_p = (
        1.0 - 0.5 * torch.exp(-(e - rho_cross)) - 0.5 * torch.exp(-(e + rho_cross))
    )
    logp_cross = torch.log(cross_p.clamp_min(1e-30))

    logp = torch.where(abs_r >= eb, logp_tail, logp_cross)
    return -logp * log2e


def entropy_bits_from_codes(codes):
    """Order-0 entropy H(codes), in bits/symbol."""
    if not codes:
        return float("nan")
    flat = torch.cat([c.reshape(-1).detach().cpu() for c in codes])
    if flat.numel() == 0:
        return float("nan")
    _, counts = torch.unique(flat, return_counts=True)
    p = counts.to(torch.float64) / flat.numel()
    return float(-(p * torch.log2(p)).sum())


class _PeakRSS:
    """Sample process resident-set size in a background thread and hold the peak
    (in MB) reached over the ``with`` body. Linux-only (/proc/self/statm); the
    training host is Linux. Captures allocator peaks the torch CUDA counters
    miss (host-side numpy/zstd/rANS buffers)."""

    _PAGE = os.sysconf("SC_PAGE_SIZE")

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def _rss_mb(self) -> float:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * self._PAGE / 1e6

    def _run(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, self._rss_mb())

    def __enter__(self):
        self.peak = self._rss_mb()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._rss_mb())


def eval_tensor_codec(model, d, args, tensor, eb, device, ckpt_path):
    """Roundtrip `tensor` (a small n-D array, e.g. 4-D) through the *real* codec
    at the current weights and return distortion / rate / peak-RAM / time metrics
    for wandb. Unlike the closed-loop image eval, this exercises the full
    compress+decompress byte path (rANS, zstd, chunking), so the rate is the
    actual stream size and the RAM/time are what the deployed codec pays.

    The codec loads weights from disk, so the live model is frozen to `ckpt_path`
    first (overwritten each call; the model cache keys on mtime so this reloads).
    """
    from deepsz.gnn_codec import GNNCompressorCodec

    torch.save(
        {
            "state_dict": model.state_dict(),
            "d": d,
            "agg_level": args.agg_level,
            "version": CKPT_VERSION,
        },
        ckpt_path,
    )
    # compile=False: each eval loads a fresh model instance, so compiling here
    # recompiles from scratch every 500 steps and (under
    # DEEPSZ_COMPILE_MODE=reduce-overhead) allocates CUDA-graph pools sized by
    # the eval tensor each time — RAM, not speed. Eval timing is therefore
    # eager-mode pessimistic; benchmark deployed speed with eval_tensor.py.
    codec = GNNCompressorCodec(
        ckpt_path,
        error_bound=eb,
        levels=(args.levels or 4),
        device=str(device),
        fp16=args.fp16,
        compile=False,
    )

    base_gpu = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_gpu = torch.cuda.memory_allocated()

    with _PeakRSS() as rss:
        t0 = time.time()
        stream = codec.compress(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_comp = time.time() - t0
        t0 = time.time()
        rec = codec.uncompress(stream).numpy().reshape(tensor.shape)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_dec = time.time() - t0

    a = tensor.astype(np.float64)
    r = rec.astype(np.float64)
    max_err = float(np.abs(a - r).max())
    mse = float(np.mean((a - r) ** 2))
    vrange = max(float(a.max()) - float(a.min()), 1e-12)
    psnr = 10 * np.log10(vrange**2 / mse) if mse > 0 else float("inf")
    bpv = 8 * len(stream) / tensor.size  # tensor is float32 -> 32 bpv raw

    metrics = {
        "eval_tensor/psnr_db": psnr,  # distortion
        "eval_tensor/max_abs_err": max_err,  # distortion (bound check: <= eb)
        "eval_tensor/bits_per_value": bpv,  # rate
        "eval_tensor/ratio": 4 * tensor.size / len(stream),  # rate (vs f32)
        "eval_tensor/compress_s": t_comp,  # inference time (encode)
        "eval_tensor/decompress_s": t_dec,  # inference time (decode)
        "eval_tensor/peak_host_mb": rss.peak,  # max ram (host RSS peak)
    }
    if device.type == "cuda":
        metrics["eval_tensor/peak_gpu_mb"] = (
            torch.cuda.max_memory_allocated() - base_gpu
        ) / 1e6  # max ram (gpu)
        torch.cuda.empty_cache()
    return metrics


def run_stages(
    model,
    x,
    geoms,
    d,
    device,
    eb,
    teacher_force=False,
    collect=False,
    collect_bins=False,
):
    """Run the stage schedule over truth `x` (B, N) using precomputed per-stage
    geometry `geoms` (from ``build_stage_geoms``); return (sum NLL bits,
    n_holes, known_vals, pred_only, aux). ``aux`` includes summed prediction
    absolute error for diagnostic MAE logging. Training passes
    ``teacher_force=True`` and feeds truth plus uniform +/- eb noise. Eval
    leaves teacher forcing off: the fed-back known values are the model's own
    linear-quantised reconstructions (real closed-loop inference, matching the
    codec).

    Mirrors the codec's evolution: at stage i we finalize the points revealed in
    stage i-1 into the field, then predict stage i. The head context of stage i-1
    equals the finalize context of stage i (same field, same geometry), so it is
    fed back as `finalize_ctx` instead of being pooled twice.

    `collect` builds the per-pixel `pred_only` image (raw preds at the holes) for
    logging; it costs a scatter per stage and is only read when logging eval
    images, so training leaves it off and gets `pred_only=None`."""
    B, N = x.shape
    eb = _batch_scalar(eb, B, device)
    E = torch.zeros(B, N, geoms[0].ndim, d, device=device)
    a0 = geoms[0].query_idx  # anchors
    known_vals = torch.full_like(x, 0.5)

    def reveal(idx):  # teacher-force truth (+ training noise) at `idx`
        nz = (torch.rand(B, idx.numel(), device=device) * 2 - 1) * eb[:, None]
        known_vals[:, idx] = (x[:, idx] + nz).clamp(0, 1)

    if teacher_force:
        reveal(a0)
    else:  # anchors quantised against pred 0, like the codec's stage 0
        known_vals[:, a0] = qz(torch.zeros(B, a0.numel(), device=device), x[:, a0], eb)
    # recon *without* residuals (raw preds at holes); only for image logging
    pred_only = known_vals.clone() if collect else None

    nll_sum = torch.zeros((), device=device)
    abs_err = torch.zeros((), device=device)
    npix = 0
    bins = []
    head_ctx = None
    for i in range(1, len(geoms)):
        gp, gh = geoms[i - 1], geoms[i]
        (pred, log_b), E, head_ctx = stage_forward(
            model,
            E,
            gp,
            gh,
            known_vals[:, gp.query_idx],
            torch,
            finalize_ctx=head_ctx,
            eb=eb,
        )
        idx = gh.query_idx
        tgt = x[:, idx]
        nll = discretized_laplace_nll(pred, log_b, tgt, eb)
        nll_sum = nll_sum + nll.sum()
        abs_err = abs_err + (pred.detach() - tgt).abs().sum()
        npix += tgt.numel()
        if collect_bins:
            bins.append(torch.round((tgt - pred) / (2 * eb[:, None])).to(torch.int64))
        if collect:
            pred_only[:, idx] = pred
        if teacher_force:
            reveal(idx)
        else:
            known_vals[:, idx] = qz(pred, tgt, eb)
    # known_vals = full closed-loop recon; pred_only = same minus the quantised
    # residual added at each hole (residual = known_vals - pred_only).
    return nll_sum, npix, known_vals, pred_only, {"bins": bins, "abs_err": abs_err}


def _run_chunk(model, cg, geoms, E, known_vals, x, gidx, eb, device, reveal):
    """One chunk of the chunked-scene step: the codec's local stage chain with
    teacher forcing. Returns (nll bits, n holes, abs err)."""
    nll = torch.zeros((), device=device)
    abs_err = torch.zeros((), device=device)
    npix = 0
    ctx = None
    for j in range(1, len(cg.chain)):
        prev, s = cg.chain[j - 1], cg.chain[j]
        gp, gh = geoms[prev], geoms[s]
        fvals = None if gp is None else known_vals[:, gidx[prev]]
        (pred, log_b), E, ctx = stage_forward(
            model, E, gp, gh, fvals, torch, finalize_ctx=ctx, eb=eb
        )
        tgt = x[:, gidx[s]]
        nll = nll + discretized_laplace_nll(pred, log_b, tgt, eb).sum()
        abs_err = abs_err + (pred.detach() - tgt).abs().sum()
        npix += tgt.numel()
        reveal(gidx[s])
    return nll, npix, abs_err


def run_chunked_scene(
    model, x, hw, axis, order, levels, stride, d, device, eb, agg_level=None
):
    """Teacher-forced two-chunk closed loop, mirroring the chunked codec: the
    n-D scene (``hw`` = full shape) is split in half along ``axis``; anchors are
    revealed, then the chunks are coded in ``order`` — the first sees only
    anchor context across the border, the second reads the first's
    reconstructed values as halo (``value_halo_embed``), same as
    ChunkedGNNPredictor's wave halo fill. Same geometry/halo code, with
    gradients."""
    B, N = x.shape
    shape = tuple(int(s) for s in hw)
    ndim = len(shape)
    eb = _batch_scalar(eb, B, device)
    edges = tuple(s // 2 if k == axis else s for k, s in enumerate(shape))
    grid = tuple(2 if k == axis else 1 for k in range(ndim))
    known_vals = torch.full_like(x, 0.5)

    def reveal(idx):
        if not torch.is_tensor(idx):
            idx = torch.from_numpy(np.asarray(idx, np.int64)).to(device)
        nz = (torch.rand(B, idx.numel(), device=device) * 2 - 1) * eb[:, None]
        known_vals[:, idx] = (x[:, idx] + nz).clamp(0, 1)

    origins = []
    for ci in range(2):
        starts = tuple(g * e for g, e in zip(np.unravel_index(ci, grid), edges))
        origins.append(starts)
        axes = [
            o + np.arange(e)[(np.arange(e) + o) % stride == 0]
            for o, e in zip(starts, edges)
        ]
        mg = np.meshgrid(*axes, indexing="ij")
        reveal(np.ravel_multi_index([m.reshape(-1) for m in mg], shape))

    cg = build_chunk_geoms(edges, levels, stride, 1, torch, device, agg_level)
    coded = np.zeros(2, bool)
    nll = torch.zeros((), device=device)
    abs_err = torch.zeros((), device=device)
    npix = 0
    for ci in order:
        frame = _CompactFrame(cg, origins[ci], shape, edges, grid, coded, torch, device)
        E = torch.zeros(B, frame.n_compact, ndim, d, device=device)
        if len(frame.h_gflat):  # trailing halo block
            gflat_t = torch.from_numpy(frame.h_gflat).to(device)
            halo = value_halo_embed(model, known_vals[:, gflat_t], ndim)
            E = torch.cat([E[:, : frame.halo_rows.start], halo], dim=1)
        # ravel(c + o) = c @ strides + o @ strides for in-bounds coords, and the
        # per-stage `c @ strides` is cached on the chunk geometry (stage_offsets).
        strides = np.cumprod((1,) + tuple(shape)[:0:-1])[::-1].astype(np.int64)
        obase = int(np.asarray(origins[ci], np.int64) @ strides)
        gidx = [
            None if off is None else torch.from_numpy(off + obase).to(device)
            for off in cg.stage_offsets(strides)
        ]
        n1, np1, a1 = _run_chunk(
            model, cg, frame.geoms, E, known_vals, x, gidx, eb, device, reveal
        )
        nll, npix, abs_err = nll + n1, npix + np1, abs_err + a1
        coded[ci] = True
    return nll, npix, {"abs_err": abs_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parent.parent / "data" / "gnn_predictor.pt"
        ),
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="save a numbered checkpoint every N steps "
        "(0 disables periodic checkpoints)",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=8,
        help="number of 2-D synthetic fields per optimizer step",
    )
    ap.add_argument(
        "--crop",
        type=int,
        default=_SYN.crop,
        help="side length of each 2-D synthetic field",
    )
    ap.add_argument(
        "--synthetic-frac",
        type=float,
        default=_SYN.frac,
        help="target fraction of scalar training points from 4-D "
        "fields in every optimizer step; the 2-D-field batch size "
        "is derived to approach this fraction",
    )
    ap.add_argument(
        "--synthetic-2d-correlation",
        type=float,
        nargs=2,
        default=_SYN.correlation_2d,
        metavar=("MIN", "MAX"),
        help="correlation-length band (grid cells) for the 2-D "
        "synthetic fields (larger than the 4-D band to match the "
        "larger crop); see --synthetic-correlation",
    )
    ap.add_argument(
        "--synthetic-turbulence-frac",
        type=float,
        default=_SYN.turbulence_frac,
        help="per-field probability the spectrum is turbulent "
        "(power-law / fractal) rather than a smooth Gaussian field",
    )
    ap.add_argument(
        "--synthetic-discontinuities",
        type=int,
        default=_SYN.max_discontinuities,
        help="maximum number of random-hyperplane jump fronts "
        "(shocks / interfaces) added per field; when triggered a field "
        "draws 1..N cuts (0 disables discontinuities)",
    )
    ap.add_argument(
        "--synthetic-discontinuity-frac",
        type=float,
        default=_SYN.discontinuity_frac,
        help="per-field probability of getting any discontinuities at all",
    )
    ap.add_argument(
        "--synthetic-shape",
        type=int,
        nargs=4,
        default=_SYN.shape_4d,
        metavar=("N0", "N1", "N2", "N3"),
        help="shape of each generated 4-D training field",
    )
    ap.add_argument(
        "--synthetic-correlation",
        type=float,
        nargs=2,
        default=_SYN.correlation_4d,
        metavar=("MIN", "MAX"),
        help="Gaussian correlation-length band (grid cells): each "
        "field draws a base sigma log-uniformly in [MIN/2, MAX] "
        "then jitters it per axis, so smoothness/anisotropy vary",
    )
    ap.add_argument(
        "--synthetic-batch",
        type=int,
        default=_SYN.batch_4d,
        help="number of 4-D fields per optimizer step; with a mixed "
        "step, --batch is derived from the target point fraction",
    )
    ap.add_argument(
        "--synthetic-stride",
        type=int,
        default=_SYN.stride,
        help="anchor stride for synthetic fields (power of two)",
    )
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="linearly ramp the LR from 0 to --lr over this many steps before "
        "the flat/decay schedule (0 disables warmup)",
    )
    ap.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="decay for the EMA of the weights used for eval and checkpoints; "
        "0 disables EMA and evals/saves the live weights",
    )
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="fixed uniform +/- noise on known values, in normalized [0,1] units",
    )
    ap.add_argument(
        "--noise-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="sample each training example's +/- noise log-uniformly "
        "from [MIN, MAX], overriding --noise",
    )
    ap.add_argument("--max-radius", type=int, default=64)
    ap.add_argument(
        "--agg-level",
        type=int,
        default=2,
        help="maximum L1 neighbourhood aggregation level: 1 uses "
        "axis-aligned directions only; 2 also uses two-axis "
        "diagonals (default: 2)",
    )
    ap.add_argument(
        "--chunk-frac",
        type=float,
        default=0.5,
        help="fraction of steps trained on the two-chunk scene "
        "(chunked-codec halo regime); the rest train the "
        "whole-crop schedule",
    )
    ap.add_argument(
        "--baseline",
        choices=("cubic", "linear"),
        default="cubic",
        help="SZ-style interpolation used for the reference line",
    )
    ap.add_argument(
        "--eval-shape",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("H", "W"),
        help="shape of the fixed held-out 2-D synthetic eval field",
    )
    ap.add_argument(
        "--eval-every",
        type=int,
        default=50,
        help="evaluate model bpp on the eval field every N steps",
    )
    ap.add_argument(
        "--img-every",
        type=int,
        default=500,
        help="log eval reconstruction images every N steps",
    )
    ap.add_argument(
        "--eval-eb",
        type=float,
        default=0.01,
        help="error bound (in [0,1]) for the real-inference eval loop",
    )
    ap.add_argument(
        "--eval-tensor",
        default=None,
        help="small n-D tensor (.npy/.pt, e.g. a 4-D field) roundtripped "
        "through the full codec each --eval-tensor-every steps; logs "
        "distortion / rate / peak-RAM / inference-time to wandb",
    )
    ap.add_argument(
        "--eval-tensor-eb",
        type=float,
        default=None,
        help="absolute error bound for --eval-tensor (default: --eval-eb)",
    )
    ap.add_argument(
        "--eval-tensor-normalize",
        action="store_true",
        help="min-max normalize --eval-tensor to [0,1] before its "
        "codec roundtrip (the error bound is then in normalized units)",
    )
    ap.add_argument(
        "--eval-tensor-every",
        type=int,
        default=500,
        help="run the --eval-tensor codec roundtrip every N steps "
        "(full compress+decompress; keep it a multiple of eval-every)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--levels",
        type=int,
        default=None,
        help="fix the stage levels (default: random 3/4/5 per step; "
        "set it for a deterministic, comparable profile)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=None,
        help="fix the anchor stride (default: random 8/16/32)",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu | cuda | cuda:N",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="train the GNN message pass with CUDA FP16 autocast "
        "and gradient scaling; readout/loss remain FP32",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the message-pass embed with dynamic "
        "shapes (one-off compilation cost on first steps)",
    )
    ap.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="wandb logging mode",
    )
    ap.add_argument("--wandb-project", default="gnn-sz")
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--profile",
        type=int,
        default=0,
        help="warm up this many steps (compile, caches), then "
        "record exactly 1 step with torch.profiler, print "
        "the op table + write trace.json, and exit",
    )
    args = ap.parse_args()
    if args.noise_range is not None:
        lo, hi = args.noise_range
        if lo <= 0 or hi < lo:
            raise SystemExit("--noise-range must satisfy 0 < MIN <= MAX")
    if not 0.0 <= args.synthetic_frac <= 1.0:
        raise SystemExit("--synthetic-frac must be in [0, 1]")
    if any(n < 2 for n in args.synthetic_shape):
        raise SystemExit("--synthetic-shape entries must all be >= 2")
    if any(s <= 0 for s in args.synthetic_correlation):
        raise SystemExit("--synthetic-correlation entries must all be > 0")
    if any(s <= 0 for s in args.synthetic_2d_correlation):
        raise SystemExit("--synthetic-2d-correlation entries must all be > 0")
    if not 0.0 <= args.synthetic_turbulence_frac <= 1.0:
        raise SystemExit("--synthetic-turbulence-frac must be in [0, 1]")
    if args.synthetic_discontinuities < 0:
        raise SystemExit("--synthetic-discontinuities must be >= 0")
    if not 0.0 <= args.synthetic_discontinuity_frac <= 1.0:
        raise SystemExit("--synthetic-discontinuity-frac must be in [0, 1]")
    if any(n < 2 for n in args.eval_shape):
        raise SystemExit("--eval-shape entries must all be >= 2")
    if args.synthetic_batch < 1:
        raise SystemExit("--synthetic-batch must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if not 0.0 <= args.ema_decay < 1.0:
        raise SystemExit("--ema-decay must be in [0, 1)")
    if args.synthetic_stride < 2 or args.synthetic_stride & (args.synthetic_stride - 1):
        raise SystemExit("--synthetic-stride must be a power of two >= 2")
    if args.agg_level < 1:
        raise SystemExit("--agg-level must be >= 1")
    if args.save_every < 0:
        raise SystemExit("--save-every must be >= 0")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    synthetic_rng = np.random.default_rng(args.seed + 1)

    # per-run dir: <out-parent>/runs/<date>-<config-hash>/ holds the checkpoint
    # and a config.json snapshot, so concurrent/repeated runs never clobber.
    cfg_hash = hashlib.sha1(repr(sorted(vars(args).items())).encode()).hexdigest()[:6]
    run_dir = (
        Path(args.out).resolve().parent
        / "runs"
        / f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg_hash}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    out = run_dir / Path(args.out).name
    print(f"run dir: {run_dir}")

    wandb_config = {
        k: v for k, v in vars(args).items() if k not in {"noise", "noise_range"}
    }

    import wandb

    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=wandb_config,
        dir=str(run_dir),
    )

    device = torch.device(args.device)
    amp_enabled = args.fp16 and device.type == "cuda"
    if args.fp16 and not amp_enabled:
        print(f"NOTE: --fp16 only engages on CUDA; using FP32 on device={device}")
    print(f"device={device}")
    field2d_batch, synthetic_batch, synthetic_point_fraction = mixed_batch_sizes(
        args.crop,
        args.synthetic_shape,
        args.synthetic_frac,
        args.batch,
        args.synthetic_batch,
    )
    if args.synthetic_frac:
        field2d_points = field2d_batch * args.crop**2
        synthetic_points = synthetic_batch * math.prod(args.synthetic_shape)
        print(
            "synthetic mix: "
            f"target={args.synthetic_frac:g}, "
            f"actual point fraction={synthetic_point_fraction:.6g}, "
            f"2d={field2d_batch}x{args.crop}^2={field2d_points} points, "
            f"4d={synthetic_batch}x{tuple(args.synthetic_shape)}="
            f"{synthetic_points} points, "
            f"2d-correlation={tuple(args.synthetic_2d_correlation)}, "
            f"4d-correlation={tuple(args.synthetic_correlation)}, "
            f"turbulence-frac={args.synthetic_turbulence_frac:g}, "
            f"max-discontinuities={args.synthetic_discontinuities}, "
            f"discontinuity-frac={args.synthetic_discontinuity_frac:g}, "
            f"stride={args.synthetic_stride}"
        )

    model = build_model(args.d, args.agg_level).to(device)
    if args.compile:
        mode = os.environ.get("DEEPSZ_COMPILE_MODE") or None
        # dynamic=True is required: embed specializes on geom.M (per-stage
        # query count) and the schedule has hundreds of distinct M values --
        # static shapes blow the recompile limit and fall back to eager.
        # CUDA graphs would need M padded to a few buckets first.
        # Raised limit: the M<=_M_TILE branch / ndim / dtype variants can
        # still exceed the default of 8.
        torch._dynamo.config.cache_size_limit = 64
        # embed is the bulk of the FLOPs; finalize + head_of are the eager
        # remainder (addmm/mul/sum storm of ~10us kernels, launch-bound).
        # finalize/head_of never get reduce-overhead: stage_forward chains
        # embed's ctx output across stages, and a CUDA-graph replay of any
        # sibling graph overwrites that static buffer before finalize reads
        # it ("output overwritten by a subsequent run"). Default mode still
        # fuses their kernel storm without cudagraphs.
        model.embed = torch.compile(model.embed, dynamic=True, mode=mode)
        model.finalize = torch.compile(model.finalize, dynamic=True)
        model.head_of = torch.compile(model.head_of, dynamic=True)
        print(
            "torch.compile enabled for embed/finalize/head_of "
            f"(embed mode={mode or 'default'})"
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    decay_start = int(args.steps * 0.6)
    warmup = max(0, args.warmup)

    def lr_factor(s):
        # Linear warmup 0 -> 1 over `warmup` steps (Adam moments are cold at
        # step 0 and the schedule is otherwise flat at full LR until 60%), then
        # flat, then linear decay to 0 over the last 40%.
        warm = min(1.0, (s + 1) / warmup) if warmup else 1.0
        decay = min(1.0, (args.steps - s) / max(args.steps - decay_start, 1))
        return warm * decay

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} params, d={args.d}")

    # Eval and checkpoints run on an EMA of the weights, not the live ones, so
    # the deterministic eval reflects the settled model rather than the latest
    # noisy step. A dedicated (uncompiled) eval model receives the EMA weights
    # before each eval; the live `model` keeps training untouched. With
    # --ema-decay 0 both fall back to the live weights.
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay else None
    if ema is not None:
        eval_model = build_model(args.d, args.agg_level).to(device)
        eval_model.eval()
    else:
        eval_model = model

    def sync_eval_model():
        if ema is not None:
            ema.copy_to(eval_model)
        return eval_model

    # Fixed held-out eval: one fixed-seed 2-D synthetic field, fixed masks. Both
    # the model and the interp baseline run a real closed-loop inference with
    # linear quantisation at --eval-eb (predictions fed the *quantised* recon of
    # earlier stages, never the truth), so the reported MAE is the prediction
    # error the codec pays. The interp baseline is model-independent -> compute
    # it once and draw it as a reference line.
    eval_x, (eh, ew) = make_eval_field(
        tuple(args.eval_shape),
        device,
        tuple(args.synthetic_2d_correlation),
        args.synthetic_turbulence_frac,
        args.synthetic_discontinuities,
        args.synthetic_discontinuity_frac,
    )
    eval_masks = stage_masks((eh, ew), 4, 16, anchor_block=1)  # interp baseline
    eval_geoms, _ = build_stage_geoms(
        (eh, ew), 4, 16, 1, args.max_radius, torch, device, args.agg_level
    )

    be, bn = interp_eval(eval_x, eval_masks, args.eval_eb, args.baseline)
    baseline_mae = be / max(bn, 1)
    eval_name = f"synthetic-{eh}x{ew}"
    print(
        f"{args.baseline} interp baseline on "
        f"{eval_name}: MAE={baseline_mae:.5f} (eb={args.eval_eb})"
    )
    wandb.summary["baseline_mae"] = baseline_mae
    truth_img = eval_x.detach().cpu().numpy().reshape(eh, ew)
    wandb.log({"eval/truth": wandb.Image(truth_img, caption=eval_name)}, step=0)

    # Optional full-codec roundtrip eval on a small n-D tensor (e.g. 4-D). Loaded
    # once as float32; weights are frozen to this temp checkpoint each eval so the
    # codec (which reads weights from disk) sees the current model.
    eval_tensor = None
    eval_tensor_ckpt = run_dir / "eval_tensor.pt"
    eval_tensor_eb = (
        args.eval_eb if args.eval_tensor_eb is None else args.eval_tensor_eb
    )
    if args.eval_tensor:
        eval_tensor = load_tensor(args.eval_tensor).astype(np.float32)
        if args.eval_tensor_normalize:
            eval_tensor = normalize_tensor(eval_tensor)

    run_loss = 0.0
    last_eval = float("nan")
    last_tensor_metrics = None
    last_saved_step = None

    prof = None
    if args.profile:
        acts = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        # ponytail: no profiler schedule -- started by hand on the last step,
        # after args.profile warmup steps (compile, caches), so exactly one
        # fully warmed-up step is recorded.
        prof = torch.profiler.profile(
            activities=acts, record_shapes=True, with_stack=True
        )
        args.steps = args.profile + 1  # warmup + 1 recorded step

    # These weights are constant for the run. Both the 2-D and 4-D fields are
    # FFT-generated directly on the device from the synthetic generator, so
    # neither source needs a CPU prefetch pipeline.
    field2d_weight = 1.0 - synthetic_point_fraction
    synthetic_weight = synthetic_point_fraction
    bar = tqdm(range(1, args.steps + 1), desc="train")
    for step in bar:
        if prof is not None and step == args.steps:
            prof.start()  # record only the final, fully warmed-up step
        opt.zero_grad()
        metric_tensors = {}

        # The 2-D and 4-D examples cannot share a rectangular tensor batch.
        # Run them sequentially and backward immediately so their gradients
        # accumulate without retaining both computation graphs in GPU memory.
        if field2d_weight:
            x = sample_synthetic_batch(
                field2d_batch,
                (args.crop, args.crop),
                tuple(args.synthetic_2d_correlation),
                synthetic_rng,
                device=device,
                turbulence_frac=args.synthetic_turbulence_frac,
                max_discontinuities=args.synthetic_discontinuities,
                discontinuity_frac=args.synthetic_discontinuity_frac,
            )
            stride = args.stride or random.choice((8, 16, 32))
            levels = args.levels or stride.bit_length() - 1
            eb = sample_noise(x.shape[0], args, device)
            with training_autocast(args.fp16, device):
                if random.random() < args.chunk_frac:
                    axis = random.randint(0, 1)
                    order = random.choice(([0, 1], [1, 0]))
                    nll, npix, aux = run_chunked_scene(
                        model,
                        x,
                        (args.crop, args.crop),
                        axis,
                        order,
                        levels,
                        stride,
                        args.d,
                        device,
                        eb,
                        args.agg_level,
                    )
                else:
                    geoms, _ = build_stage_geoms(
                        (args.crop, args.crop),
                        levels,
                        stride,
                        1,
                        args.max_radius,
                        torch,
                        device,
                        args.agg_level,
                    )
                    nll, npix, _, _, aux = run_stages(
                        model, x, geoms, args.d, device, eb=eb, teacher_force=True
                    )
            field2d_loss = nll / max(npix, 1)
            field2d_mae = aux["abs_err"].detach() / max(npix, 1)
            scaler.scale(field2d_weight * field2d_loss).backward()
            metric_tensors.update(
                {
                    "train/field2d_bpp": field2d_loss.detach(),
                    "train/field2d_mae": field2d_mae,
                }
            )

        if synthetic_weight:
            field_shape = tuple(args.synthetic_shape)
            x = sample_synthetic_batch(
                synthetic_batch,
                field_shape,
                tuple(args.synthetic_correlation),
                synthetic_rng,
                device=device,
                turbulence_frac=args.synthetic_turbulence_frac,
                max_discontinuities=args.synthetic_discontinuities,
                discontinuity_frac=args.synthetic_discontinuity_frac,
            )
            stride = args.synthetic_stride
            levels = args.levels or stride.bit_length() - 1
            eb = sample_noise(x.shape[0], args, device)
            with training_autocast(args.fp16, device):
                geoms, _ = build_stage_geoms(
                    field_shape,
                    levels,
                    stride,
                    1,
                    args.max_radius,
                    torch,
                    device,
                    args.agg_level,
                )
                nll, npix, _, _, aux = run_stages(
                    model, x, geoms, args.d, device, eb=eb, teacher_force=True
                )
            synthetic_loss = nll / max(npix, 1)
            synthetic_mae = aux["abs_err"].detach() / max(npix, 1)
            scaler.scale(synthetic_weight * synthetic_loss).backward()
            metric_tensors.update(
                {
                    "train/synthetic_bpp": synthetic_loss.detach(),
                    "train/synthetic_mae": synthetic_mae,
                }
            )

        scaler.step(opt)
        scaler.update()
        sched.step()
        if ema is not None:
            ema.update(model, step)
        combined_loss = sum(
            weight * metric_tensors[key]
            for weight, key in (
                (field2d_weight, "train/field2d_bpp"),
                (synthetic_weight, "train/synthetic_bpp"),
            )
            if weight
        )
        combined_mae = sum(
            weight * metric_tensors[key]
            for weight, key in (
                (field2d_weight, "train/field2d_mae"),
                (synthetic_weight, "train/synthetic_mae"),
            )
            if weight
        )
        metric_tensors.update({"train/bpp": combined_loss, "train/mae": combined_mae})
        # Transfer all scalar metrics together.  Calling .item() for the 2-D
        # metrics before launching the 4-D pass introduced several host
        # synchronizations and exposed the field-generation latency.
        names = tuple(metric_tensors)
        values = (
            torch.stack([metric_tensors[name].float() for name in names]).cpu().tolist()
        )
        train_log = dict(zip(names, values))
        run_loss += train_log["train/bpp"]
        train_log.update(
            {
                "train/synthetic_point_fraction": synthetic_point_fraction,
                "lr": sched.get_last_lr()[0],
            }
        )
        wandb.log(train_log, step=step)

        if step % args.eval_every == 0:
            m = sync_eval_model()
            m.eval()
            with torch.no_grad(), training_autocast(args.fp16, device):
                bits, en, recon, pred_only, aux = run_stages(
                    m,
                    eval_x,
                    eval_geoms,
                    args.d,
                    device,
                    eb=args.eval_eb,
                    collect=True,
                    collect_bins=True,
                )
            if m is model:
                model.train()
            # eval runs the whole image (>>train crop); free its reserved pool
            # so the next train step doesn't OOM on the fragmented remainder.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            last_eval = bits.item() / max(en, 1)
            eval_mae = aux["abs_err"].item() / max(en, 1)
            marginal = entropy_bits_from_codes(aux["bins"])
            log = {
                "eval/bpp_model": last_eval,
                "eval/bpp_marginal": marginal,
                "eval/bpp_gain_frac": (
                    (marginal - last_eval) / marginal if marginal > 0 else float("nan")
                ),
            }
            log["eval/mae"] = eval_mae
            if step % args.img_every == 0:  # img-every: use a multiple of eval-every
                recon = recon.detach().cpu().numpy().reshape(eh, ew)
                pred_only = pred_only.detach().cpu().numpy().reshape(eh, ew)
                # prediction reconstruction (no residuals) + the quantised
                # residuals it added, on a diverging map centred at zero.
                log["eval/recon_pred"] = wandb.Image(pred_only.clip(0, 1))
                log["eval/residual"] = wandb.Image(residual_rgb(recon - pred_only))
            wandb.log(log, step=step)

        if eval_tensor is not None and step % args.eval_tensor_every == 0:
            m = sync_eval_model()
            m.eval()
            with torch.no_grad():
                tmetrics = eval_tensor_codec(
                    m,
                    args.d,
                    args,
                    eval_tensor,
                    eval_tensor_eb,
                    device,
                    eval_tensor_ckpt,
                )
            if m is model:
                model.train()
            wandb.log(tmetrics, step=step)
            last_tensor_metrics = tmetrics

        if args.save_every and step % args.save_every == 0:
            checkpoint = run_dir / f"{out.stem}-step-{step:06d}{out.suffix}"
            torch.save(
                {
                    "state_dict": sync_eval_model().state_dict(),
                    "d": args.d,
                    "agg_level": args.agg_level,
                    "version": CKPT_VERSION,
                },
                checkpoint,
            )
            last_saved_step = step

        if step % 100 == 0:
            postfix = {"bpp": f"{run_loss / 100:.5f}", "eval_bpp": f"{last_eval:.5f}"}
            if last_tensor_metrics is not None:
                postfix["tensor_bpv"] = (
                    f"{last_tensor_metrics['eval_tensor/bits_per_value']:.3f}"
                )
                postfix["tensor_psnr"] = (
                    f"{last_tensor_metrics['eval_tensor/psnr_db']:.2f}"
                )
            if last_saved_step is not None:
                postfix["saved"] = last_saved_step
            bar.set_postfix(postfix)
            run_loss = 0.0

    if prof is not None:
        prof.stop()
        sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
        print(prof.key_averages().table(sort_by=sort_key, row_limit=25))
        # self-CPU view: cudaLaunchKernel/dispatch on top = launch-bound
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
        prof.export_chrome_trace("trace.json")
        print("wrote trace.json (open in chrome://tracing or perfetto.dev)")
        return

    torch.save(
        {
            "state_dict": sync_eval_model().state_dict(),
            "d": args.d,
            "agg_level": args.agg_level,
            "version": CKPT_VERSION,
        },
        out,
    )
    print(f"saved {out}")
    wandb.save(str(out))
    wandb.finish()


if __name__ == "__main__":
    main()
