"""Scaling study: memory and wall time of the GNN codec vs. data size.

Sweeps synthetic random tensors of rank 3, 4 and 5 over a range of sizes,
running compress + decompress with the codec's *default* settings, and plots

    peak GPU memory / peak host RSS / compress time / decompress time

against the input size in MiB on linear axes (y starts at 0), one line per rank.
One run per point: the codec is deterministic at a fixed shape.

Each measurement runs in a fresh subprocess so peak host RSS and peak CUDA
allocation are attributable to that single configuration.

    python scripts/plot_scaling.py --out outputs/scaling

Re-plot without re-measuring:

    python scripts/plot_scaling.py --from-json outputs/scaling.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_CKPT = ROOT / "checkpoints" / "v7-d64-1agg.pt"
# Schedule depth is fixed per rank at the codec's tuned rank level, so the anchor
# stride -- and with it the chunk edge -- does not vary with tensor size. The one
# exception is a tensor smaller than a single chunk: there a full-depth schedule
# has nothing to anchor, so the level drops to the codec's size-capped value.
# ``levels="auto"`` would instead apply that size cap at every size, which is why
# a 248**3 input ran at level 6 while 96**3 ran at level 5.
RANKS = (3, 4, 5)
# Target point counts (millions) common to every rank, so the x-axes overlap.
# Evenly spaced input sizes in MiB (float32). Starting well above the
# auto-chunk budget keeps every shape in the many-chunks regime, so no
# point sits in the sub-chunk region where the schedule depth drops.
TARGET_MIB = (20.0, 83.3, 146.7, 210.0, 273.3, 336.7, 400.0)

# dataviz categorical slots 1-3 (light / dark), see references/palette.md
SERIES_LIGHT = {3: "#2a78d6", 4: "#eb6834", 5: "#1baf7a"}

# one marker shape per repeat (seed), reused across ranks: the colour carries the
# rank, the shape carries the seed, so both are readable on the same axes.
SEED_MARKERS = ("o", "^", "s", "D", "v", "P")

# Codec comparison: categorical slots 1-4, in the palette's fixed order, plus a
# marker per codec so identity never rests on colour alone.
CODEC_COLOR = {
    "gnn": "#2a78d6", "interp-linear": "#eb6834", "sz3": "#1baf7a",
    "sperr": "#eda100", "interp-cubic": "#e87ba4",
    "gnn-interp-linear": "#4a3aa7", "gnn-interp-cubic": "#e34948",
    "tthresh": "#a5713a",
}
CODEC_MARKER = {
    "gnn": "o", "interp-linear": "s", "sz3": "^", "sperr": "D",
    "interp-cubic": "P", "gnn-interp-linear": "v", "gnn-interp-cubic": "X",
    "tthresh": "*",
}
CODEC_LABEL = {
    "gnn": "GNN (chunked, GPU)", "interp-linear": "interp linear (CPU)",
    "interp-cubic": "interp cubic (CPU)", "sz3": "SZ3 (CPU)", "sperr": "SPERR (CPU)",
    "gnn-interp-linear": "same pipeline, linear interp (GPU)",
    "gnn-interp-cubic": "same pipeline, cubic interp (GPU)",
    "tthresh": "TTHRESH (CPU, OpenMP)",
}


CODECS = (
    "gnn",
    "gnn-interp-linear",  # the GNN pipeline with the model swapped out (ablation)
    "gnn-interp-cubic",
    "interp-linear",
    "interp-cubic",
    "sz3",
    "sperr",
    "tthresh",
)
# Ranks these baselines can encode natively; anything deeper is folded (leading
# axes merged) the way a user would have to feed them the data. SZ3 tops out at
# 4-D, SPERR at 3-D volumes.
CODEC_MAX_RANK = {"sz3": 4, "sperr": 3}

# TTHRESH ships as a CLI, not a library: build it once (see its README) and point
# the sweep at the binary with --tthresh-bin or $LATEC_TTHRESH_BIN.
TTHRESH_BIN = os.environ.get("LATEC_TTHRESH_BIN", "")


def agg_level(checkpoint) -> int:
    """Neighbourhood aggregation level frozen into the checkpoint."""
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return int(ckpt.get("agg_level", 2))


def fold_shape(shape: tuple[int, ...], max_rank: int) -> tuple[int, ...]:
    """Merge leading axes until the shape has at most ``max_rank`` of them."""
    if len(shape) <= max_rank:
        return tuple(shape)
    keep = max_rank - 1
    lead = int(np.prod(shape[: len(shape) - keep]))
    return (lead,) + tuple(shape[len(shape) - keep :])


def rank_chunk_edge(ndim: int, checkpoint) -> int:
    """The chunk edge the codec picks for this rank, at its tuned rank level.

    Read back from the codec (on a shape so large its size guards cannot bind)
    so the sweep's shapes stay in step with the auto-chunking rule.
    """
    from latec.gnn_codec import _auto_chunk_edges, _auto_levels

    levels = _auto_levels((1 << 20,) * ndim, agg_level(checkpoint))
    return int(_auto_chunk_edges((1 << 20,) * ndim, 1 << levels)[0])


def shape_for(ndim: int, mib: float, unit: int) -> tuple[int, ...]:
    """Near-cubic shape of ~mib MiB whose every axis is a multiple of ``unit``.

    Axes that are exact multiples of the chunk edge keep the partition flush:
    the last chunk of an axis is a full one, so no axis contributes a row of
    near-empty sliver chunks. That matters because cost per chunk is nearly
    independent of a chunk's extent (same schedule of kernel launches) while the
    cached geometry is keyed by chunk *shape*, so a one-cell tail both inflates
    the chunk count and collapses the cache -- time and memory then jump in
    opposite directions and the size scaling stops being about size.

    The shape is the near-cube closest to the target: ``r`` axes get one extra
    unit so the volume can land between successive cubes.
    """
    target = mib * 2**20 / 4.0 / unit**ndim  # in units**ndim
    k = max(2, int(round(target ** (1.0 / ndim))))
    best = None
    for base in (k - 1, k):
        if base < 2:
            continue
        for r in range(ndim):
            vol = (base + 1) ** r * base ** (ndim - r)
            score = abs(vol - target)
            if best is None or score < best[0]:
                best = (score, base, r)
    _, base, r = best
    return tuple(unit * (base + 1 if a < r else base) for a in range(ndim))


def fixed_levels(shape: tuple[int, ...], agg_level: int) -> int:
    """Rank-fixed schedule depth, relaxed only when the tensor is under one chunk.

    The rank level is read back from the codec itself (``_auto_levels`` on a huge
    shape, where its size guard cannot bind) so the two stay in sync. If the
    tensor is strictly smaller than the chunk that level implies, it cannot carry
    a full-depth schedule, and the codec's size-capped level is used instead.
    """
    from latec.gnn_codec import _auto_chunk_edges, _auto_levels

    ndim = len(shape)
    rank_levels = _auto_levels((1 << 20,) * ndim, agg_level)  # size guard inactive
    edges = _auto_chunk_edges(shape, 1 << rank_levels)
    if any(n < e for n, e in zip(shape, edges)):  # tensor smaller than one chunk
        return min(rank_levels, _auto_levels(shape, agg_level))
    return rank_levels


def make_field(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Smooth-ish separable-cosine field plus noise (compressible, realistic)."""
    rng = np.random.default_rng(seed)
    # float32 end to end: rand() in float64 then astype would peak at 3x the
    # field, which would show up in the measured host peak as codec cost.
    x = rng.random(shape, dtype=np.float32)
    x *= 0.05
    for k, s in enumerate(shape):
        phase = float(rng.uniform(0, 2 * np.pi))
        freq = float(rng.uniform(2.0, 6.0))
        wave = np.cos(np.linspace(0, freq * np.pi, s, dtype=np.float32) + phase)
        x += wave.reshape([-1 if i == k else 1 for i in range(len(shape))])
    return x


# --------------------------------------------------------------------------- #
# worker: one (shape, seed) measurement, JSON on stdout
# --------------------------------------------------------------------------- #
def run_worker(args) -> None:
    """One measurement, dispatched by codec; JSON record on stdout."""
    if args.codec in ("sz3", "sperr"):
        rec = run_baseline_worker(args)
    elif args.codec == "tthresh":
        rec = run_tthresh_worker(args)
    elif args.codec.startswith("interp"):
        rec = run_interp_worker(args)
    else:
        rec = run_gnn_worker(args)
    rec["codec"] = args.codec
    print("@@JSON@@" + json.dumps(rec))


def run_baseline_worker(args) -> dict:
    """SZ3 / SPERR through imagecodecs, at the same range-relative bound.

    Both are whole-field CPU codecs, so there is no chunking to report and the
    GPU columns stay zero. Ranks past what they accept are folded, and the error
    is still checked against the original n-D field: folding only relabels the
    axes, it never changes which values are compared.
    """
    import imagecodecs

    shape = tuple(args.shape)
    x = make_field(shape, args.seed)
    span = float(x.max()) - float(x.min())
    work = np.ascontiguousarray(x.reshape(fold_shape(shape, CODEC_MAX_RANK[args.codec])))

    t0 = time.perf_counter()
    if args.codec == "sz3":
        stream = imagecodecs.sz3_encode(
            work, mode=imagecodecs.SZ3.MODE.REL, rel=args.eb
        )
    else:  # SPERR has no relative mode: convert to the point-wise bound
        stream = imagecodecs.sperr_encode(
            work, mode=imagecodecs.SPERR.MODE.PWE, level=args.eb * span
        )
    t1 = time.perf_counter()
    decode = imagecodecs.sz3_decode if args.codec == "sz3" else imagecodecs.sperr_decode
    y = decode(stream, shape=work.shape, dtype=work.dtype)
    t2 = time.perf_counter()

    y = np.asarray(y, dtype=np.float32).reshape(shape)
    return {
        "ndim": len(shape),
        "shape": list(shape),
        "coded_shape": list(work.shape),
        "seed": args.seed,
        "points": int(np.prod(shape)),
        "levels": 0,
        "chunk_edges": list(shape),
        "n_chunks": 1,
        "in_bytes": int(x.nbytes),
        "stream_bytes": len(stream),
        "encode_s": t1 - t0,
        "decode_s": t2 - t1,
        "gpu_peak_encode": 0,
        "gpu_peak_decode": 0,
        "gpu_peak": 0,
        "rss_peak": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "max_abs_err": float(np.abs(y - x).max()),
        "device": "cpu",
    }


# TTHRESH targets a *global* accuracy (relative L2, RMSE or PSNR) and gives no
# point-wise guarantee, so it cannot simply be handed this sweep's bound: its
# -e target has to be tuned until the achieved max error clears eb*(max-min).
# That tuning is an oracle the other codecs never get -- they meet the bound by
# construction, in one pass -- so it is kept out of the reported time: --tthresh-eps
# carries a target calibrated once per rank (on a small field of that rank),
# and a run that still misses the bound tightens it and retries.
_TT_PROBES = 6  # round trips the search may spend before it reports
_TT_TOL = 1.05  # stop once the bracket on -e is this tight


def _tthresh_roundtrip(binary, raw, comp, out, shape, eps, env):
    """One compress + decompress at target ``eps``; (enc_s, dec_s, bytes, recon)."""
    # -s is documented as Fortran order (first axis fastest), so the C-order
    # shape goes in reversed: same bytes, axes labelled the way they are stored.
    enc = [binary, "-i", raw, "-t", "float", "-s", *[str(n) for n in shape[::-1]],
           "-e", repr(float(eps)), "-c", comp]
    t0 = time.perf_counter()
    subprocess.run(enc, check=True, capture_output=True, env=env)
    t1 = time.perf_counter()
    subprocess.run([binary, "-c", comp, "-o", out], check=True,
                   capture_output=True, env=env)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, os.path.getsize(comp), np.fromfile(out, dtype=np.float32)


def tthresh_search(binary, raw, comp, out, shape, x, bound, eps0, env, probes):
    """Largest -e target whose max error still clears ``bound``; (record, probes).

    Bracket then bisect in log space. First valid probe wins is not enough: the
    max error jumps at the rank-truncation cliff, so two targets 15% apart can
    differ by 100x in rate, and where a fixed start lands relative to that cliff
    is an accident of the field, not a property of the codec.
    """
    eps = eps0
    lo = hi = None  # largest valid target / smallest invalid one
    best, used = None, 0
    for used in range(1, probes + 1):
        enc_s, dec_s, nbytes, flat = _tthresh_roundtrip(
            binary, raw, comp, out, shape, eps, env
        )
        err = float(np.abs(flat.reshape(shape) - x).max())
        del flat
        if err <= bound:
            lo = eps
            best = {"stream_bytes": nbytes, "encode_s": enc_s, "decode_s": dec_s,
                    "max_abs_err": err, "eps": eps}
        else:
            hi = eps if hi is None else min(hi, eps)
        if lo is not None and hi is not None and hi / lo <= _TT_TOL:
            break
        if lo is None:  # nothing valid yet: tighten the target
            eps /= 1.5
        elif hi is None:  # nothing invalid yet: loosen it until it breaks
            eps *= 1.5
        else:
            eps = float(np.sqrt(lo * hi))
    if best is None:
        raise SystemExit(f"tthresh missed the bound at every target (last {eps:g})")
    return best, used


def run_tthresh_worker(args) -> dict:
    """TTHRESH (Tucker/HOSVD + bit-plane coding) through its command line.

    Whole-field and CPU-only, like the other baselines, but the only arm that
    threads: it is measured as it ships (OpenMP over every core unless
    --tthresh-threads says otherwise), and the record carries the thread count.
    """
    import shutil
    import tempfile

    binary = args.tthresh_bin or TTHRESH_BIN
    if not binary:
        raise SystemExit("--tthresh-bin (or $LATEC_TTHRESH_BIN) is required")

    shape = tuple(args.shape)
    x = make_field(shape, args.seed)
    span = float(x.max()) - float(x.min())
    bound = args.eb * (span if span > 0 else 1.0)

    env = dict(os.environ)
    if args.tthresh_threads > 0:
        env["OMP_NUM_THREADS"] = str(args.tthresh_threads)
    threads = int(env.get("OMP_NUM_THREADS", os.cpu_count() or 1))

    work = tempfile.mkdtemp(prefix="tthresh-", dir=args.tthresh_dir or None)
    try:
        raw, comp, out = (f"{work}/in.raw", f"{work}/c.tt", f"{work}/o.raw")
        x.tofile(raw)
        best, iters = tthresh_search(
            binary, raw, comp, out, shape, x, bound,
            args.tthresh_eps if args.tthresh_eps > 0 else args.eb,
            env, args.tthresh_probes,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)

    rss = max(
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss),
    ) * 1024
    return {
        "ndim": len(shape),
        "shape": list(shape),
        "coded_shape": list(shape),
        "seed": args.seed,
        "points": int(np.prod(shape)),
        "levels": 0,
        "chunk_edges": list(shape),
        "n_chunks": 1,
        "in_bytes": int(x.nbytes),
        "stream_bytes": int(best["stream_bytes"]),
        "encode_s": best["encode_s"],
        "decode_s": best["decode_s"],
        "gpu_peak_encode": 0,
        "gpu_peak_decode": 0,
        "gpu_peak": 0,
        "rss_peak": rss,
        "max_abs_err": best["max_abs_err"],
        "device": "cpu",
        "tthresh_eps": best["eps"],
        "tthresh_iters": iters,
        "threads": threads,
    }


def run_interp_worker(args) -> dict:
    """LATEC's own pipeline with the GNN swapped for SZ-style interpolation.

    Same quantizer, same stage schedule, same entropy back end -- only the
    predictor differs -- so this isolates what the model buys. The interpolation
    path is whole-field and CPU-only: no chunking, no device memory.
    """
    from latec.codec import compress, decompress
    from latec.predictor import InterpPredictor

    shape = tuple(args.shape)
    x = make_field(shape, args.seed)
    levels = args.levels if args.levels > 0 else fixed_levels(shape, agg_level(args.checkpoint))
    stride = 1 << levels
    order = "linear" if args.codec == "interp-linear" else "cubic"
    span = float(x.max()) - float(x.min())
    eb_abs = args.eb * (span if span > 0 else 1.0)
    predictor = InterpPredictor(order, levels, stride, 1)

    t0 = time.perf_counter()
    stream, stats = compress(
        x, eb_abs, predictor, levels=levels, anchor_stride=stride, anchor_block=1
    )
    t1 = time.perf_counter()
    del stats  # holds the encoder's reconstruction; the decode peak is measured
    y = decompress(
        stream,
        lambda h: InterpPredictor(order, h.levels, h.anchor_stride, h.anchor_block),
    )
    t2 = time.perf_counter()

    y = np.asarray(y, dtype=np.float32).reshape(shape)
    return {
        "ndim": len(shape),
        "shape": list(shape),
        "seed": args.seed,
        "points": int(np.prod(shape)),
        "levels": int(levels),
        "chunk_edges": list(shape),
        "n_chunks": 1,
        "in_bytes": int(x.nbytes),
        "stream_bytes": len(stream),
        "encode_s": t1 - t0,
        "decode_s": t2 - t1,
        "gpu_peak_encode": 0,
        "gpu_peak_decode": 0,
        "gpu_peak": 0,
        "rss_peak": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "max_abs_err": float(np.abs(y - x).max()),
        "device": "cpu",
    }


def run_gnn_worker(args) -> dict:
    import torch

    from latec import GNNCompressorCodec
    from latec import gnn_predictor as gp
    from latec.gnn_codec import _auto_chunk_edges

    shape = tuple(args.shape)
    x = make_field(shape, args.seed)
    codec = GNNCompressorCodec(args.checkpoint)

    # Always take the chunked path. Auto chunking only kicks in above
    # _AUTO_CHUNK_THRESHOLD points, so below it we hand the codec the very edges
    # its own auto rule would pick; every other setting stays at the default.
    agg = gp._load_inference_model(codec.checkpoint_path, torch, codec.device)[3]
    levels = args.levels if args.levels > 0 else fixed_levels(shape, agg)
    stride = 1 << levels
    if args.chunk_edge:  # fixed edge: many real chunks at every size
        e = max(stride, args.chunk_edge // stride * stride)
        chunk_edges = tuple(min(e, max(stride, -(-n // stride) * stride)) for n in shape)
    else:
        chunk_edges = _auto_chunk_edges(shape, stride)
    n_chunks = int(np.prod([-(-n // e) for n, e in zip(shape, chunk_edges)]))

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    # fixed levels + chunked path; every other setting stays at the default
    stream = codec.compress(
        x,
        error_bound=args.eb,
        levels=levels,
        chunk_size=chunk_edges,
        predictor=args.codec.removeprefix("gnn-") if args.codec != "gnn" else "gnn",
    )
    if cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    gpu_enc = torch.cuda.max_memory_allocated() if cuda else 0
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    y = codec.uncompress(stream)
    if cuda:
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    gpu_dec = torch.cuda.max_memory_allocated() if cuda else 0

    y = np.asarray(y.cpu() if hasattr(y, "cpu") else y, dtype=np.float32)
    return {
        "ndim": len(shape),
        "shape": list(shape),
        "seed": args.seed,
        "points": int(np.prod(shape)),
        "levels": int(levels),
        "chunk_edges": list(chunk_edges),
        "n_chunks": n_chunks,
        "in_bytes": int(x.nbytes),
        "stream_bytes": len(stream),
        "encode_s": t1 - t0,
        "decode_s": t2 - t1,
        "gpu_peak_encode": int(gpu_enc),
        "gpu_peak_decode": int(gpu_dec),
        "gpu_peak": int(max(gpu_enc, gpu_dec)),
        "rss_peak": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "max_abs_err": float(np.abs(y - x).max()),
        "device": "cuda" if cuda else "cpu",
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def pick_gpu() -> str | None:
    """Index of the least-used visible GPU, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except Exception:
        return None
    best, best_key = None, None
    for line in out.strip().splitlines():
        idx, mem, util = (f.strip() for f in line.split(","))
        key = (int(mem), int(util))
        if best_key is None or key < best_key:
            best, best_key = idx, key
    return best


def measure(args) -> list[dict]:
    # Everything on the "gnn" prefix runs the chunked GPU pipeline (including the
    # predictor ablation); the standalone baselines are CPU codecs.
    cpu_only = not args.codec.startswith("gnn")
    gpu = None if cpu_only else (args.gpu if args.gpu != "auto" else pick_gpu())
    env = dict(os.environ)
    if cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # imagecodecs lives in its own interpreter (the SZ3/SPERR-enabled wheel)
    worker_python = (
        args.baseline_python
        if args.codec in ("sz3", "sperr") and args.baseline_python
        else sys.executable
    )
    print(f"codec={args.codec} python={worker_python} "
          f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    records: list[dict] = []
    for ndim in args.ranks:
        unit = args.chunk_edge or rank_chunk_edge(ndim, args.checkpoint)
        seen: set[tuple[int, ...]] = set()
        for mib in args.mib:
            shape = list(shape_for(ndim, mib, unit))
            if tuple(shape) in seen:  # rounding collapsed two targets onto one shape
                continue
            seen.add(tuple(shape))
            for rep in range(args.rep_offset, args.rep_offset + args.reps):
                cmd = [worker_python, str(Path(__file__).resolve()), "--worker",
                       "--codec", args.codec,
                       "--eb", repr(args.eb),
                       "--chunk-edge", str(args.chunk_edge),
                       "--levels", str(max(args.levels, 0)),
                       "--tthresh-bin", str(args.tthresh_bin or ""),
                       "--tthresh-eps", repr(args.tthresh_eps),
                       "--tthresh-probes", str(args.tthresh_probes),
                       "--tthresh-threads", str(args.tthresh_threads),
                       "--tthresh-dir", str(args.tthresh_dir or ""),
                       "--shape", *map(str, shape),
                       "--seed", str(1000 * ndim + 10 * rep + 1),
                       "--checkpoint", str(args.checkpoint)]
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                tag = f"{args.codec} {ndim}D {'x'.join(map(str, shape))} rep{rep}"
                line = next(
                    (ln for ln in proc.stdout.splitlines() if ln.startswith("@@JSON@@")),
                    None,
                )
                if proc.returncode != 0 or line is None:
                    print(f"  FAILED {tag} ({time.time() - t0:.0f}s)")
                    print("   " + (proc.stderr.strip().splitlines() or ["<no stderr>"])[-1])
                    continue
                rec = json.loads(line[len("@@JSON@@"):])
                records.append(rec)
                print(
                    f"  {tag}: {rec['in_bytes'] / 2**20:7.1f} MiB  "
                    f"enc {rec['encode_s']:6.2f}s  dec {rec['decode_s']:6.2f}s  "
                    f"gpu {rec['gpu_peak'] / 2**20:7.1f} MiB  "
                    f"rss {rec['rss_peak'] / 2**20:7.1f} MiB  "
                    f"chunks {rec['n_chunks']:5d}  L{rec['levels']}  "
                    f"err {rec['max_abs_err']:.2e}"
                )
    return records


def aggregate(records: list[dict], key: str):
    """{ndim: (size_MiB, mean, lo, hi, n_chunks)} by size; band = min/max U +/-1 sd."""
    out = {}
    for ndim in sorted({r["ndim"] for r in records}):
        rows = [r for r in records if r["ndim"] == ndim]
        sizes = sorted({r["in_bytes"] for r in rows})
        xs, mean, lo, hi, pts = [], [], [], [], []
        for s in sizes:
            vals = np.array([r[key] for r in rows if r["in_bytes"] == s], float)
            m, sd = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
            xs.append(s / 2**20)
            pts.append([r["n_chunks"] for r in rows if r["in_bytes"] == s][0])
            mean.append(m)
            lo.append(max(min(vals.min(), m - sd), 1e-12))
            hi.append(max(vals.max(), m + sd))
        out[ndim] = (
            np.array(xs), np.array(mean), np.array(lo), np.array(hi), np.array(pts)
        )
    return out

def fit_affine(records: list[dict], key: str) -> dict[int, dict]:
    """Least-squares y = a + b*x per rank, over every individual run.

    x is the input size in MiB, y the metric in its raw unit. Fitting the runs
    (not the per-size means) weights each size by its repeat count, which is
    uniform here, so the two agree; the residual spread is the run-to-run one.
    """
    out = {}
    for ndim in sorted({r["ndim"] for r in records}):
        rows = [r for r in records if r["ndim"] == ndim]
        x = np.array([r["in_bytes"] / 2**20 for r in rows], float)
        y = np.array([r[key] for r in rows], float)
        if len(np.unique(x)) < 2:
            continue
        b, a = np.polyfit(x, y, 1)
        resid = y - (a + b * x)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out[ndim] = {
            "intercept": float(a),
            "slope": float(b),
            "r2": 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
            "rmse": float(np.sqrt((resid**2).mean())),
            "n": len(rows),
        }
    return out


FIT_PANELS = [
    ("gpu_peak", 2**20, "MiB", "MiB/MiB"),
    ("rss_peak", 2**20, "MiB", "MiB/MiB"),
    ("encode_s", 1.0, "s", "s/MiB"),
    ("decode_s", 1.0, "s", "s/MiB"),
]


def write_fits(records: list[dict], out_csv: Path) -> None:
    """Affine-fit coefficients per rank per metric, and echo them to stdout."""
    rows = []
    for key, scale, unit, slope_unit in FIT_PANELS:
        for ndim, f in fit_affine(records, key).items():
            rows.append({
                "metric": key, "ndim": ndim,
                "intercept": f["intercept"] / scale, "intercept_unit": unit,
                "slope": f["slope"] / scale, "slope_unit": slope_unit,
                "r2": f["r2"], "rmse": f["rmse"] / scale, "n_runs": f["n"],
            })
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    for r in rows:
        print(f"  {r['metric']:<10} {r['ndim']}D: "
              f"{r['intercept']:9.3f} {r['intercept_unit']} + "
              f"{r['slope']:.5f} {r['slope_unit']} * size   "
              f"R2={r['r2']:.4f} rmse={r['rmse']:.3f} n={r['n_runs']}")


CSV_COLUMNS = [
    "codec", "ndim", "edge", "shape", "points", "levels", "chunk_edge", "chunk_points",
    "n_chunks", "in_bytes", "in_MiB", "stream_bytes", "ratio", "bits_per_point",
    "encode_s", "decode_s", "enc_Mpts_per_s", "dec_Mpts_per_s",
    "gpu_peak_encode", "gpu_peak_decode", "gpu_peak", "gpu_peak_MiB",
    "rss_peak", "rss_peak_MiB", "max_abs_err", "seed", "device",
]


def csv_rows(records: list[dict]) -> list[dict]:
    """One tidy row per run: raw fields plus the usual derived quantities."""
    rows = []
    for r in records:
        n = r["points"]
        chunk_pts = int(np.prod(r["chunk_edges"]))
        rows.append({
            "codec": r.get("codec", "gnn"),
            "ndim": r["ndim"],
            "edge": r["shape"][0],
            "shape": "x".join(str(v) for v in r["shape"]),
            "points": n,
            "levels": r["levels"],
            "chunk_edge": r["chunk_edges"][0],
            "chunk_points": chunk_pts,
            "n_chunks": r["n_chunks"],
            "in_bytes": r["in_bytes"],
            "in_MiB": r["in_bytes"] / 2**20,
            "stream_bytes": r["stream_bytes"],
            "ratio": r["in_bytes"] / r["stream_bytes"],
            "bits_per_point": 8 * r["stream_bytes"] / n,
            "encode_s": r["encode_s"],
            "decode_s": r["decode_s"],
            "enc_Mpts_per_s": n / r["encode_s"] / 1e6,
            "dec_Mpts_per_s": n / r["decode_s"] / 1e6,
            "gpu_peak_encode": r["gpu_peak_encode"],
            "gpu_peak_decode": r["gpu_peak_decode"],
            "gpu_peak": r["gpu_peak"],
            "gpu_peak_MiB": r["gpu_peak"] / 2**20,
            "rss_peak": r["rss_peak"],
            "rss_peak_MiB": r["rss_peak"] / 2**20,
            "max_abs_err": r["max_abs_err"],
            "seed": r["seed"],
            "device": r.get("device", ""),
        })
    rows.sort(key=lambda d: (d["ndim"], d["points"]))
    return rows


def write_csv(records: list[dict], out_csv: Path) -> None:
    """Long-format CSV, one row per measured run, for regressions/re-plotting."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(csv_rows(records))
    print(f"wrote {out_csv}")


def plot(records: list[dict], out_png: Path, note: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("gpu_peak", "Peak GPU memory", "MiB", 2**20),
        ("rss_peak", "Peak host memory (RSS)", "MiB", 2**20),
        ("encode_s", "Compress time", "s", 1.0),
        ("decode_s", "Decompress time", "s", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.6), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.085, 1, 0.885))
    fig.patch.set_facecolor("#fcfcfb")

    slope_units = {k: su for k, _, _, su in FIT_PANELS}
    for ax, (key, title, unit, scale) in zip(axes.ravel(), panels):
        ax.set_facecolor("#fcfcfb")
        fits = fit_affine(records, key)
        agg = aggregate(records, key)
        for ndim, (xs, mean, lo, hi, pts) in agg.items():
            c = SERIES_LIGHT.get(ndim, "#52514e")
            # every individual run, one marker shape per seed, so a single
            # outlying repeat is attributable rather than hidden in a band
            runs = [r for r in records if r["ndim"] == ndim]
            for si, seed in enumerate(sorted({r["seed"] for r in runs})):
                sr = [r for r in runs if r["seed"] == seed]
                ax.scatter([r["in_bytes"] / 2**20 for r in sr],
                           [r[key] / scale for r in sr],
                           s=26, marker=SEED_MARKERS[si % len(SEED_MARKERS)],
                           facecolors="none", edgecolors=c, linewidths=1.0,
                           alpha=0.9, zorder=5)
            if np.any(hi > lo):
                ax.fill_between(xs, lo / scale, hi / scale, color=c, alpha=0.13, lw=0)
            ax.plot(xs, mean / scale, color=c, lw=1.8, zorder=3)
            f = fits.get(ndim)
            if f is not None:  # affine fit, drawn back to x=0 to show the intercept
                xf = np.array([0.0, xs.max() * 1.02])
                ax.plot(xf, (f["intercept"] + f["slope"] * xf) / scale, color=c,
                        lw=1.1, ls="--", alpha=0.85, zorder=4,
                        label=f"{ndim}D  {f['intercept'] / scale:.0f} {unit} + "
                              f"{f['slope'] / scale:.3g} {slope_units[key]}"
                              f"   R\u00b2={f['r2']:.3f}")
            # small fixed vertical stagger so line-end labels never overlap
            dy = {3: -7.0, 4: 7.0}.get(ndim, 0.0)
            ax.annotate(f"{ndim}D", (xs[-1], mean[-1] / scale), color=c,
                        fontsize=9, fontweight="bold",
                        xytext=(5, dy), textcoords="offset points", va="center")
        fit_leg = ax.legend(
            frameon=False, fontsize=7.5, labelcolor="#52514e",
            loc="upper left", handlelength=1.6, borderaxespad=0.2,
            title="affine fit: y = a + b\u00b7size", title_fontsize=7.5)
        if ax is axes.ravel()[0]:  # marker key: shape -> repeat, drawn once
            from matplotlib.lines import Line2D

            n_seeds = max(len({r["seed"] for r in records if r["ndim"] == nd})
                          for nd in {r["ndim"] for r in records})
            handles = [
                Line2D([], [], ls="none", marker=SEED_MARKERS[i % len(SEED_MARKERS)],
                       mfc="none", mec="#52514e", ms=5.5, label=f"seed {i + 1}")
                for i in range(n_seeds)
            ]
            ax.legend(handles=handles, frameon=False, fontsize=7.5,
                      labelcolor="#52514e", loc="lower right", handlelength=1.2,
                      borderaxespad=0.4, title="repeat", title_fontsize=7.5)
            ax.add_artist(fit_leg)  # keep the fit legend the second call displaced
        ax.margins(x=0.08, y=0.12)
        ax.autoscale_view()
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=11, color="#0b0b0b", loc="left")
        ax.set_xlabel("input size (MiB, float32)", fontsize=9, color="#52514e")
        ax.set_ylabel(unit, fontsize=9, color="#52514e")
        ax.grid(True, which="major", color="#d9d8d3", lw=0.6)
        ax.grid(True, which="minor", color="#ececE7", lw=0.4)
        ax.tick_params(colors="#52514e", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#d9d8d3")

    dev = records[0].get("device", "?") if records else "?"
    lv_by_rank = {r["ndim"]: max(x["levels"] for x in records if x["ndim"] == r["ndim"])
                  for r in records}
    fig.suptitle(
        f"GNN codec scaling on random fields ({dev}, chunked path, "
        f"fixed levels per rank, codec defaults otherwise)",
        fontsize=12.5, color="#0b0b0b", x=0.01, y=0.985, ha="left",
    )
    reps = max(
        sum(1 for r in records if (r["ndim"], r["in_bytes"]) == k)
        for k in {(r["ndim"], r["in_bytes"]) for r in records}
    )
    head = (
        f"{reps} repeats per point (distinct field seeds); every run is drawn, one "
        "marker shape per seed, with the solid line through the per-size mean and "
        "the band its min/max."
        if reps > 1
        else "One run per point (the codec is deterministic at a fixed shape)."
    )
    caption = (
        f"{head} Linear axes, y starting at 0. Chunked path at every size, schedule "
        "depth fixed per rank "
        "(" + ", ".join(f"{k}D L{v}" for k, v in sorted(lv_by_rank.items())) + "); every "
        "size here is many chunks, so no point falls in the sub-chunk regime."
    )
    if note:
        caption += " " + note
    fig.text(
        0.01, 0.012, "\n".join(textwrap.wrap(caption, 165)),
        fontsize=8.5, color="#52514e", ha="left", va="bottom", linespacing=1.5,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    print(f"wrote {out_png}")


COMPARE_PANELS = [
    ("encode_s", "Compress time", "s", 1.0, True),
    ("decode_s", "Decompress time", "s", 1.0, True),
    ("rss_peak", "Peak host memory (RSS)", "MiB", 2**20, True),
    ("gpu_peak", "Peak GPU memory", "MiB", 2**20, True),
    ("ratio", "Compression ratio", "x", 1.0, False),
]


def _with_ratio(records: list[dict]) -> list[dict]:
    """Add the derived size ratio so it can be plotted like any other metric."""
    for r in records:
        r.setdefault("ratio", r["in_bytes"] / r["stream_bytes"])
        r.setdefault("codec", "gnn")
    return records


def plot_compare(records: list[dict], out_png: Path, note: str | None = None) -> None:
    """One row per rank, one column per metric, one colour+marker per codec."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = _with_ratio(records)
    ranks = sorted({r["ndim"] for r in records})
    codecs = [c for c in CODECS if any(r["codec"] == c for r in records)]
    fig, axes = plt.subplots(
        len(ranks), len(COMPARE_PANELS),
        figsize=(4.0 * len(COMPARE_PANELS), 3.3 * len(ranks) + 1.6),
        constrained_layout=True,
    )
    fig.patch.set_facecolor("#fcfcfb")
    axes = np.atleast_2d(axes)

    for row, ndim in enumerate(ranks):
        for col, (key, title, unit, scale, do_fit) in enumerate(COMPARE_PANELS):
            ax = axes[row, col]
            ax.set_facecolor("#fcfcfb")
            for codec in codecs:
                runs = [
                    r for r in records if r["ndim"] == ndim and r["codec"] == codec
                ]
                if not runs:
                    continue
                c = CODEC_COLOR[codec]
                agg = aggregate(runs, key)[ndim]
                xs, mean, lo, hi, _ = agg
                ax.scatter([r["in_bytes"] / 2**20 for r in runs],
                           [r[key] / scale for r in runs], s=14,
                           marker=CODEC_MARKER[codec], facecolors="none",
                           edgecolors=c, linewidths=0.9, alpha=0.85, zorder=5)
                if np.any(hi > lo):
                    ax.fill_between(xs, lo / scale, hi / scale, color=c,
                                    alpha=0.13, lw=0)
                ax.plot(xs, mean / scale, color=c, lw=2,
                        label=CODEC_LABEL[codec] if (row, col) == (0, 0) else None,
                        zorder=4)
                if do_fit:
                    f = fit_affine(runs, key).get(ndim)
                    if f is not None:
                        xf = np.array([0.0, xs.max() * 1.02])
                        ax.plot(xf, (f["intercept"] + f["slope"] * xf) / scale,
                                color=c, lw=1.0, ls="--", alpha=0.8, zorder=3)
            ax.margins(x=0.08, y=0.12)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.set_title(f"{ndim}D  {title}", fontsize=10, color="#0b0b0b", loc="left")
            ax.set_ylabel(unit, fontsize=8.5, color="#52514e")
            if row == len(ranks) - 1:
                ax.set_xlabel("input size (MiB, float32)", fontsize=8.5,
                              color="#52514e")
            ax.grid(True, which="major", color="#d9d8d3", lw=0.6)
            ax.tick_params(colors="#52514e", labelsize=8)
            for sp in ax.spines.values():
                sp.set_color("#d9d8d3")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", frameon=False,
               ncols=len(codecs), fontsize=9.5, labelcolor="#52514e")
    fig.suptitle(
        "Error-bounded compressors on the same random fields "
        "(same shapes, same seeds, same range-relative bound)",
        fontsize=13, color="#0b0b0b", x=0.01, ha="left",
    )
    if note:
        fig.supxlabel("\n".join(textwrap.wrap(note, 200)), fontsize=8.5,
                      color="#52514e", x=0.006, ha="left", linespacing=1.5)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    print(f"wrote {out_png}")


def write_compare_csv(records: list[dict], out_csv: Path) -> None:
    """Per (codec, rank): affine fits for the cost metrics plus the mean ratio."""
    records = _with_ratio(records)
    rows = []
    for codec in CODECS:
        sub = [r for r in records if r["codec"] == codec]
        if not sub:
            continue
        for key, _title, unit, scale, do_fit in COMPARE_PANELS:
            if not do_fit:
                continue
            for ndim, f in fit_affine(sub, key).items():
                same = [r for r in sub if r["ndim"] == ndim]
                rows.append({
                    "codec": codec, "metric": key, "ndim": ndim,
                    "intercept": f["intercept"] / scale, "unit": unit,
                    "slope_per_MiB": f["slope"] / scale, "r2": f["r2"],
                    "mean_ratio": float(np.mean([r["ratio"] for r in same])),
                    "mean_bits_per_point": float(np.mean(
                        [8 * r["stream_bytes"] / r["points"] for r in same])),
                    "n_runs": f["n"],
                })
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")
    for r in rows:
        print(f"  {r['codec']:<14} {r['metric']:<9} {r['ndim']}D: "
              f"{r['intercept']:8.2f} {r['unit']} + {r['slope_per_MiB']:.4f}/MiB  "
              f"R2={r['r2']:.3f}  ratio={r['mean_ratio']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shape", type=int, nargs="+", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--codec", choices=CODECS, default="gnn",
                    help="gnn = the GNN codec; interp-* = the same pipeline with "
                         "SZ-style interpolation instead of the model; sz3/sperr = "
                         "external baselines through imagecodecs; tthresh = the "
                         "Tucker/HOSVD compressor through its CLI")
    ap.add_argument("--eb", type=float, default=1e-2,
                    help="error bound relative to the field's (max - min)")
    ap.add_argument("--tthresh-bin", default=TTHRESH_BIN,
                    help="path to the tthresh executable (default: $LATEC_TTHRESH_BIN)")
    ap.add_argument("--tthresh-eps", type=float, default=0.0,
                    help="tthresh -e target (relative L2); 0 = start from --eb. "
                         "Calibrate it per rank: tthresh has no point-wise bound")
    ap.add_argument("--tthresh-probes", type=int, default=_TT_PROBES,
                    help="round trips the -e search may spend per run")
    ap.add_argument("--tthresh-threads", type=int, default=0,
                    help="OMP_NUM_THREADS for tthresh; 0 = leave it to OpenMP")
    ap.add_argument("--tthresh-dir", default=None,
                    help="directory for tthresh's scratch files (default: $TMPDIR)")
    ap.add_argument("--baseline-python", default=os.environ.get("LATEC_BASELINE_PYTHON"),
                    help="interpreter with SZ3/SPERR-enabled imagecodecs "
                         "(default: $LATEC_BASELINE_PYTHON, else this one)")
    ap.add_argument("--levels", type=int, default=0,
                    help="schedule depth; 0 = rank-fixed (see fixed_levels)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    ap.add_argument("--mib", type=float, nargs="+", default=list(TARGET_MIB),
                    help="target input sizes in MiB of float32")
    ap.add_argument("--reps", type=int, default=1, help="repeats per point")
    ap.add_argument("--rep-offset", type=int, default=0,
                    help="index of the first repeat (distinct seeds); lets one "
                         "rank's reps be split over several GPUs and merged")
    ap.add_argument("--chunk-edge", type=int, default=0,
                    help="force this chunk edge on every axis (snapped down to a "
                         "multiple of the anchor stride); 0 = the codec's auto edges")
    ap.add_argument("--gpu", default="auto", help="GPU index, 'auto', or 'none'")
    ap.add_argument("--out", default="outputs/scaling", help="output stem (.json/.png)")
    ap.add_argument("--note", default=None, help="caption appended under the figure")
    ap.add_argument("--from-json", default=None, help="re-plot an existing json")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="json datasets (one per codec) to overlay in one figure")
    args = ap.parse_args()

    if args.worker:
        run_worker(args)
        return
    if args.gpu == "none":
        args.gpu = ""

    if args.compare:
        records: list[dict] = []
        for path in args.compare:
            data = json.loads(Path(path).read_text())
            for rec in data["records"]:
                rec.setdefault("codec", data.get("codec", "gnn"))
            records += data["records"]
        stem = Path(args.out)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps({"records": records}, indent=1))
        write_csv(records, stem.with_suffix(".csv"))
        write_compare_csv(records, stem.with_name(stem.name + "_fits.csv"))
        plot_compare(records, stem.with_suffix(".png"), args.note)
        return
    if args.from_json:
        records = json.loads(Path(args.from_json).read_text())["records"]
        data = json.loads(Path(args.from_json).read_text())
        write_csv(records, Path(args.from_json).with_suffix(".csv"))
        write_fits(records, Path(str(Path(args.from_json).with_suffix("")) + "_fits.csv"))
        plot(records, Path(args.from_json).with_suffix(".png"), data.get("note"))
        return

    records = measure(args)
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "reps": args.reps,
                    "codec": args.codec, "eb": args.eb, "records": records}, indent=1)
    )
    print(f"wrote {stem.with_suffix('.json')}")
    if records:
        write_csv(records, stem.with_suffix(".csv"))
        if len({r["in_bytes"] for r in records}) > 1:
            write_fits(records, stem.with_name(stem.name + "_fits.csv"))
            plot(records, stem.with_suffix(".png"))
        else:
            print("single size: no scaling fit or plot")


if __name__ == "__main__":
    main()
