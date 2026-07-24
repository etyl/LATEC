"""Roundtrip-eval DeepSZ on an arbitrary tensor loaded from .npy or torch.

    python scripts/eval_tensor.py field.npy --eb 1e-3
    python scripts/eval_tensor.py weights.pt --eb 0.01 --rel

Same flags as `deepsz eval`; the only difference is the input is a raw tensor
(any dtype/shape the codec accepts, i.e. 2-D or 3-D with 1 or 3 channels)
instead of an image decoded by PIL.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

# Use this worktree's deepsz, not a stale pip-installed copy in site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepsz.cli import add_common, build_predictor, run_compress
from deepsz.codec import decompress


def _cap_crop(arr: np.ndarray, stride: int, cap: int) -> np.ndarray:
    """Centered crop with <= ``cap`` voxels, each edge a multiple of ``stride``
    so the interp level schedule stays valid. Returns arr unchanged if it fits."""
    if arr.size <= cap:
        return arr
    edge = max(int(cap ** (1.0 / arr.ndim)), stride)
    edge -= edge % stride or 0
    edge = max(edge, stride)
    sl = tuple(
        slice((d - min(d, edge)) // 2, (d - min(d, edge)) // 2 + min(d, edge))
        for d in arr.shape
    )
    return np.ascontiguousarray(arr[sl])


def report(label, arr, rec, nbytes, eb, t_comp=None, t_dec=None):
    """One comparable line per codec: same PSNR (value-range peak) and bits/value."""
    a = arr.astype(np.float64)
    r = rec.reshape(arr.shape).astype(np.float64)
    max_err = float(np.abs(a - r).max())
    mse = float(np.mean((a - r) ** 2))
    peak = (
        255.0
        if arr.dtype == np.uint8
        else max(float(arr.max()) - float(arr.min()), 1e-12)
    )
    psnr = 10 * np.log10(peak**2 / mse) if mse > 0 else float("inf")
    bpv = 8 * nbytes / arr.size
    ratio = arr.nbytes / nbytes
    t = f"  {t_comp:.1f}s/{t_dec:.1f}s" if t_comp is not None else ""
    print(
        f"  [{label:6s}] {nbytes:>10} B  ratio {ratio:7.2f}  {bpv:7.3f} bpv  "
        f"PSNR {psnr:6.2f} dB  maxerr {max_err:.3g} "
        f"{'PASS' if max_err <= eb else 'FAIL'}{t}"
    )
    return max_err <= eb


def load_tensor(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith((".pt", ".pth")):
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(obj, torch.Tensor):
            raise ValueError(
                f"{path} holds {type(obj).__name__}, expected a single tensor"
            )
        return obj.detach().cpu().numpy()
    raise ValueError(f"unsupported extension: {path} (use .npy/.pt/.pth)")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", help="tensor file (.npy or .pt/.pth)")
    add_common(ap)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="gnn only: force chunk edge (multiple of 2 ** levels); "
        "0 = whole-tensor, omit = maximize chunk size within "
        "the automatic point budget",
    )
    ap.add_argument(
        "--fp16",
        action="store_true",
        help="gnn only: fp16 autocast on the GNN message pass (cuda; "
        "~2x forward, readout stays fp32). May cost a little ratio "
        "at small eb -- compare bits/value with and without",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="min-max scale the tensor to [0,1] before compressing, so "
        "eb is comparable across tensors of different raw scale",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="gnn only: torch.compile the message-pass embed (fuses "
        "the elementwise ~40%%; one-off compile cost on first "
        "chunk). Same float path replayed at decode",
    )
    args = ap.parse_args(argv)
    # add_common forces --levels to a fixed int default (for the interp path);
    # let the GNN codec pick levels from the tensor shape unless the user asked
    # for a specific depth.
    _argv = sys.argv[1:] if argv is None else list(argv)
    levels_explicit = any(a == "--levels" or a.startswith("--levels=") for a in _argv)

    arr = load_tensor(args.input)
    orig_bytes = arr.nbytes
    if arr.dtype == np.float64:  # codec pipeline is float32; store as such
        print("note: float64 input cast to float32 (float32-precision reconstruction)")
        arr = arr.astype(np.float32)
    if args.normalize:
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr.astype(np.float32) - lo) / max(hi - lo, 1e-12)
        print(f"normalized [{lo:.4g},{hi:.4g}] -> [0,1]")
    eb = args.eb
    if args.rel:
        eb = args.eb * max(float(arr.max()) - float(arr.min()), 1.0)

    if args.predictor == "gnn":
        # GNNCompressorCodec auto-chunks large tensors; the codec.compress path
        # allocates a dense embedding field over the whole tensor and OOMs.
        os.environ.setdefault("DEEPSZ_PROGRESS", "1")  # per-chunk progress to stderr
        from deepsz.gnn_codec import GNNCompressorCodec

        # GNNCompressorCodec normalizes the tensor internally and treats
        # error_bound as relative to (max - min); convert this script's
        # absolute eb (used below for report()/baselines) back to that.
        gnn_span = max(float(arr.max()) - float(arr.min()), 1.0)
        codec = GNNCompressorCodec(
            args.gnn_checkpoint,
            error_bound=eb / gnn_span,
            levels=args.levels if levels_explicit else "auto",
            radius=args.radius,
            zstd_level=args.zstd_level,
            eb_ratio=args.eb_ratio,
            tune=args.tune if args.tune in ("fast", "size") else "fast",
            chunk_size=args.chunk_size,
            fp16=args.fp16,
            compile=args.compile,
        )
        t0 = time.time()
        stream = codec.compress(arr)
        t_comp = time.time() - t0
        stats = {"outliers": 0}  # ponytail: codec doesn't surface outlier count
        t0 = time.time()
        rec = codec.uncompress(stream).numpy()
        t_dec = time.time() - t0
    else:
        t0 = time.time()
        stream, stats = run_compress(arr, args)
        t_comp = time.time() - t0

        t0 = time.time()
        rec = decompress(stream, lambda hdr: build_predictor(args, hdr))
        t_dec = time.time() - t0

    print(f"tensor {args.input} {arr.shape} {arr.dtype}, eb={eb} (orig {orig_bytes} B)")
    main_label = args.predictor
    bound_ok = report(main_label, arr, rec, len(stream), eb, t_comp, t_dec)
    print(
        f"  ({main_label}: outliers {stats['outliers']} "
        f"= {100 * stats['outliers'] / arr.size:.3f}%)"
    )

    # Baselines at the identical eb. Two gotchas this handles:
    #  * interp compresses the whole field un-chunked -> OOMs on a big field
    #    (the GNN run chunks; interp doesn't), so it runs on a memory-capped crop.
    #  * bits/value is NOT scale-invariant: a small crop costs more bpv than the
    #    whole field for ANY codec. So sz3 runs on the SAME crop (fair vs interp)
    #    AND on the full field (the headline vs the GNN).
    # DEEPSZ_BASELINE_MAXVOX=0 disables the cap (interp whole-field; may OOM).
    from deepsz.baselines import _sz3_pysz

    def sz3(field):
        tag = "sz3" if field.shape == arr.shape else f"sz3{list(field.shape)}"
        try:
            r = _sz3_pysz(np.ascontiguousarray(field, np.float32), eb)
        except Exception as exc:  # pysz can reject high-rank shapes; keep going
            print(f"  [{tag:6s}] failed: {exc}")
            return
        if r is not None:
            report(tag, field, r[1], r[0], eb)

    if main_label not in ("interp", "interp-linear"):
        cap = int(os.environ.get("DEEPSZ_BASELINE_MAXVOX", 1 << 24))
        sub = _cap_crop(arr, args.anchor_stride, cap) if cap else arr
        cropped = sub.shape != arr.shape
        bargs = copy.copy(args)
        bargs.predictor = "interp"
        t0 = time.time()
        b_stream, _ = run_compress(sub, bargs)
        b_tc = time.time() - t0
        t0 = time.time()
        b_rec = decompress(b_stream, lambda hdr: build_predictor(bargs, hdr))
        report(
            f"interp{list(sub.shape) if cropped else ''}",
            sub,
            b_rec,
            len(b_stream),
            eb,
            b_tc,
            time.time() - t0,
        )
        sz3(sub)  # same crop -> fair interp-vs-sz3
        if cropped:
            sz3(arr)  # full field -> headline vs the GNN
    else:
        sz3(arr)

    if not bound_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
