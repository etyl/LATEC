"""Benchmark and profile the chunked GNN codec roundtrip.

Examples:
    python scripts/profile_gnn_inference.py --checkpoint data/gnn_predictor.pt
    python scripts/profile_gnn_inference.py --checkpoint model.pt --input field.npy --eb 1e-3
    python scripts/profile_gnn_inference.py --checkpoint model.pt --profile \
        --trace /tmp/codec_trace.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from latec.gnn_codec import GNNCompressorCodec


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True, help="versioned GNN checkpoint")
    ap.add_argument("--input", type=Path, help="image or .npy input")
    ap.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=(128, 128),
        help="synthetic spatial shape when --input is omitted",
    )
    ap.add_argument(
        "--eb",
        type=float,
        default=1e-2,
        help="error bound, relative to the input's (max - min)",
    )
    ap.add_argument("--levels", default="auto", help="schedule depth, or 'auto'")
    ap.add_argument(
        "--chunk-size", type=int, default=None, help="chunk edge (default: auto)"
    )
    ap.add_argument("--no-fp16", action="store_true", help="run the message pass fp32")
    ap.add_argument(
        "--device", help="cpu, cuda, or cuda:N (default: CUDA if available)"
    )
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, help="PyTorch CPU thread count")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--profile", action="store_true", help="print a torch.profiler operator table"
    )
    ap.add_argument(
        "--trace", type=Path, help="write a Chrome profiler trace (implies --profile)"
    )
    ap.add_argument("--profile-rows", type=int, default=30)
    return ap.parse_args(argv)


def load_input(args) -> np.ndarray:
    if args.input is None:
        rng = np.random.default_rng(args.seed)
        return rng.random(tuple(args.shape), dtype=np.float32)
    if args.input.suffix.lower() == ".npy":
        arr = np.load(args.input)
    else:
        from PIL import Image

        image = Image.open(args.input)
        if image.mode not in ("L", "RGB"):
            image = image.convert("RGB")
        arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("input cannot be empty")
    return arr


def synchronize(device: str) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def summarize(name: str, samples: list[float]) -> None:
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    print(
        f"{name:<18} mean {mean:9.2f} ms | p50 {median:9.2f} ms | "
        f"std {std:8.2f} ms | min {min(samples):9.2f} ms"
    )


def make_roundtrip(args, arr):
    codec = GNNCompressorCodec(args.checkpoint, args.device)
    levels = args.levels if args.levels == "auto" else int(args.levels)

    def roundtrip():
        synchronize(args.device)
        t0 = time.perf_counter()
        stream = codec.compress(
            arr,
            error_bound=args.eb,
            levels=levels,
            chunk_size=args.chunk_size,
            fp16=not args.no_fp16,
        )
        synchronize(args.device)
        t1 = time.perf_counter()
        codec.uncompress(stream)
        synchronize(args.device)
        return (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3, len(stream)

    return roundtrip


def profile_once(args, roundtrip):
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(args.device).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        roundtrip()

    sort_by = (
        "self_cuda_time_total"
        if torch.device(args.device).type == "cuda"
        else "self_cpu_time_total"
    )
    print("\nPyTorch operator profile")
    print(prof.key_averages().table(sort_by=sort_by, row_limit=args.profile_rows))
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))
        print(f"Chrome trace: {args.trace.resolve()}")


def main(argv=None):
    args = parse_args(argv)
    if args.eb <= 0 or args.warmup < 0 or args.repeats < 1:
        raise SystemExit(
            "--eb must be positive, --warmup non-negative, and --repeats >= 1"
        )

    import torch

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    arr = load_input(args)
    print(
        f"input={arr.shape} {arr.dtype} | device={args.device} | eb={args.eb} | "
        f"warmup={args.warmup} repeats={args.repeats}"
    )

    roundtrip = make_roundtrip(args, arr)
    for _ in range(args.warmup):
        roundtrip()
    enc, dec, sizes = [], [], []
    for _ in range(args.repeats):
        te, td, n = roundtrip()
        enc.append(te)
        dec.append(td)
        sizes.append(n)

    print("\nChunked codec benchmark")
    summarize("compress", enc)
    summarize("decompress", dec)
    print(f"stream {sizes[0]} bytes | ratio {arr.nbytes / sizes[0]:.2f}")
    if args.profile or args.trace:
        print("\nProfiling one additional roundtrip...")
        profile_once(args, roundtrip)


if __name__ == "__main__":
    main()
