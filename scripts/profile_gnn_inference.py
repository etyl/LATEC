"""Benchmark and profile GNN inference on the real closed-loop codec path.

Examples:
    python scripts/profile_gnn_inference.py --checkpoint data/gnn_predictor.pt
    python scripts/profile_gnn_inference.py --checkpoint model.pt --input image.png --eb 2
    python scripts/profile_gnn_inference.py --checkpoint model.pt --profile \
        --trace inference_trace.json
    python scripts/profile_gnn_inference.py --checkpoint model.pt --mode codec
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

from latec.codec import compress, decompress
from latec.gnn_predictor import GNNPredictor
from latec.levels import stage_ebs, stage_masks
from latec.quantizer import dequantize, quantize


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
        "--eb", type=float, default=0.01, help="absolute error bound in input units"
    )
    ap.add_argument(
        "--device", help="cpu, cuda, or cuda:N (default: CUDA if available)"
    )
    ap.add_argument(
        "--mode",
        choices=("predictor", "codec"),
        default="predictor",
        help="profile staged prediction or the complete image codec",
    )
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--anchor-stride", type=int, default=16)
    ap.add_argument("--anchor-block", type=int, default=1)
    ap.add_argument(
        "--max-radius", type=int, default=64, help="maximum GNN neighbour radius"
    )
    ap.add_argument("--radius", type=int, default=1 << 15, help="quantizer radius")
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


def predictor_values(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return channel-first values and the spatial shape."""
    if arr.ndim == 2:
        values = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3):
        values = np.moveaxis(arr, -1, 0)
    else:
        values = arr.reshape((1, *arr.shape))
    return values.astype(np.float32, copy=False), values.shape[1:]


def make_predictor(args, vmin: float, vmax: float) -> GNNPredictor:
    return GNNPredictor(
        args.checkpoint,
        vmin,
        vmax,
        max_radius=args.max_radius,
        device=args.device,
        levels=args.levels,
        anchor_stride=args.anchor_stride,
        anchor_block=args.anchor_block,
    )


def synchronize(device: str) -> None:
    import torch

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def predictor_pass(
    values, predictor, masks, ebs, args, round_output=False, collect_stages=True
):
    """Run the encoder's prediction/quantization feedback loop without coding."""
    import torch

    recon = np.zeros_like(values)
    known = np.zeros(values.shape[1:], dtype=bool)
    stage_ms = []
    t_total = time.perf_counter()
    for stage_idx, (pos, eb) in enumerate(zip(masks, ebs)):
        n = int(pos.sum())
        if not n:
            continue
        if stage_idx == 0:
            pred = np.zeros((values.shape[0], n), np.float32)
        else:
            synchronize(args.device)
            t0 = time.perf_counter()
            with torch.profiler.record_function(f"gnn_predict_stage_{stage_idx}"):
                pred, _scale = predictor.predict(recon, known, pos, eb=eb)
            synchronize(args.device)
            if collect_stages:
                stage_ms.append((stage_idx, n, (time.perf_counter() - t0) * 1e3))
        codes, outliers = quantize(
            values[:, pos], pred, eb, args.radius, round_output=round_output
        )
        recon[:, pos] = dequantize(pred, codes, outliers, eb, args.radius).reshape(
            values.shape[0], n
        )
        known |= pos
    synchronize(args.device)
    return (time.perf_counter() - t_total) * 1e3, stage_ms


def summarize(name: str, samples: list[float]) -> None:
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    print(
        f"{name:<18} mean {mean:9.2f} ms | p50 {median:9.2f} ms | "
        f"std {std:8.2f} ms | min {min(samples):9.2f} ms"
    )


def run_predictor(args, arr):
    values, shape = predictor_values(arr)
    vmin, vmax = float(values.min()), float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    predictor = make_predictor(args, vmin, vmax)
    masks = stage_masks(shape, args.levels, args.anchor_stride, args.anchor_block)
    ebs = stage_ebs(
        shape, args.levels, args.anchor_stride, args.anchor_block, args.eb, 1.0
    )
    round_output = np.issubdtype(arr.dtype, np.integer)

    for _ in range(args.warmup):
        predictor_pass(
            values, predictor, masks, ebs, args, round_output, collect_stages=False
        )

    totals, per_stage = [], {}
    for _ in range(args.repeats):
        total, stages = predictor_pass(
            values, predictor, masks, ebs, args, round_output
        )
        totals.append(total)
        for stage_idx, n, elapsed in stages:
            per_stage.setdefault((stage_idx, n), []).append(elapsed)

    print("\nClosed-loop predictor benchmark")
    summarize("total inference", totals)
    print("\nPer-stage prediction (mean across repeats)")
    print(f"{'stage':>7} {'values':>12} {'latency':>12}")
    for (stage_idx, n), samples in per_stage.items():
        print(
            f"{stage_idx:>7} {n * values.shape[0]:>12,} "
            f"{statistics.fmean(samples):>10.2f} ms"
        )

    if args.profile or args.trace:
        profile_predictor(args, values, predictor, masks, ebs, round_output)


def profile_predictor(args, values, predictor, masks, ebs, round_output):
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.device(args.device).type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True, with_stack=False
    ) as prof:
        predictor_pass(
            values, predictor, masks, ebs, args, round_output, collect_stages=False
        )

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


def run_codec(args, arr):
    if arr.ndim not in (2, 3) or (arr.ndim == 3 and arr.shape[-1] not in (1, 3)):
        raise ValueError("--mode codec requires an HxW or HxWx{1,3} input")
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    encoder = make_predictor(args, vmin, vmax)
    decoder = make_predictor(args, vmin, vmax)

    def roundtrip():
        synchronize(args.device)
        t0 = time.perf_counter()
        stream, stats = compress(
            arr,
            args.eb,
            encoder,
            levels=args.levels,
            anchor_stride=args.anchor_stride,
            anchor_block=args.anchor_block,
            radius=args.radius,
            eb_ratio=1.0,
            tune="fast",
        )
        synchronize(args.device)
        t1 = time.perf_counter()
        decompress(stream, lambda _header: decoder)
        synchronize(args.device)
        return (t1 - t0) * 1e3, (time.perf_counter() - t1) * 1e3, stats

    for _ in range(args.warmup):
        roundtrip()
    enc, dec, predict, quantize_ms, entropy = [], [], [], [], []
    for _ in range(args.repeats):
        te, td, stats = roundtrip()
        enc.append(te)
        dec.append(td)
        predict.append(stats["predict_s"] * 1e3)
        quantize_ms.append(stats["quantize_s"] * 1e3)
        entropy.append(stats["entropy_s"] * 1e3)

    print("\nFull codec benchmark")
    summarize("compress", enc)
    summarize("decompress", dec)
    summarize("encode prediction", predict)
    summarize("encode quantize", quantize_ms)
    summarize("encode entropy", entropy)
    if args.profile or args.trace:
        print("\nProfiling one additional codec roundtrip...")
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

    if args.mode == "predictor":
        run_predictor(args, arr)
    else:
        run_codec(args, arr)


if __name__ == "__main__":
    main()
