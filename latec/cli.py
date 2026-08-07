"""LATEC command line: compress / decompress / eval.

Examples:
    latec compress photo.png photo.msz --eb 2
    latec decompress photo.msz rec.png
    latec eval photo.png --eb 2 --levels 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from .bitstream import (
    FLAG_COMPILED,
    FLAG_CUBIC,
    FLAG_FP16,
    FLAG_GNN,
    FLAG_INTERP,
    Header,
)
from .codec import compress, decompress
from .predictor import InterpPredictor

DEFAULT_GNN = Path(__file__).resolve().parent.parent / "data" / "gnn_predictor.pt"


def load_image(path: str) -> np.ndarray:
    from PIL import Image

    im = Image.open(path)
    if im.mode in ("L", "I;16"):
        return np.asarray(im.convert("L"))
    if im.mode != "RGB":
        if "A" in im.mode:
            print(
                f"warning: dropping alpha channel of {im.mode} image", file=sys.stderr
            )
        im = im.convert("RGB")
    return np.asarray(im)


def save_image(path: str, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr).save(path)


def build_predictor(args, header: Header):
    """Decompress-side predictor; all parameters come from the stream header."""
    if header.flags & FLAG_INTERP:
        return InterpPredictor(
            "cubic" if header.flags & FLAG_CUBIC else "linear",
            header.levels,
            header.anchor_stride,
            header.anchor_block,
        )
    if not header.flags & FLAG_GNN:
        raise ValueError("stream needs a predictor factory the CLI can't build")
    from .gnn_predictor import GNNPredictor

    pred = GNNPredictor(
        args.gnn_checkpoint,
        header.vmin,
        header.vmax,
        max_radius=header.max_radius,
        levels=header.levels,
        anchor_stride=header.anchor_stride,
        anchor_block=header.anchor_block,
    )
    pred.fp16 = bool(header.flags & FLAG_FP16)
    pred.compile = bool(header.flags & FLAG_COMPILED)
    if pred.checkpoint_hash != header.ckpt_hash:
        print(
            "warning: checkpoint hash differs from the one used to compress; "
            "decoded output may violate the error bound",
            file=sys.stderr,
        )
    return pred


def add_common(ap):
    ap.add_argument(
        "--eb",
        type=float,
        required=True,
        help="error bound in original data units (e.g. 0-255 for uint8)",
    )
    ap.add_argument(
        "--rel",
        action="store_true",
        help="interpret --eb as relative to the data range",
    )
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--anchor-stride", type=int, default=16)
    ap.add_argument("--anchor-block", type=int, default=1)
    ap.add_argument("--radius", type=int, default=1 << 15)
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument(
        "--eb-ratio",
        type=float,
        default=None,
        help="per-level eb decay; omit to use --tune",
    )
    ap.add_argument(
        "--tune",
        choices=("fast", "size", "rd"),
        default="fast",
        help="auto search when --eb-ratio is omitted",
    )
    ap.add_argument(
        "--tune-size-slack",
        type=float,
        default=1.05,
        help="for --tune rd, may spend up to this size factor for "
        "lower reconstruction SSE",
    )
    ap.add_argument(
        "--predictor",
        choices=("gnn", "interp", "interp-linear"),
        default="interp",
        help="interp = SZ-style cubic interpolation (default); "
        "interp-linear = its linear variant; gnn = trained GNN",
    )
    ap.add_argument("--gnn-checkpoint", default=str(DEFAULT_GNN))
    ap.add_argument("-v", "--verbose", action="store_true")


def run_compress(img: np.ndarray, args) -> tuple[bytes, dict]:
    eb = args.eb
    if args.rel:
        span = float(img.max()) - float(img.min())
        eb = args.eb * (span if span > 0 else 1.0)
    kind = args.predictor
    if kind in ("interp", "interp-linear"):
        order = "linear" if kind == "interp-linear" else "cubic"
        predictor = InterpPredictor(
            order, args.levels, args.anchor_stride, args.anchor_block
        )
    else:  # gnn
        from .gnn_predictor import GNNPredictor

        predictor = GNNPredictor(
            args.gnn_checkpoint,
            float(img.min()),
            float(img.max()),
            levels=args.levels,
            anchor_stride=args.anchor_stride,
            anchor_block=args.anchor_block,
        )
    return compress(
        img,
        eb,
        predictor,
        levels=args.levels,
        anchor_stride=args.anchor_stride,
        anchor_block=args.anchor_block,
        radius=args.radius,
        zstd_level=args.zstd_level,
        eb_ratio=args.eb_ratio,
        tune=args.tune,
        tune_size_slack=args.tune_size_slack,
        verbose=args.verbose,
    )


def cmd_compress(args):
    img = load_image(args.input)
    t0 = time.time()
    stream, stats = run_compress(img, args)
    Path(args.output).write_bytes(stream)
    bpp = 8 * len(stream) / (img.shape[0] * img.shape[1])
    print(
        f"{args.input}: {stats['original_bytes']} -> {len(stream)} bytes "
        f"(ratio {stats['ratio']:.2f}, {bpp:.3f} bpp) in {time.time() - t0:.1f}s"
    )
    print(
        f"  predict {stats['predict_s']:.1f}s | quantize {stats['quantize_s']:.1f}s "
        f"| entropy {stats['entropy_s']:.1f}s | outliers {stats['outliers']}"
    )


def cmd_decompress(args):
    stream = Path(args.input).read_bytes()
    t0 = time.time()
    out = decompress(stream, lambda hdr: build_predictor(args, hdr))
    save_image(args.output, out)
    print(f"{args.input} -> {args.output} {out.shape} in {time.time() - t0:.1f}s")


def cmd_eval(args):
    img = load_image(args.input)
    eb = (
        args.eb
        if not args.rel
        else args.eb * max(float(img.max()) - float(img.min()), 1.0)
    )

    t0 = time.time()
    stream, stats = run_compress(img, args)
    t_comp = time.time() - t0

    t0 = time.time()
    rec = decompress(stream, lambda hdr: build_predictor(args, hdr))
    t_dec = time.time() - t0

    rec2d = rec if rec.ndim == img.ndim else rec[..., None]
    max_err = float(np.abs(img.astype(np.float64) - rec2d.astype(np.float64)).max())
    mse = float(np.mean((img.astype(np.float64) - rec2d.astype(np.float64)) ** 2))
    peak = 255.0 if img.dtype == np.uint8 else float(img.max()) - float(img.min())
    psnr = 10 * np.log10(peak**2 / mse) if mse > 0 else float("inf")
    bpp = 8 * len(stream) / (img.shape[0] * img.shape[1])

    import zstandard

    zstd_raw = len(
        zstandard.ZstdCompressor(level=args.zstd_level).compress(
            np.ascontiguousarray(img).tobytes()
        )
    )

    bound_ok = max_err <= eb
    print(f"image {args.input} {img.shape} {img.dtype}, eb={eb}")
    print(
        f"  compressed:  {len(stream)} bytes, ratio {stats['ratio']:.2f}, {bpp:.3f} bpp"
    )
    print(f"  zstd-raw:    {zstd_raw} bytes, ratio {img.nbytes / zstd_raw:.2f}")

    from .baselines import sz3_roundtrip

    sz3 = sz3_roundtrip(img, eb)
    if sz3 is not None:
        sz3_bytes, sz3_rec = sz3
        sz3_mse = float(
            np.mean((img.astype(np.float64) - sz3_rec.astype(np.float64)) ** 2)
        )
        sz3_psnr = 10 * np.log10(peak**2 / sz3_mse) if sz3_mse > 0 else float("inf")
        sz3_err = float(
            np.abs(img.astype(np.float64) - sz3_rec.astype(np.float64)).max()
        )
        print(
            f"  sz3:         {sz3_bytes} bytes, ratio {img.nbytes / sz3_bytes:.2f}, "
            f"{8 * sz3_bytes / (img.shape[0] * img.shape[1]):.3f} bpp, "
            f"PSNR {sz3_psnr:.2f} dB, max err {sz3_err}"
        )
    else:
        print("  sz3:         (sz3 binary not found, baseline skipped)")
    print(f"  PSNR:        {psnr:.2f} dB")
    print(f"  max abs err: {max_err} <= eb: {'PASS' if bound_ok else 'FAIL'}")
    print(
        f"  outliers:    {stats['outliers']} "
        f"({100 * stats['outliers'] / img.size:.3f}% of values)"
    )
    print(
        f"  time:        compress {t_comp:.1f}s, decompress {t_dec:.1f}s "
        f"(predict {stats['predict_s']:.1f}s)"
    )
    if not bound_ok:
        raise SystemExit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="latec", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("compress", help="compress an image to a .msz stream")
    p.add_argument("input")
    p.add_argument("output")
    add_common(p)
    p.set_defaults(fn=cmd_compress)

    p = sub.add_parser("decompress", help="decompress a .msz stream to an image")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--gnn-checkpoint", default=str(DEFAULT_GNN))
    p.set_defaults(fn=cmd_decompress)

    p = sub.add_parser("eval", help="in-memory roundtrip with metrics")
    p.add_argument("input")
    add_common(p)
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
