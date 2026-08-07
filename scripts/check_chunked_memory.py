"""Verify the chunked GNN codec's memory bound on tensors far beyond what the
whole-tensor path can handle.

    /usr/bin/time -v python scripts/check_chunked_memory.py --shape 256 256 256

Compresses + decompresses a synthetic tensor with a random v6 checkpoint (the
error-bound guarantee and the memory profile are independent of the weights),
checks |x - recon| <= eb, and prints the process peak RSS. On hosts without
``constriction`` a bit-exact raw stand-in is patched in for the rANS backend —
payload sizes are then meaningless, but memory and correctness are not.
"""

from __future__ import annotations

import argparse
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ensure_rans_backend():
    try:
        import constriction  # noqa: F401

        return "constriction"
    except ImportError:
        import deepsz.bitstream as bitstream
        import deepsz.rans as rans

        def fake_encode(codes, levels64, tables):
            return np.asarray(codes, np.uint32).ravel().astype("<u4").tobytes()

        def fake_decode(blob, levels64, tables):
            out = np.frombuffer(blob, dtype="<u4").astype(np.uint32)
            if len(out) != len(np.asarray(levels64).ravel()):
                raise ValueError("fake rANS length mismatch")
            return out

        rans.rans_encode = bitstream.rans_encode = fake_encode
        rans.rans_decode = bitstream.rans_decode = fake_decode
        return "raw stand-in (no constriction; sizes not meaningful)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", type=int, nargs="+", default=(256, 256, 256))
    ap.add_argument(
        "--chunk-size", type=int, default=None, help="chunk edge (default: codec auto)"
    )
    ap.add_argument("--eb", type=float, default=1e-2)
    ap.add_argument("--d", type=int, default=32, help="model width")
    ap.add_argument(
        "--agg-level",
        type=int,
        default=2,
        help="aggregation level baked into the random checkpoint",
    )
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument(
        "--checkpoint", default=None, help="real checkpoint (default: random weights)"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    backend = _ensure_rans_backend()
    import torch

    from deepsz import GNNCompressorCodec
    from deepsz.gnn_predictor import CKPT_VERSION, build_model

    ckpt = args.checkpoint
    if ckpt is None:
        torch.manual_seed(args.seed)
        model = build_model(d=args.d, agg_level=args.agg_level).eval()
        ckpt = str(Path(tempfile.mkdtemp()) / "gnn_random.pt")
        torch.save(
            {
                "d": args.d,
                "agg_level": args.agg_level,
                "state_dict": model.state_dict(),
                "version": CKPT_VERSION,
            },
            ckpt,
        )

    shape = tuple(args.shape)
    n = int(np.prod(shape))
    rng = np.random.RandomState(args.seed)
    # smooth-ish field: sum of low-frequency separable cosines + noise
    x = rng.rand(*shape).astype(np.float32) * 0.05
    for k, s in enumerate(shape):
        wave = np.cos(np.linspace(0, 4 * np.pi, s, dtype=np.float32))
        x += wave.reshape([-1 if i == k else 1 for i in range(len(shape))])
    print(
        f"shape={shape} ({n / 1e6:.1f}M points, {x.nbytes / 2**20:.0f} MiB), "
        f"eb={args.eb}, d={args.d}, rans backend: {backend}"
    )

    # GNNCompressorCodec normalizes internally and treats error_bound as
    # relative to (max - min); convert args.eb (absolute, used in the
    # OK/VIOLATED check below) back to that.
    span = max(float(x.max()) - float(x.min()), 1.0)
    codec = GNNCompressorCodec(ckpt)

    t0 = time.time()
    stream = codec.compress(
        x, error_bound=args.eb / span, levels=args.levels, chunk_size=args.chunk_size
    )
    t1 = time.time()
    y = codec.uncompress(stream).numpy()
    t2 = time.time()

    err = float(np.abs(y - x).max())
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(
        f"compress {t1 - t0:.1f}s, decompress {t2 - t1:.1f}s, "
        f"stream {len(stream) / 2**20:.1f} MiB"
    )
    print(
        f"max |x - recon| = {err:.6g} (eb {args.eb}): "
        f"{'OK' if err <= args.eb else 'VIOLATED'}"
    )
    print(f"peak RSS: {peak_kb / 2**20:.2f} GiB")
    if err > args.eb:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
