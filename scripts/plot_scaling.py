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


# Edge snapping granularity per rank: coarse in 3-D, fine in 5-D, so the
# requested point counts stay distinct after rounding.
EDGE_MULTIPLE = {3: 4, 4: 2, 5: 1}


def edge_for(ndim: int, mib: float) -> int:
    """Cubic edge length giving ~mib MiB of float32, snapped to a multiple."""
    multiple = EDGE_MULTIPLE.get(ndim, 4)
    e = (mib * 2**20 / 4.0) ** (1.0 / ndim)
    e = int(round(e / multiple)) * multiple
    return max(e, 4 * multiple)


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
    agg_level = gp._load_inference_model(codec.checkpoint_path, torch, codec.device)[3]
    levels = args.levels if args.levels > 0 else fixed_levels(shape, agg_level)
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
    stream = codec.compress(x, levels=levels, chunk_size=chunk_edges)
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
    rec = {
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
    print("@@JSON@@" + json.dumps(rec))


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
    gpu = args.gpu if args.gpu != "auto" else pick_gpu()
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"using CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    records: list[dict] = []
    for ndim in args.ranks:
        seen: set[int] = set()
        for mib in args.mib:
            edge = edge_for(ndim, mib)
            if edge in seen:  # rounding collapsed two targets onto one shape
                continue
            seen.add(edge)
            shape = [edge] * ndim
            for rep in range(args.reps):
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
                       "--chunk-edge", str(args.chunk_edge),
                       "--levels", str(max(args.levels, 0)),
                       "--shape", *map(str, shape),
                       "--seed", str(1000 * ndim + 10 * rep + 1),
                       "--checkpoint", str(args.checkpoint)]
                t0 = time.time()
                proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
                tag = f"{ndim}D {'x'.join(map(str, shape))} rep{rep}"
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

CSV_COLUMNS = [
    "ndim", "edge", "shape", "points", "levels", "chunk_edge", "chunk_points",
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


def plot(records: list[dict], out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("gpu_peak", "Peak GPU memory", "MiB", 2**20),
        ("rss_peak", "Peak host memory (RSS)", "MiB", 2**20),
        ("encode_s", "Compress time", "s", 1.0),
        ("decode_s", "Decompress time", "s", 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9.0), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0, 0.05, 1, 0.90))
    fig.patch.set_facecolor("#fcfcfb")

    for ax, (key, title, unit, scale) in zip(axes.ravel(), panels):
        ax.set_facecolor("#fcfcfb")
        for ndim, (xs, mean, lo, hi, pts) in aggregate(records, key).items():
            c = SERIES_LIGHT.get(ndim, "#52514e")
            ax.plot(xs, mean / scale, color=c, lw=2, marker="o", ms=4.5,
                    label=f"{ndim}D", zorder=3)
            # small fixed vertical stagger so line-end labels never overlap
            dy = {3: -7.0, 4: 7.0}.get(ndim, 0.0)
            ax.annotate(f"{ndim}D", (xs[-1], mean[-1] / scale), color=c,
                        fontsize=9, fontweight="bold",
                        xytext=(5, dy), textcoords="offset points", va="center")
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

    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor="#52514e", title="rank",
                      title_fontsize=9)
    dev = records[0].get("device", "?") if records else "?"
    lv_by_rank = {r["ndim"]: max(x["levels"] for x in records if x["ndim"] == r["ndim"])
                  for r in records}
    fig.suptitle(
        f"GNN codec scaling on random fields ({dev}, chunked path, "
        f"fixed levels per rank, codec defaults otherwise)",
        fontsize=12.5, color="#0b0b0b", x=0.01, y=0.985, ha="left",
    )
    fig.text(
        0.01, 0.012,
        "One run per point (the codec is deterministic at a fixed shape). "
        "Linear axes, y starting at 0.\nChunked path at every size, schedule depth "
        "fixed per rank "
        "(" + ", ".join(f"{k}D L{v}" for k, v in sorted(lv_by_rank.items())) + "); every "
        "size here is many chunks, so no point falls in the sub-chunk regime.",
        fontsize=8.5, color="#52514e", ha="left", va="bottom", linespacing=1.5,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor=fig.get_facecolor())
    print(f"wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shape", type=int, nargs="+", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levels", type=int, default=0,
                    help="schedule depth; 0 = rank-fixed (see fixed_levels)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--ranks", type=int, nargs="+", default=list(RANKS))
    ap.add_argument("--mib", type=float, nargs="+", default=list(TARGET_MIB),
                    help="target input sizes in MiB of float32")
    ap.add_argument("--reps", type=int, default=1, help="repeats per point")
    ap.add_argument("--chunk-edge", type=int, default=0,
                    help="force this chunk edge on every axis (snapped down to a "
                         "multiple of the anchor stride); 0 = the codec's auto edges")
    ap.add_argument("--gpu", default="auto", help="GPU index, 'auto', or 'none'")
    ap.add_argument("--out", default="outputs/scaling", help="output stem (.json/.png)")
    ap.add_argument("--from-json", default=None, help="re-plot an existing json")
    args = ap.parse_args()

    if args.worker:
        run_worker(args)
        return
    if args.gpu == "none":
        args.gpu = ""

    if args.from_json:
        records = json.loads(Path(args.from_json).read_text())["records"]
        write_csv(records, Path(args.from_json).with_suffix(".csv"))
        plot(records, Path(args.from_json).with_suffix(".png"))
        return

    records = measure(args)
    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(
        json.dumps({"checkpoint": args.checkpoint, "reps": args.reps,
                    "records": records}, indent=1)
    )
    print(f"wrote {stem.with_suffix('.json')}")
    if records:
        write_csv(records, stem.with_suffix(".csv"))
        plot(records, stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
