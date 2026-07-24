"""Benchmark the GNN codec on a fixed-size n-D subset of a large tensor.

Crops a centred ``--subset-edge`` (default 64) hypercube out of a large tensor
and runs one closed-loop GNN compress + decompress on it, reporting the metrics
that matter for judging an optimisation: reconstruction quality (PSNR, max
error), size (bits/value, ratio), speed (compress/decompress wall + voxel
throughput), and resource use (peak host RAM, peak GPU memory, mean/peak GPU
SM utilisation). The subset edge is capped per axis to the tensor's real extent
and floored to a multiple of ``2 ** levels``, so the *same* script runs on
the 32^4 local ``rti_normal.npy`` and on the full-size Jean Zay tensor.

    python scripts/bench_gnn_subset.py data/rti_normal.npy \
        --gnn-checkpoint checkpoints/d64.pt --eb 1e-4 --normalize \
        --levels 5 --chunk-size 32

Each run prints one report (config + git commit + every metric) to stdout;
tag it with ``--label`` and re-run before/after an optimisation to compare.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Use this worktree's deepsz, not a stale pip-installed copy in site-packages.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_tensor(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    if path.endswith((".pt", ".pth")):
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(obj, torch.Tensor):
            raise ValueError(f"{path} holds {type(obj).__name__}, expected a tensor")
        return obj.detach().cpu().numpy()
    raise ValueError(f"unsupported extension: {path} (use .npy/.pt/.pth)")


def centred_subset(arr: np.ndarray, edge: int, stride: int) -> np.ndarray:
    """Centred crop with per-axis length ``min(edge_stride_floored, dim)``.

    Each edge is floored to a multiple of ``stride`` (the interp/GNN level
    schedule needs anchor-aligned extents) but never below ``stride`` and never
    above the axis. Materialises the crop into a contiguous in-RAM array (the
    source may be an mmap)."""
    lens = []
    for d in arr.shape:
        e = min(edge, d)
        e -= e % stride
        e = max(e, min(stride, d))
        lens.append(e)
    sl = tuple(slice((d - e) // 2, (d - e) // 2 + e) for d, e in zip(arr.shape, lens))
    return np.ascontiguousarray(arr[sl])


class GpuSampler:
    """Poll GPU SM utilisation and used memory on a background thread.

    Prefers NVML (``pynvml``); falls back to parsing ``nvidia-smi``; degrades to
    a no-op if neither is available (e.g. CPU-only box) so the benchmark still
    runs. ``.summary()`` returns mean/peak utilisation (%) and peak used MiB, or
    ``None`` fields when no samples were taken."""

    def __init__(self, device_index: int = 0, interval: float = 0.1):
        self.index = device_index
        self.interval = interval
        self.util: list[float] = []
        self.mem: list[float] = []
        self.baseline_mem_mib: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sample = self._pick_backend()

    def _pick_backend(self):
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(self.index)

            def sample():
                r = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                return float(r.gpu), m.used / (1024 * 1024)

            sample()  # probe once; raises if the query is broken
            return sample
        except Exception:
            pass
        # nvidia-smi fallback
        try:
            q = (
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(self.index),
            )

            def sample():
                out = subprocess.check_output(q, text=True, timeout=5).strip()
                u, m = out.splitlines()[0].split(",")
                return float(u), float(m)

            sample()
            return sample
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self._sample is not None

    def _loop(self):
        while not self._stop.is_set():
            try:
                u, m = self._sample()
                self.util.append(u)
                self.mem.append(m)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        if self._sample is not None:
            try:
                u, m = self._sample()
                self.util.append(u)
                self.mem.append(m)
                self.baseline_mem_mib = m
            except Exception:
                pass
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        if not self.util:
            return {
                "gpu_util_mean_pct": None,
                "gpu_util_peak_pct": None,
                "gpu_mem_peak_mib": None,
                "gpu_mem_increase_mib": None,
                "gpu_samples": 0,
            }
        peak_mem = max(self.mem)
        increase = (
            max(0.0, peak_mem - self.baseline_mem_mib)
            if self.baseline_mem_mib is not None
            else None
        )
        return {
            "gpu_util_mean_pct": round(sum(self.util) / len(self.util), 1),
            "gpu_util_peak_pct": round(max(self.util), 1),
            "gpu_mem_peak_mib": round(peak_mem, 1),
            "gpu_mem_increase_mib": (
                round(increase, 1) if increase is not None else None
            ),
            "gpu_samples": len(self.util),
        }


class HostMemorySampler:
    """Measure this process's RSS only while the benchmark is running.

    ``resource.ru_maxrss`` is a high-water mark for the process's entire
    lifetime and cannot be reset before a benchmark.  Sampling the current RSS
    gives both the peak during the measured region and the increase over the
    region's initial footprint.
    """

    _PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.baseline_mib = 0.0
        self.peak_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _rss_mib(self) -> float:
        with open("/proc/self/statm", encoding="ascii") as statm:
            resident_pages = int(statm.read().split()[1])
        return resident_pages * self._PAGE_SIZE / (1024 * 1024)

    def _sample(self) -> None:
        self.peak_mib = max(self.peak_mib, self._rss_mib())

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    @property
    def increase_mib(self) -> float:
        return max(0.0, self.peak_mib - self.baseline_mib)

    def __enter__(self):
        self.baseline_mib = self._rss_mib()
        self.peak_mib = self.baseline_mib
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ("git", "rev-parse", "--short", "HEAD"),
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.call(
            ("git", "diff", "--quiet"),
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "input",
        nargs="?",
        default="data/rti_normal.npy",
        help="tensor file (.npy or .pt/.pth); default data/rti_normal.npy",
    )
    ap.add_argument("--gnn-checkpoint", default="checkpoints/d64.pt")
    ap.add_argument(
        "--subset-edge",
        type=int,
        default=64,
        help="edge of the centred n-D hypercube to benchmark "
        "(capped to each axis, floored to a multiple of "
        "2 ** --levels)",
    )
    ap.add_argument("--eb", type=float, default=1e-4)
    ap.add_argument(
        "--rel", action="store_true", help="scale eb by the value range (max-min)"
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="min-max scale the subset to [0,1] before compressing",
    )
    # Codec knobs (defaults = the realistic 4-D level-5 case).
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--radius", type=int, default=1 << 15)
    ap.add_argument("--zstd-level", type=int, default=9)
    ap.add_argument("--eb-ratio", type=float, default=None)
    ap.add_argument("--tune", default="fast", choices=("fast", "size"))
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="GPU sampling period in seconds",
    )
    ap.add_argument(
        "--label", default="", help="free-text tag echoed in the report header"
    )
    args = ap.parse_args(argv)

    os.environ.setdefault("DEEPSZ_PROGRESS", "0")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(device)
    is_cuda = torch_device.type == "cuda"

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    raw = load_tensor(args.input)
    anchor_stride = 1 << args.levels
    sub = centred_subset(np.asarray(raw), args.subset_edge, anchor_stride)
    if sub.dtype == np.float64:
        print("note: float64 input cast to float32")
        sub = sub.astype(np.float32)
    orig_bytes = sub.nbytes
    if args.normalize:
        lo, hi = float(sub.min()), float(sub.max())
        sub = (sub.astype(np.float32) - lo) / max(hi - lo, 1e-12)
        print(f"normalized [{lo:.4g},{hi:.4g}] -> [0,1]")
    span = max(float(sub.max()) - float(sub.min()), 1.0)
    eb = args.eb * span if args.rel else args.eb

    from deepsz.gnn_codec import GNNCompressorCodec

    # GNNCompressorCodec normalizes internally and treats error_bound as
    # relative to (max - min); convert this script's absolute eb (used below
    # for the PASS/FAIL report) back to that.
    codec = GNNCompressorCodec(
        args.gnn_checkpoint,
        error_bound=eb / span,
        levels=args.levels,
        radius=args.radius,
        zstd_level=args.zstd_level,
        eb_ratio=args.eb_ratio,
        tune=args.tune,
        chunk_size=args.chunk_size,
        fp16=args.fp16,
        compile=args.compile,
        device=device,
    )

    if is_cuda:
        torch.cuda.synchronize(torch_device)
        gpu_base_alloc = torch.cuda.memory_allocated(torch_device)
        gpu_base_resv = torch.cuda.memory_reserved(torch_device)
        torch.cuda.reset_peak_memory_stats(torch_device)
    else:
        gpu_base_alloc = gpu_base_resv = 0

    with HostMemorySampler() as host_memory:
        gpu_index = torch_device.index or 0
        with GpuSampler(device_index=gpu_index, interval=args.poll_interval) as gpu:
            sync()
            t0 = time.perf_counter()
            stream = codec.compress(sub)
            sync()
            t_comp = time.perf_counter() - t0

            t0 = time.perf_counter()
            rec = codec.uncompress(stream).numpy().reshape(sub.shape)
            sync()
            t_dec = time.perf_counter() - t0

    # Quality / size.
    a = sub.astype(np.float64)
    r = rec.astype(np.float64)
    max_err = float(np.abs(a - r).max())
    mse = float(np.mean((a - r) ** 2))
    peak = max(float(sub.max()) - float(sub.min()), 1e-12)
    psnr = 10 * np.log10(peak**2 / mse) if mse > 0 else float("inf")
    nbytes = len(stream)
    bpv = 8 * nbytes / sub.size
    ratio = orig_bytes / nbytes
    mvox_s = sub.size / t_comp / 1e6

    # Resources.
    gpu_peak_alloc = (
        torch.cuda.max_memory_allocated(torch_device) / (1024 * 1024)
        if is_cuda
        else None
    )
    gpu_peak_resv = (
        torch.cuda.max_memory_reserved(torch_device) / (1024 * 1024)
        if is_cuda
        else None
    )
    gpu_increase_alloc = (
        (torch.cuda.max_memory_allocated(torch_device) - gpu_base_alloc) / (1024 * 1024)
        if is_cuda
        else None
    )
    gpu_increase_resv = (
        (torch.cuda.max_memory_reserved(torch_device) - gpu_base_resv) / (1024 * 1024)
        if is_cuda
        else None
    )
    gpu_stats = gpu.summary()

    # Human report.
    w = 24
    tag = f" [{args.label}]" if args.label else ""
    print(f"\n=== GNN subset benchmark ({git_commit()}){tag} ===")
    print(
        f"{'input':<{w}} {args.input} {tuple(sub.shape)} "
        f"({sub.size} voxels, {orig_bytes} B)"
    )
    print(f"{'eb':<{w}} {eb:g}   device {device}")
    print(
        f"{'levels/stride':<{w}} {args.levels}/{anchor_stride}"
        f"   chunk {args.chunk_size} "
        f"tune {args.tune} fp16 {args.fp16} compile {args.compile}"
    )
    print("-" * 60)
    print(f"{'PSNR':<{w}} {psnr:8.3f} dB")
    print(
        f"{'max error':<{w}} {max_err:.4g}  "
        f"({'PASS' if max_err <= eb else 'FAIL'} vs eb {eb:g})"
    )
    print(f"{'bits/value':<{w}} {bpv:8.4f} bpv   ratio {ratio:.2f}x  ({nbytes} B)")
    print(f"{'compress':<{w}} {t_comp:8.3f} s   ({mvox_s:.2f} Mvox/s)")
    print(f"{'decompress':<{w}} {t_dec:8.3f} s")
    print(
        f"{'peak host RAM':<{w}} {host_memory.peak_mib:8.1f} MiB  "
        f"(+{host_memory.increase_mib:.1f} MiB during roundtrip)"
    )
    if is_cuda:
        print(
            f"{'peak GPU alloc/resv':<{w}} {gpu_peak_alloc:8.1f} / "
            f"{gpu_peak_resv:.1f} MiB  (+{gpu_increase_alloc:.1f} / "
            f"+{gpu_increase_resv:.1f} MiB during roundtrip)"
        )
    if gpu_stats["gpu_samples"]:
        print(
            f"{'mean/peak GPU util':<{w}} "
            f"{gpu_stats['gpu_util_mean_pct']:8.1f} / "
            f"{gpu_stats['gpu_util_peak_pct']:.1f} %  "
            f"({gpu_stats['gpu_samples']} samples @ {args.poll_interval}s, "
            f"device peak used {gpu_stats['gpu_mem_peak_mib']} MiB, "
            f"+{gpu_stats['gpu_mem_increase_mib']} MiB)"
        )
    else:
        print(f"{'GPU util':<{w}} (no NVML / nvidia-smi samples)")

    if max_err > eb:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
