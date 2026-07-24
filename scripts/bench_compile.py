"""When does ``torch.compile`` pay off for the chunked GNN encode?

The codec decides ``compile`` per encode and freezes it into the stream, so the
break-even is *within a single compress call*: the one-off dynamo warmup must be
repaid by the per-wave savings over the chunk waves in that same encode. This
harness measures compress wall-time with compile on vs off across a sweep of
chunk counts and prints the crossover -- the number of chunks past which compile
wins -- so ``_COMPILE_MIN_CHUNKS`` in ``gnn_codec.py`` can be set from data.

Each (nchunks, compile) point runs in a *fresh subprocess* with an isolated
inductor cache dir, so compile is measured cold -- exactly the first-encode cost
the frozen decision pays. Model load is excluded from the timed region; only
``codec.compress`` is timed. Weights are irrelevant to speed, so a random-weight
v6 checkpoint is fine (pass CKPT= to use a trained one).

Env knobs (all optional):
  CKPT=checkpoints/rand_v6_d64.pt   D=64  AGG=2   (checkpoint arch for random ckpt)
  LEVELS=2   CHUNK=8   EB=1e-3   FP16=1   RANK=4
  NCHUNKS="1,8,16,32,64,128"        chunk-count sweep points
  MODES="off,default,reduce-overhead"  compile variants to compare per chunk count
                                    (reduce-overhead = CUDA graphs, the launch-bound win)
  CACHE_LIMIT=256                   torch._dynamo cache_size_limit for compile runs
  DYNAMIC=1                         torch.compile dynamic= (1=True symbolic-M graph,
                                    0=False specialize per shape). Measured: 0 is
                                    WORSE -- it compiles every distinct stage geometry
                                    (>420s timeout at 64 chunks) instead of amortizing.
  CHILD_TIMEOUT=300                 per-point subprocess timeout; inf on overrun
  REPS=1                            timed reps per point (min taken)

The CHUNK edge is held fixed (a realistic per-chunk cost); the tensor is grown to
hit each NCHUNKS target, so the sweep isolates "how many chunks", not chunk size.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CKPT = os.environ.get("CKPT", "checkpoints/rand_v6_d64.pt")
D = int(os.environ.get("D", "64"))
AGG = int(os.environ.get("AGG", "2"))
LEVELS = int(os.environ.get("LEVELS", "2"))
CHUNK = int(os.environ.get("CHUNK", "8"))
EB = float(os.environ.get("EB", "1e-3"))
FP16 = os.environ.get("FP16", "1") == "1"
RANK = int(os.environ.get("RANK", "4"))
CACHE_LIMIT = int(os.environ.get("CACHE_LIMIT", "8"))  # torch dynamo default
CHILD_TIMEOUT = float(os.environ.get("CHILD_TIMEOUT", "300"))
DYNAMIC = os.environ.get("DYNAMIC", "1") == "1"  # torch.compile dynamic= for all runs
REPS = int(os.environ.get("REPS", "1"))
NCHUNKS = [int(x) for x in os.environ.get("NCHUNKS", "1,8,16,32,64,128").split(",")]


def grid_for(nchunks: int, rank: int) -> tuple[int, ...]:
    """Factor ``nchunks`` into a near-cubic per-axis chunk grid of ``rank`` axes."""
    g = [1] * rank
    rem = nchunks
    ax = 0
    # greedy: peel the smallest prime factor onto the currently-smallest axis
    for p in _factors(rem):
        ax = int(np.argmin(g))
        g[ax] *= p
    return tuple(g)


def _factors(n: int) -> list[int]:
    out = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            out.append(d)
            n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out or [1]


def ensure_ckpt(path: str) -> str:
    """Create a random-weight v6 checkpoint on demand (speed is arch-bound)."""
    p = Path(path)
    if p.exists():
        return path
    import torch

    from deepsz.gnn_predictor import CKPT_VERSION, build_model

    torch.manual_seed(0)
    m = build_model(d=D, agg_level=AGG).eval()
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"d": m.d, "agg_level": AGG, "state_dict": m.state_dict(), "version": CKPT_VERSION},
        p,
    )
    print(f"  (wrote random-weight v6 checkpoint {path})")
    return path


# ---- child: measure one (nchunks, compile) point, print one CSV line ---------
def _run_point() -> None:
    import time
    import gc

    nchunks = int(os.environ["_BC_NCHUNKS"])
    mode = os.environ["_BC_MODE"]  # off | default | reduce-overhead

    import torch
    import torch._dynamo as dyn

    dyn.config.cache_size_limit = CACHE_LIMIT
    dyn.config.accumulated_cache_size_limit = max(CACHE_LIMIT * 4, 512)

    # Force the dynamic= setting of every torch.compile the codec issues, without
    # touching library code: dynamic=False specializes per shape (a bounded set of
    # ~stage geometries that repeat across same-edge chunks -> replay, no per-chunk
    # recompile), dynamic=True keeps one symbolic graph across M. DYNAMIC env picks.
    if not DYNAMIC:
        _orig_compile = torch.compile

        def _forced(fn=None, **kw):
            kw["dynamic"] = False
            return _orig_compile(fn, **kw) if fn is not None else _orig_compile(**kw)

        torch.compile = _forced

    import deepsz.gnn_codec as gcodec
    from deepsz.gnn_codec import GNNCompressorCodec, _read_stream

    # Neutralize the production gate so the compile flag alone decides: we are
    # measuring where that gate *should* sit, so it must not pre-empt the sweep.
    gcodec._COMPILE_MIN_CHUNKS = 0

    compile_on = mode != "off"
    # The codec reads DEEPSZ_COMPILE_MODE inside _maybe_compile: unset/"" -> plain
    # inductor fusion; "reduce-overhead" -> CUDA graphs (cuts the ~30 tiny per-wave
    # kernel launches, the launch-bound win that grows as chunks repeat the shape).
    if mode == "reduce-overhead":
        os.environ["DEEPSZ_COMPILE_MODE"] = "reduce-overhead"
    else:
        os.environ.pop("DEEPSZ_COMPILE_MODE", None)

    grid = grid_for(nchunks, RANK)
    shape = tuple(CHUNK * g for g in grid)
    x = np.random.RandomState(0).rand(*shape).astype(np.float32)

    codec = GNNCompressorCodec(
        CKPT, error_bound=EB, levels=LEVELS, chunk_size=CHUNK,
        fp16=FP16, compile=compile_on, gate=False,
    )
    best = float("inf")
    meta = None
    for _ in range(REPS):
        t = time.time()
        stream = codec.compress(x)
        best = min(best, time.time() - t)
        meta = _read_stream(stream)[0]
        gc.collect()
        torch.cuda.empty_cache()
    real_nch = int(np.prod([-(-n // e) for n, e in zip(shape, meta["chunks"])]))
    print(
        f"POINT nchunks={real_nch} mode={mode} dynamic={int(DYNAMIC)} "
        f"compiled={int(bool(meta.get('compiled')))} secs={best:.3f} "
        f"shape={'x'.join(map(str, shape))}",
        flush=True,
    )


def _spawn(nchunks: int, mode: str, cache_dir: Path) -> float:
    """Run one point in a fresh process; inf on timeout (compile storm) or crash."""
    env = dict(os.environ)
    env["_BC_CHILD"] = "1"
    env["_BC_NCHUNKS"] = str(nchunks)
    env["_BC_MODE"] = mode
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)  # cold compile per point
    try:
        out = subprocess.run(
            [sys.executable, __file__],
            env=env, capture_output=True, text=True, timeout=CHILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return float("inf")  # recompile storm never converged in CHILD_TIMEOUT
    for line in out.stdout.splitlines():
        if line.startswith("POINT "):
            fields = dict(kv.split("=", 1) for kv in line.split()[1:])
            return float(fields["secs"])
    sys.stderr.write(out.stdout + out.stderr)
    return float("nan")  # child crashed; keep the sweep going


def main() -> None:
    ensure_ckpt(CKPT)
    prec = "fp16" if FP16 else "fp32"
    modes = [m.strip() for m in os.environ.get("MODES", "off,default,reduce-overhead").split(",")]
    print(
        f"bench_compile: ckpt={CKPT} rank={RANK} levels={LEVELS} chunk_edge={CHUNK} "
        f"eb={EB} {prec} cache_limit={CACHE_LIMIT} reps={REPS}"
    )
    print(f"  chunk sweep: {NCHUNKS}  modes: {modes}  (cold subprocess per point)")
    tmp = Path(os.environ.get("BC_CACHE_ROOT", "/tmp/bc_inductor"))
    tmp.mkdir(parents=True, exist_ok=True)

    hdr = f"  {'nchunks':>8}" + "".join(f"{m:>16}" for m in modes)
    print("\n" + hdr)
    rows = []
    for nc in NCHUNKS:
        times = {}
        for m in modes:
            times[m] = _spawn(nc, m, tmp / f"n{nc}_{m}")
        base = times.get("off", min(times.values()))
        cells = "".join(
            f"{times[m]:>10.2f}({base / times[m]:>4.2f}x)" for m in modes
        )
        print(f"  {nc:>8}{cells}", flush=True)
        rows.append((nc, times))

    # crossover per compiled mode vs off
    print("\n  crossover vs off (speedup > 1.0 == compile wins):")
    for m in modes:
        if m == "off":
            continue
        wins = [nc for nc, t in rows if "off" in t and t["off"] / t[m] > 1.0]
        if wins:
            print(f"    {m}: wins from ~{min(wins)} chunks up "
                  f"-> suggest _COMPILE_MIN_CHUNKS >= {min(wins)} for this mode")
        else:
            print(f"    {m}: never beats off across {NCHUNKS} (keep compile off)")


if __name__ == "__main__":
    if os.environ.get("_BC_CHILD") == "1":
        _run_point()
    else:
        main()
