"""Per-level bpp + MSE breakdown, interpolation vs chunked GNN.

For every interpolation level we report:
  - bpp        : order-0 entropy of that level's quantized codes (+32 b/outlier)
                 = the ideal entropy-coder cost, free of rANS/zstd overhead.
  - pred-RMSE  : RMSE of the raw prediction (x - pred) BEFORE the residual is
                 coded. This is prediction QUALITY and is what drives bpp;
                 it is NOT bounded by eb.
  - rec-RMSE   : RMSE of the reconstruction (x - recon) AFTER coding. Bounded:
                 |x - recon| <= eb at every point, so rec-RMSE <= eb.
  - sat+/sat-% : fraction of coded points whose Laplace scale sits at the top
                 / bottom edge of the rANS scale grid (rans.SCALE_LO_DIV /
                 SCALE_HI_MULT = the head's delta clamp). High sat+ at low eb
                 means the required scale exceeds the grid ceiling (prediction
                 error stuck above it): widening the grid, not the head, is
                 the fix. Codecs that don't code a scale show "—".

Levels are tagged by the per-stage error bound (each interpolation level has a
distinct eb under the default schedule; the anchor pass = coarsest level).
Point counts per level are analytic (point_levels).

Purpose: locate where the GNN loses to interpolation across progressive levels.
Run in fp32 so the fp16 message-pass noise floor (~1e-3) no longer masks the
finest predictions.

Env knobs (all optional):
  N=64  EB=1e-4  LEVELS=5  STRIDE=32  BLOCK=1  CHUNK=32
  CODECS="interp,gnn"   TUNE=fast   FP16=0   DATA=<path.npy>
  CKPT=checkpoints/d64.pt
  WHATIF=1 : simulate a scale-gated interp fallback inside the GNN encode
             (if delta=log2(b/eb) < T, swap the GNN prediction for chunk-local
             cubic interp and code at scale b/2^shift), sweeping (T, shift)
             offline on ideal discretized-Laplace cost. Answers "what would the
             gate buy" before touching the codec.
"""

import os
import sys
import time
import gc
import collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepsz.quantizer import _recon_from_codes

CKPT = os.environ.get("CKPT", "checkpoints/d64.pt")
EB = float(os.environ.get("EB", "1e-4"))
N = int(os.environ.get("N", "64"))
LEVELS = int(os.environ.get("LEVELS", "5"))
STRIDE = int(os.environ.get("STRIDE", "32"))
BLOCK = int(os.environ.get("BLOCK", "1"))
CHUNK = int(os.environ.get("CHUNK", "32"))
TUNE = os.environ.get("TUNE", "fast")
FP16 = os.environ.get("FP16", "0") == "1"
CODECS = [
    c.strip() for c in os.environ.get("CODECS", "interp,gnn").split(",") if c.strip()
]
DATA = os.environ.get("DATA", "").strip()
WHATIF = os.environ.get("WHATIF", "0") == "1"
GATE = os.environ.get("GATE", "1") == "1"  # real in-codec scale gate (gnn)
NDIM = 4


def make_field(n):
    """Non-periodic synthetic RTI-like field, normalised to [0,1]."""
    u = [(np.arange(n, dtype=np.float32) / (n - 1)) for _ in range(NDIM)]
    ux, uy, uz, uw = (
        u[0].reshape(n, 1, 1, 1),
        u[1].reshape(1, n, 1, 1),
        u[2].reshape(1, 1, n, 1),
        u[3].reshape(1, 1, 1, n),
    )
    h = (
        0.5
        + 0.06 * np.sin(4.3 * np.pi * uy + 0.7) * np.sin(3.1 * np.pi * uz + 1.3)
        + 0.03 * np.sin(2.7 * np.pi * uw + 0.4)
    ).astype(np.float32)
    base = np.tanh((ux - h) / 0.08).astype(np.float32)
    turb = (
        0.12 * np.sin(5.7 * ux + 2.1 * uy + 3.3 * uz + 1.9 * uw)
        + 0.07 * np.sin(9.1 * ux - 4.4 * uy + 2.2 * uz)
        + 0.04 * np.sin(13.7 * uy + 6.6 * uz - 5.1 * uw)
    ).astype(np.float32)
    f = base + turb
    f = (f - f.min()) / (f.max() - f.min())
    return np.ascontiguousarray(f.astype(np.float32))


def load_field():
    if DATA:
        a = np.load(DATA).astype(np.float32)
        if a.ndim != NDIM:
            raise SystemExit(f"DATA must be {NDIM}-D, got shape {a.shape}")
        a = a[tuple(slice(0, N) for _ in range(NDIM))]
        a = (a - a.min()) / (a.max() - a.min())
        return np.ascontiguousarray(a.astype(np.float32))
    return make_field(N)


def analytic_pts(n):
    from deepsz.levels import point_levels

    c = np.indices((n,) * NDIM)
    lv = point_levels([c[i] for i in range(NDIM)], LEVELS, STRIDE, BLOCK)
    return np.bincount(lv.ravel(), minlength=LEVELS + 1)


class LevelStats:
    """Per-eb histogram of codes + running sums for pred/recon squared error."""

    def __init__(self):
        self.hist = collections.defaultdict(collections.Counter)
        self.nout = collections.defaultdict(int)
        self.sse_pred = collections.defaultdict(float)  # sum (x-pred)^2
        self.sse_rec = collections.defaultdict(float)  # sum (x-recon)^2
        self.npt = collections.defaultdict(int)
        # Laplace-scale saturation at the rANS grid edges (rans.SCALE_LO_DIV /
        # SCALE_HI_MULT = the head's delta clamp). sat_hi ~100% means the true
        # residual scale exceeds the grid ceiling: the eb-relative head can't
        # help those points, the window itself is too narrow.
        self.nsc = collections.defaultdict(int)
        self.sat_hi = collections.defaultdict(int)
        self.sat_lo = collections.defaultdict(int)

    def add_scale(self, eb, scale):
        e = round(float(eb), 15)
        s = np.asarray(scale, dtype=np.float64).ravel()
        self.nsc[e] += s.size
        from deepsz.rans import SCALE_HI_MULT, SCALE_LO_DIV

        self.sat_hi[e] += int((s >= SCALE_HI_MULT * float(eb) * 0.999).sum())
        self.sat_lo[e] += int((s <= float(eb) / SCALE_LO_DIV * 1.001).sum())

    def add(self, eb, x, pred, codes, radius):
        e = round(float(eb), 15)
        x = np.asarray(x, dtype=np.float64).ravel()
        pred = np.asarray(pred, dtype=np.float64).ravel()
        codes = np.asarray(codes).ravel()
        # ideal-entropy histogram
        vals, cnts = np.unique(codes, return_counts=True)
        c = self.hist[e]
        for v, nn in zip(vals.tolist(), cnts.tolist()):
            c[v] += nn
        is_out = codes == 0
        self.nout[e] += int(is_out.sum())
        # prediction error (drives bpp)
        self.sse_pred[e] += float(np.sum((x - pred) ** 2))
        # reconstruction error (bounded by eb): recon = decoder arithmetic,
        # outliers reproduce x exactly.
        recon = _recon_from_codes(
            pred.astype(np.float32), codes.astype(np.uint32), float(eb), int(radius)
        )
        recon = recon.astype(np.float64)
        recon[is_out] = x[is_out]
        self.sse_rec[e] += float(np.sum((x - recon) ** 2))
        self.npt[e] += x.size

    def per_level(self):
        out = {}
        for lv, e in enumerate(sorted(self.hist.keys())):  # smallest eb = level 0
            c = self.hist[e]
            counts = np.array(list(c.values()), dtype=np.float64)
            tot = counts.sum()
            p = counts / tot
            ent = float(-(counts * np.log2(p)).sum())
            bpp = (ent + 32.0 * self.nout[e]) / tot
            pred_rmse = (self.sse_pred[e] / self.npt[e]) ** 0.5
            rec_rmse = (self.sse_rec[e] / self.npt[e]) ** 0.5
            nsc = self.nsc[e]
            out[lv] = dict(
                bpp=bpp,
                ent_bpp=ent / tot,
                tot=int(tot),
                nout=self.nout[e],
                pred_rmse=pred_rmse,
                rec_rmse=rec_rmse,
                eb=e,
                sat_hi=100 * self.sat_hi[e] / nsc if nsc else None,
                sat_lo=100 * self.sat_lo[e] / nsc if nsc else None,
            )
        return out


def hook_all(stats):
    """Wrap `quantize` in every module that owns a reference, restore on exit."""
    import deepsz.codec as cc
    import deepsz.gnn_codec as gc

    mods = [cc, gc]
    saved = []
    for m in mods:
        if hasattr(m, "quantize"):
            orig = m.quantize
            saved.append((m, "quantize", orig))

            def make(orig):
                def q(x, pred, eb, radius=1 << 15, *a, **k):
                    codes, outliers = orig(x, pred, eb, radius, *a, **k)
                    stats.add(eb, x, pred, codes, radius)
                    return codes, outliers

                return q

            m.quantize = make(orig)
        if hasattr(m, "scale_to_level"):
            orig = m.scale_to_level
            saved.append((m, "scale_to_level", orig))

            def make_s(orig):
                def s(scale, eb, *a, **k):
                    stats.add_scale(eb, scale)
                    return orig(scale, eb, *a, **k)

                return s

            m.scale_to_level = make_s(orig)
    return saved


def unhook(saved):
    for m, name, orig in saved:
        setattr(m, name, orig)


def _laplace_bits(absr, b, eb):
    """Ideal discretized-Laplace cost/pt at coded scale b (clipped to the rANS
    grid), capped at 32 bits = the codec's raw-f32 outlier escape.
    ponytail: ignores the 64-level scale-grid rounding and rANS overhead; use
    it to compare columns within the what-if, not against the real stream."""
    from deepsz.rans import SCALE_HI_MULT, SCALE_LO_DIV

    b = np.clip(b.astype(np.float64), eb / SCALE_LO_DIV, eb * SCALE_HI_MULT)
    k = np.rint(np.abs(absr).astype(np.float64) / (2 * eb))
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        p = np.where(
            k == 0,
            -np.expm1(-eb / b),
            0.5 * np.exp(-((2 * k - 1) * eb / b)) * -np.expm1(-2 * eb / b),
        )
        bits = -np.log2(p)
    return np.minimum(bits, 32.0)


class WhatIfGate:
    """Scale-gated interp fallback, simulated during the real GNN encode.

    Gate: if delta = log2(b/eb) < T, replace the GNN prediction with cubic
    interp and code the interp residual at scale b/2^shift. Both sides of the
    gate are decoder-reproducible: b comes from the model, interp is computed
    chunk-locally from the same causal recon the wrapper sees here (the codec
    codes sub-stages sequentially, so interp's +-stride neighbours are already
    reconstructed, exactly like in the interp codec).

    Captures per-point (|r_gnn|, |r_interp|, b) per stage-eb; the (T, shift)
    sweep runs offline on bucketed sums, so memory stays O(levels * buckets)."""

    T_GRID = [2, 3, 4, 5, 6, 7, 8]
    SHIFTS = [0, 2, 4, 6]

    def __init__(self, field):
        self.field = np.ascontiguousarray(field)[None]  # (1, *shape)
        self.agg = {}  # eb -> dict of bucketed sums
        self._plans = {}

    def grab(self, pr, s, recon, eb, pred, scale):
        from deepsz.predictor import _interp_axis_at, default_interp_center

        for bi, ci in enumerate(pr._wave_ids):
            sls = tuple(pr.chunk_slices(ci))
            cshape = tuple(sl.stop - sl.start for sl in sls)
            if cshape not in self._plans:
                from deepsz.levels import stage_plan

                self._plans[cshape] = stage_plan(
                    cshape, pr.levels, pr.anchor_stride, pr.anchor_block
                )
            mask, stride, axes = self._plans[cshape][s]
            coords = np.nonzero(mask)
            W = recon[(slice(None), *sls)].astype(np.float64)
            center = default_interp_center(len(cshape))
            # 0 = the original linear/edge-copy ends, which is what the GNN
            # codec's device gate interps (`gnn_codec._interp_axis_at_t`) still
            # do; keep this comparison apples-to-apples with the deployed gate
            # rather than with the interp predictor's newer default.
            end_mode = 0
            if center == 0 or len(axes) == 1:
                ip = sum(
                    _interp_axis_at(W, coords, a, stride, "cubic", cshape, end_mode)
                    for a in axes
                ) / len(axes)
            else:
                a = axes[0] if center == 1 else axes[-1]
                ip = _interp_axis_at(W, coords, a, stride, "cubic", cshape, end_mode)
            truth = self.field[(slice(None), *sls)][:, mask][0]
            self._accum(
                float(eb),
                np.abs(truth - pred[bi]),
                np.abs(truth - ip[0].astype(np.float32)),
                np.asarray(scale[bi], np.float32),
            )

    def _accum(self, eb, r_g, r_i, b):
        e = round(eb, 15)
        nb = len(self.T_GRID) + 1
        a = self.agg.setdefault(
            e,
            dict(
                n=0,
                cnt=np.zeros(nb),
                base=np.zeros(nb),
                choice=0.0,
                oracle=0.0,
                sse_g=0.0,
                sse_i=0.0,
                i_bits={sh: np.zeros(nb) for sh in self.SHIFTS},
                mis={sh: np.zeros(nb) for sh in self.SHIFTS},
            ),
        )
        # rms(r_gnn) must reproduce the main table's pred-RMSE: it is the
        # built-in check that truth/pred/mask stayed aligned in grab().
        a["sse_g"] += float(np.square(r_g, dtype=np.float64).sum())
        a["sse_i"] += float(np.square(r_i, dtype=np.float64).sum())
        base = _laplace_bits(r_g, b, e)
        # bucket k: e*2^T_GRID[k-1] <= b < e*2^T_GRID[k]; gate at T_GRID[j]
        # captures buckets <= j, so every (T, shift) total is a prefix sum.
        bucket = np.digitize(b, e * np.exp2(self.T_GRID))
        a["n"] += b.size
        a["cnt"] += np.bincount(bucket, minlength=nb)
        a["base"] += np.bincount(bucket, weights=base, minlength=nb)
        for sh in self.SHIFTS:
            ib = _laplace_bits(r_i, b / 2.0**sh, e)
            a["i_bits"][sh] += np.bincount(bucket, weights=ib, minlength=nb)
            a["mis"][sh] += np.bincount(bucket, weights=(ib > base), minlength=nb)
        a["choice"] += float(np.minimum(base, _laplace_bits(r_i, b, e)).sum())
        r_min = np.minimum(r_g, r_i)
        a["oracle"] += float(_laplace_bits(r_min, r_min, e).sum())

    def report(self):
        npts = N**NDIM
        ebs = sorted(self.agg)
        base_bpv = sum(a["base"].sum() for a in self.agg.values()) / npts
        total = {}
        for j, T in enumerate(self.T_GRID):
            for sh in self.SHIFTS:
                bits = 0.0
                for a in self.agg.values():
                    g = slice(0, j + 1)
                    bits += (
                        a["i_bits"][sh][g].sum() + a["base"].sum() - a["base"][g].sum()
                    )
                total[(T, sh)] = bits / npts
        print(
            "\n===== what-if: scale-gated interp fallback "
            "(model-cost bpv, whole tensor) ====="
        )
        print("  gate: delta=log2(b/eb) < T -> code interp residual at scale b/2^shift")
        print(
            "  base (GNN, same cost model): "
            f"{base_bpv:.4f} bpv   choice-oracle "
            f"{sum(a['choice'] for a in self.agg.values()) / npts:.4f}   "
            "scale-oracle "
            f"{sum(a['oracle'] for a in self.agg.values()) / npts:.4f}"
        )
        print("  T\\shift" + "".join(f"{sh:>9}" for sh in self.SHIFTS))
        for T in self.T_GRID:
            print(
                f"  {T:>7}" + "".join(f"{total[(T, sh)]:>9.4f}" for sh in self.SHIFTS)
            )
        (bT, bsh), bbpv = min(total.items(), key=lambda kv: kv[1])
        print(f"  best: T={bT} shift={bsh} -> {bbpv:.4f} bpv (base {base_bpv:.4f})")
        j = self.T_GRID.index(bT)
        print(f"\n  per level at T={bT} shift={bsh}:")
        print(
            f"  {'lvl':>3}{'pts':>13}{'base-bpp':>10}{'gated-bpp':>11}"
            f"{'gated%':>8}{'misfire%':>10}{'rms(r_gnn)':>12}{'rms(r_int)':>12}"
        )
        for lv, e in enumerate(ebs):
            a = self.agg[e]
            g = slice(0, j + 1)
            ng = a["cnt"][g].sum()
            bits = a["i_bits"][bsh][g].sum() + a["base"].sum() - a["base"][g].sum()
            mis = a["mis"][bsh][g].sum()
            print(
                f"  {lv:>3}{a['n']:>13,}{a['base'].sum() / a['n']:>10.3f}"
                f"{bits / a['n']:>11.3f}{100 * ng / a['n']:>7.1f}%"
                f"{100 * mis / ng if ng else 0.0:>9.2f}%"
                f"{(a['sse_g'] / a['n']) ** 0.5:>12.2e}"
                f"{(a['sse_i'] / a['n']) ** 0.5:>12.2e}"
            )


def attach_whatif(wi):
    from deepsz.gnn_predictor import ChunkedGNNPredictor as _P

    orig = _P.predict_wave_stage

    def wrapped(self, s, recon, eb):
        pred, scale = orig(self, s, recon, eb)
        wi.grab(self, s, recon, eb, pred, scale)
        return pred, scale

    _P.predict_wave_stage = wrapped
    return [(_P, "predict_wave_stage", orig)]


def run_interp(field):
    import deepsz.codec as cc
    from deepsz.predictor import InterpPredictor

    pred = InterpPredictor(
        order="cubic", levels=LEVELS, anchor_stride=STRIDE, anchor_block=BLOCK
    )
    t0 = time.time()
    stream, _ = cc.compress(
        field,
        EB,
        pred,
        levels=LEVELS,
        anchor_stride=STRIDE,
        anchor_block=BLOCK,
        tune=TUNE,
    )
    return len(stream), time.time() - t0


def run_gnn(field):
    from deepsz.gnn_codec import GNNCompressorCodec

    kw = dict(
        error_bound=EB,
        levels=LEVELS,
        chunk_size=CHUNK,
        tune=TUNE,
        fp16=FP16,
        compile=False,
    )
    codec = None
    try:
        codec = GNNCompressorCodec(CKPT, gate=GATE, **kw)
        t0 = time.time()
        stream = codec.compress(field)
        return len(stream), time.time() - t0
    finally:
        del codec
        try:
            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def report(tag, total, dt, stats, pts):
    npts = N**NDIM
    pl = stats.per_level()
    print(
        f"\n### {tag}   {N}^4={npts:,}   {total:,} B   "
        f"bpv {8 * total / npts:.4f}   ratio(f32) {npts * 4 / total:.2f}   {dt:.1f}s"
    )
    print(
        f"  {'lvl':>3}{'stride':>7}{'pts':>13}{'pt%':>7}"
        f"{'bpp':>9}{'pred-RMSE':>12}{'rec-RMSE':>11}{'out%':>7}"
        f"{'sat+%':>7}{'sat-%':>7}{'lvl-bits':>13}{'bit%':>7}"
    )
    lvl_bits = {lv: pl[lv]["bpp"] * pts[lv] for lv in pl}
    tb = sum(lvl_bits.values())
    for lv in range(LEVELS + 1):
        if lv not in pl:
            continue
        d = pl[lv]
        stride = STRIDE >> lv if lv else STRIDE
        bits = lvl_bits[lv]
        sat = "".join(
            "      —" if d[k] is None else f"{d[k]:>6.2f}%"
            for k in ("sat_hi", "sat_lo")
        )
        print(
            f"  {lv:>3}{stride:>7}{pts[lv]:>13,}{100 * pts[lv] / npts:>6.2f}%"
            f"{d['bpp']:>9.3f}{d['pred_rmse']:>12.2e}{d['rec_rmse']:>11.2e}"
            f"{100 * d['nout'] / d['tot']:>6.2f}%{sat}"
            f"{int(bits):>13,}{100 * bits / tb:>6.2f}%"
        )
    print(
        f"      ideal-entropy total {tb / 8 / 1e3:.1f} KB ({tb / npts:.4f} bpv)  "
        f"vs real stream {total / 1e3:.1f} KB"
    )
    return pl


if __name__ == "__main__":
    prec = "fp16" if FP16 else "fp32"
    print(
        f"bench_levels: N={N} eb={EB} levels={LEVELS} stride={STRIDE} "
        f"chunk={CHUNK} tune={TUNE} precision={prec}"
    )
    print(f"  data={'synthetic RTI' if not DATA else DATA}  codecs={CODECS}")
    field = load_field()
    print(f"  field range [{field.min():.4f},{field.max():.4f}] std {field.std():.4f}")
    pts = analytic_pts(N)
    print("  pts/level: " + " ".join(f"L{i}:{c:,}" for i, c in enumerate(pts)))

    wi = WhatIfGate(field) if WHATIF and "gnn" in CODECS else None
    results = {}
    for codec in CODECS:
        st = LevelStats()
        saved = hook_all(st)
        if wi is not None and codec == "gnn":
            saved += attach_whatif(wi)
        try:
            if codec == "interp":
                tot, dt = run_interp(field)
            elif codec == "gnn":
                tot, dt = run_gnn(field)
            else:
                print(f"  (unknown codec {codec}, skipping)")
                continue
        finally:
            unhook(saved)
        results[codec] = report(codec.upper(), tot, dt, st, pts)

    if wi is not None and wi.agg:
        wi.report()

    # finest-level head-to-head (where ~90% of the bits live)
    if len(results) > 1:
        fin = LEVELS
        print(f"\n===== finest level L{fin} (stride 1) head-to-head =====")
        print(f"  {'codec':>8}{'bpp':>9}{'pred-RMSE':>12}{'rec-RMSE':>11}{'sat+%':>8}")
        for c in CODECS:
            if c in results and fin in results[c]:
                d = results[c][fin]
                sat = "       —" if d["sat_hi"] is None else f"{d['sat_hi']:>7.2f}%"
                print(
                    f"  {c:>8}{d['bpp']:>9.3f}{d['pred_rmse']:>12.2e}"
                    f"{d['rec_rmse']:>11.2e}{sat}"
                )
