"""Synthetic scientific-like field generator used to train the GNN predictor
(scripts/train_gnn.py) without any natural images or external data: smooth
Gaussian random fields and power-law turbulence, with random-hyperplane
discontinuities. See ``sample_synthetic_batch`` for the full distribution.

Default distribution knobs are loaded from ``synthetic_dist.yaml`` at the
repo root so they can be tuned without touching code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

_YAML_PATH = Path(__file__).resolve().parent.parent / "synthetic_dist.yaml"


@dataclass(frozen=True)
class SyntheticDistConfig:
    crop: int
    frac: float
    correlation_2d: tuple[float, float]
    turbulence_frac: float
    max_discontinuities: int
    discontinuity_frac: float
    shape_4d: tuple[int, int, int, int]
    correlation_4d: tuple[float, float]
    batch_4d: int
    stride: int


def load_synthetic_dist(path: Path = _YAML_PATH) -> SyntheticDistConfig:
    data = yaml.safe_load(path.read_text())
    return SyntheticDistConfig(
        crop=data["crop"],
        frac=data["frac"],
        correlation_2d=tuple(data["correlation_2d"]),
        turbulence_frac=data["turbulence_frac"],
        max_discontinuities=data["max_discontinuities"],
        discontinuity_frac=data["discontinuity_frac"],
        shape_4d=tuple(data["shape_4d"]),
        correlation_4d=tuple(data["correlation_4d"]),
        batch_4d=data["batch_4d"],
        stride=data["stride"],
    )


DEFAULT_SYNTHETIC_DIST = load_synthetic_dist()


def _gaussian_smooth(x, sigma, shape, device):
    """Isotropic Gaussian blur of ``x`` (B, *shape) via the same mirror-doubled
    FFT product used for the base fields (exact reflect boundary, no seam)."""
    ndim = len(shape)
    dims = tuple(range(1, ndim + 1))
    for ax in dims:
        x = torch.cat([x, x.flip(ax)], dim=ax)
    spec = torch.fft.rfftn(x, dim=dims)
    for k, n in enumerate(shape):
        f = (torch.fft.rfftfreq if k == ndim - 1 else torch.fft.fftfreq)(
            2 * n, device=device
        )
        g = torch.exp(-2.0 * (math.pi * f) ** 2 * sigma**2)
        spec = spec * g.reshape(1, *(len(f) if a == k else 1 for a in range(ndim)))
    x = torch.fft.irfftn(spec, s=x.shape[1:], dim=dims)
    return x[(slice(None),) + tuple(slice(n) for n in shape)]


def _warp(field, disp, shape, device):
    """Backward-sample ``field`` (B, *shape) at ``grid + disp`` with n-D
    multilinear interpolation and a reflecting boundary. ``disp`` is (B, ndim,
    *shape) in cell units. Applied iteratively this advects the field along the
    (frozen) displacement flow — the mechanism that turns a phase-random field
    into one with coherent swirls and filaments."""
    B = field.shape[0]
    ndim = len(shape)
    strides = []
    s = 1
    for n in reversed(shape):
        strides.insert(0, s)
        s *= n
    base = torch.meshgrid(
        *[torch.arange(n, device=device, dtype=torch.float32) for n in shape],
        indexing="ij",
    )
    src = torch.stack(base)[None] + disp  # (B, ndim, *shape)
    lo = torch.floor(src)
    frac = src - lo
    lo = lo.long()
    flat = field.reshape(B, -1)
    out = torch.zeros_like(field)
    for corner in range(1 << ndim):  # 2**ndim interpolation corners
        w = torch.ones_like(field)
        flat_idx = torch.zeros(field.shape, dtype=torch.long, device=device)
        for k in range(ndim):
            bit = (corner >> k) & 1
            n = shape[k]
            ck = torch.remainder(lo[:, k] + bit, 2 * n)  # reflect into [0, n)
            ck = torch.where(ck >= n, 2 * n - 1 - ck, ck)
            w = w * (frac[:, k] if bit else 1.0 - frac[:, k])
            flat_idx = flat_idx + ck * strides[k]
        gathered = torch.gather(flat, 1, flat_idx.reshape(B, -1)).reshape(field.shape)
        out = out + w * gathered
    return out


def _turbulent_advect(field, is_turb, rng, shape, device, iters=3):
    """Give the ``is_turb`` rows of ``field`` coherent turbulent structure by
    advecting them through a smooth random flow (iterated domain warping).

    A power-law spectrum alone only makes a field *rougher* — its Fourier phases
    are still independent, so it has no eddies, filaments, or roll-ups. Warping
    the field along a smooth divergence-carrying displacement, repeated a few
    times, correlates those phases into the swirling, stretched, filamentary
    patterns turbulence actually produces. Non-turbulent rows get zero
    displacement (an exact identity warp), so they pass through unchanged."""
    B = field.shape[0]
    ndim = len(shape)
    mn = min(shape)
    amp = np.array(
        [float(rng.uniform(0.04, 0.10)) * mn if t else 0.0 for t in is_turb],
        dtype=np.float32,
    )
    if not amp.any():
        return field
    # One smooth vector flow per field: ndim components, correlation a sizable
    # fraction of the domain so the swirls are large-scale coherent structures.
    # A wider, larger-sigma band (vs. a lower displacement amp) biases toward
    # broader filaments and away from thin, heavily-stretched ones.
    flow_sigma = float(rng.uniform(0.18, 0.40) * mn)
    white = torch.from_numpy(
        rng.standard_normal((B * ndim, *shape)).astype(np.float32)
    ).to(device)
    flow = _gaussian_smooth(white, flow_sigma, shape, device).reshape(B, ndim, *shape)
    std = flow.reshape(B, ndim, -1).std(dim=2).clamp_min(1e-6)
    disp = flow / std.reshape(B, ndim, *([1] * ndim))
    disp = disp * torch.from_numpy(amp).to(device).reshape(B, 1, *([1] * ndim))
    for _ in range(iters):
        field = _warp(field, disp, shape, device)
    return field


def _add_discontinuities(fi, n_cuts, coords, rng, device):
    """Overlay ``n_cuts`` piecewise-constant jumps on the field ``fi`` (shape
    ``shape``) to inject sharp fronts — the shocks, material interfaces, and
    contact discontinuities that smooth Gaussian fields never contain but that
    dominate the residual (and the bit budget) in real scientific data.

    Each cut is a random hyperplane ``v · x = t``: one side gets a constant
    offset added, so the field steps discontinuously across an (n-1)-D face.
    ``coords`` holds the per-axis normalized [0, 1] coordinate grids (broadcast
    shape) so a single dot product gives the signed distance field. Jump sizes
    scale with the field's own spread so the fronts stay comparable in amplitude
    to the smooth variation regardless of the marginal warp."""
    if n_cuts <= 0:
        return fi
    ndim = len(coords)
    scale = float(fi.std()) or 1.0
    for _ in range(n_cuts):
        v = rng.standard_normal(ndim)
        v = v / (np.linalg.norm(v) + 1e-12)
        proj = sum(float(v[k]) * coords[k] for k in range(ndim))
        # Threshold at a random interior quantile so both sides carry area.
        q = float(rng.uniform(0.25, 0.75))
        t = torch.quantile(proj.reshape(-1), q)
        jump = float(rng.choice((-1.0, 1.0))) * float(rng.uniform(0.4, 1.5)) * scale
        fi = fi + jump * (proj > t).to(fi.dtype)
    return fi


def sample_synthetic_batch(
    batch,
    shape,
    correlation,
    rng=None,
    randomize=True,
    device="cpu",
    turbulence_frac=0.5,
    max_discontinuities=3,
    discontinuity_frac=1.0,
):
    """Return scientific-like random fields in ``[0, 1]``.

    Instead of one fixed texture, each field randomizes its generative knobs so
    the training distribution *covers* the variety real scientific fields show,
    rather than memorizing a single one — a narrow prior is exactly what a
    high-capacity model overfits (it then generalizes worse on out-of-domain
    eval even as train/in-domain loss improves). Per field we randomize:

      * spectrum & structure: with probability ``turbulence_frac`` the field is
        *turbulent* — a power-law (fractal / fBm-like) amplitude spectrum
        ``|k|**-p`` for the multi-scale energy cascade, then advected through a
        smooth random flow (iterated domain warping) to correlate its Fourier
        phases into coherent swirls, filaments, and roll-ups. The spectrum alone
        only makes a field rougher; the advection is what produces actual
        turbulent *patterns*. Otherwise the field is a *smooth* two-scale
        Gaussian random field (a fine scale plus a 2x-coarser one);
      * smoothness & anisotropy (smooth fields): a per-field base sigma drawn
        log-uniformly within the band set by ``correlation`` (its min/max); each
        axis then jitters off it by a per-field random spread, so the axis-ratio
        ranges from near-isotropic (spread~0) to strongly anisotropic;
      * value marginal: a random monotone warp (identity / signed power / tanh)
        yields unimodal, skewed, or bimodal histograms;
      * discontinuities: with probability ``discontinuity_frac`` the field gets
        1..``max_discontinuities`` random hyperplane fronts, adding
        piecewise-constant jumps (shocks / material interfaces), so the model
        also trains on the sharp edges smooth GRFs lack.

    Filtering runs as an FFT product on ``device`` (mirror-doubling each axis
    gives an exact 'reflect' boundary, avoiding an artificial periodic seam),
    so generation is GPU-fast and its cost is independent of sigma. Random
    knobs and the raw noise come from the numpy ``rng``, keeping seeded runs
    reproducible. ``randomize=False`` is the deterministic single-texture path
    (exact ``correlation`` per axis, smooth Gaussian marginal, no turbulence or
    discontinuities) for diagnostics.
    """
    shape = tuple(int(n) for n in shape)
    correlation = tuple(float(s) for s in correlation)
    rng = np.random.default_rng() if rng is None else rng
    ndim = len(shape)
    lo_s, hi_s = min(correlation) * 0.5, max(correlation)
    sigmas, warps, turb, exps, cuts = [], [], [], [], []
    for _ in range(batch):
        if randomize:
            base = float(np.exp(rng.uniform(np.log(lo_s), np.log(hi_s))))
            spread = float(rng.uniform(0.0, 1.0))  # 0 isotropic -> large aniso
            sigmas.append(
                [
                    max(0.3, base * float(np.exp(rng.normal(0.0, spread))))
                    for _ in range(ndim)
                ]
            )
            kind = int(rng.integers(3))
            param = (
                float(rng.uniform(0.4, 3.0))
                if kind == 1  # power
                else float(rng.uniform(1.5, 4.0))
            )  # tanh gain
            warps.append((kind, param))
            turb.append(bool(rng.random() < turbulence_frac))
            exps.append(float(rng.uniform(0.7, 1.6)))  # |k|**-p amplitude falloff
            if max_discontinuities > 0 and rng.random() < discontinuity_frac:
                cuts.append(int(rng.integers(1, int(max_discontinuities) + 1)))
            else:
                cuts.append(0)
        else:
            sigmas.append(list(correlation))
            warps.append((0, 0.0))
            turb.append(False)
            exps.append(1.0)
            cuts.append(0)
    # Fine noise plus an independent 2x-coarser realization, filtered in one
    # batched FFT: rows [0, B) are fine, rows [B, 2B) their coarse partners.
    # Each row's filter is either its anisotropic Gaussian (smooth fields) or a
    # radial power law (turbulent fields), selected per row below.
    sig = torch.tensor(
        sigmas + [[2.0 * s for s in row] for row in sigmas],
        dtype=torch.float32,
        device=device,
    )
    is_turb = torch.tensor(turb + turb, dtype=torch.bool, device=device)
    p = torch.tensor(exps + exps, dtype=torch.float32, device=device)
    x = torch.from_numpy(
        rng.standard_normal((2 * batch, *shape)).astype(np.float32)
    ).to(device)
    dims = tuple(range(1, ndim + 1))
    for ax in dims:  # even extension -> exact 'reflect'
        x = torch.cat([x, x.flip(ax)], dim=ax)
    spec = torch.fft.rfftn(x, dim=dims)
    # Build the per-row filter over the full rfft grid: gaussian exponent
    # (separable sum) and radial |k|^2 accumulate axis by axis.
    gauss_exp = torch.zeros((2 * batch,) + spec.shape[1:], device=device)
    kmag2 = torch.zeros(spec.shape[1:], device=device)
    for k, n in enumerate(shape):
        f = (torch.fft.rfftfreq if k == ndim - 1 else torch.fft.fftfreq)(
            2 * n, device=device
        )
        bshape = tuple(len(f) if a == k else 1 for a in range(ndim))
        f2 = (f**2).reshape(bshape)
        gauss_exp = gauss_exp + f2[None] * sig[:, k].reshape(-1, *([1] * ndim)) ** 2
        kmag2 = kmag2 + f2
    gauss = torch.exp(-2.0 * math.pi**2 * gauss_exp)
    # Radial power law amplitude; DC (k=0) forced to 0 so the field is centred.
    turb_filt = (kmag2[None] + 1e-8) ** (-0.5 * p.reshape(-1, *([1] * ndim)))
    turb_filt = torch.where(
        kmag2[None] > 0, turb_filt, torch.zeros_like(turb_filt)
    )
    filt = torch.where(is_turb.reshape(-1, *([1] * ndim)), turb_filt, gauss)
    spec = spec * filt
    x = torch.fft.irfftn(spec, s=x.shape[1:], dim=dims)
    x = x[(slice(None),) + tuple(slice(n) for n in shape)]
    field = x[:batch] + 0.5 * x[batch:]
    # Advect the turbulent rows along a smooth flow so they gain coherent
    # swirl/filament structure rather than just a rougher spectrum.
    if randomize:
        field = _turbulent_advect(field, turb, rng, shape, device)
    # Normalized [0, 1] coordinate grids, broadcast-shaped, for hyperplane cuts.
    coords = [
        torch.linspace(0.0, 1.0, n, device=device).reshape(
            tuple(n if a == k else 1 for a in range(ndim))
        )
        for k, n in enumerate(shape)
    ]
    fields = []
    for i in range(batch):
        fi = field[i]
        s = float(fi.std())
        if s >= 1e-8:
            z = (fi - fi.mean()) / s
            kind, param = warps[i]
            if kind == 1:  # <1 peaks, >1 heavy tails
                z = z.sign() * z.abs() ** param
            elif kind == 2:  # push toward two phases
                z = torch.tanh(param * z)
            fi = z
        fi = _add_discontinuities(fi, cuts[i], coords, rng, device)
        # Robust scaling retains local variation when one realization contains
        # a rare extreme. Values match the normalized [0, 1] training range.
        lo, hi = torch.quantile(
            fi.reshape(-1), torch.tensor([0.01, 0.99], device=device)
        )
        if float(hi) <= float(lo):  # realistically only degenerate tiny shapes
            fi = torch.full(shape, 0.5, device=device)
        else:
            fi = ((fi - lo) / (hi - lo)).clamp(0.0, 1.0)
        fields.append(fi)
    return torch.stack(fields).reshape(batch, -1)


def mixed_batch_sizes(
    crop, synthetic_shape, synthetic_fraction, field2d_batch, synthetic_batch
):
    """Choose source batch sizes and return their actual point fraction.

    A 2-D crop and a 4-D field are indivisible training examples, so arbitrary
    fractions can only be approximated at that granularity. ``synthetic_batch``
    fixes the number of 4-D fields; the 2-D-field batch is derived to make the
    fraction of scalar points as close as possible to ``synthetic_fraction``. At
    the two endpoints the explicitly configured batch size for the active source
    is retained.
    """
    fraction = float(synthetic_fraction)
    if fraction == 0.0:
        return int(field2d_batch), 0, 0.0
    if fraction == 1.0:
        return 0, int(synthetic_batch), 1.0
    field2d_points = int(crop) ** 2
    field_points = math.prod(int(n) for n in synthetic_shape)
    synthetic_batch = int(synthetic_batch)
    target_2d_points = synthetic_batch * field_points * (1.0 - fraction) / fraction
    field2d_batch = max(1, round(target_2d_points / field2d_points))
    ns = synthetic_batch * field_points
    ni = field2d_batch * field2d_points
    return field2d_batch, synthetic_batch, ns / (ni + ns)


def make_eval_field(
    shape, device, correlation, turbulence_frac, max_disc, discontinuity_frac=1.0, seed=12345
):
    """Fixed held-out 2-D synthetic field as ((1, N), (h, w)).

    Drawn from the same generator as the training data (turbulence +
    discontinuities included), but from a dedicated fixed-seed RNG so the eval
    field is identical across steps and across runs — a stable yardstick that is
    still representative of the training distribution rather than a single
    natural image.
    """
    eval_rng = np.random.default_rng(seed)
    x = sample_synthetic_batch(
        1,
        shape,
        correlation,
        eval_rng,
        device=device,
        turbulence_frac=turbulence_frac,
        max_discontinuities=max_disc,
        discontinuity_frac=discontinuity_frac,
    )
    return x, tuple(int(n) for n in shape)
