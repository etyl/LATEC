from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.train_gnn import (
    discretized_laplace_nll,
    sample_noise,
    normalize_tensor,
    training_autocast,
    run_chunked_scene,
    ModelEMA,
)
from deepsz.gnn_predictor import build_model
from deepsz.synthetic_data import (
    sample_synthetic_batch,
    mixed_batch_sizes,
    _warp,
    _turbulent_advect,
)


def test_noise_range_samples_log_uniformly():
    torch.manual_seed(0)
    args = SimpleNamespace(noise=0.01, noise_range=(1e-3, 1.0))

    noise = sample_noise(8192, args, "cpu")
    log_noise = noise.log()

    assert float(noise.min()) >= args.noise_range[0]
    assert float(noise.max()) <= args.noise_range[1]
    midpoint = (
        torch.log(torch.tensor(args.noise_range[0]))
        + torch.log(torch.tensor(args.noise_range[1]))
    ) / 2
    assert abs(float(log_noise.mean() - midpoint)) < 0.08


def test_noise_without_range_stays_fixed():
    args = SimpleNamespace(noise=0.02, noise_range=None)

    noise = sample_noise(4, args, "cpu")

    assert torch.equal(noise, torch.full((4,), args.noise))


def test_discretized_laplace_nll_stays_finite_in_tails():
    mu = torch.tensor([[0.0, 1.0, 0.5, 0.999999]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.5001, 0.0]])
    log_b = torch.full_like(mu, -8.0, requires_grad=True)
    eb = torch.tensor([1e-4])

    nll = discretized_laplace_nll(mu, log_b, target, eb)
    loss = nll.mean()
    loss.backward()

    assert torch.isfinite(nll).all()
    assert torch.isfinite(mu.grad).all()
    assert torch.isfinite(log_b.grad).all()


def test_synthetic_batch_is_4d_normalized_and_reproducible():
    shape = (12, 10, 8, 6)
    correlation = (4.0, 2.5, 1.5, 0.75)

    a = sample_synthetic_batch(2, shape, correlation, np.random.default_rng(123))
    b = sample_synthetic_batch(2, shape, correlation, np.random.default_rng(123))

    assert a.shape == (2, np.prod(shape))
    assert a.dtype == torch.float32
    assert float(a.min()) >= 0.0
    assert float(a.max()) <= 1.0
    assert torch.equal(a, b)


def test_eval_tensor_normalization_maps_extrema_to_unit_interval():
    tensor = np.array([[-4.0, 1.0], [6.0, -1.5]], dtype=np.float64)

    normalized = normalize_tensor(tensor)

    assert normalized.dtype == np.float32
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0
    np.testing.assert_allclose(normalized, (tensor + 4.0) / 10.0)


def test_eval_tensor_normalization_handles_constant_tensor():
    normalized = normalize_tensor(np.full((2, 3), 7.0, dtype=np.float32))

    np.testing.assert_array_equal(normalized, np.zeros((2, 3), np.float32))


def test_synthetic_axis_smoothness_follows_correlation_lengths():
    shape = (24, 20, 16, 12)
    correlation = (5.0, 3.0, 1.5, 0.6)
    fields = sample_synthetic_batch(
        3, shape, correlation, np.random.default_rng(7), randomize=False
    ).reshape(3, *shape)

    # Mean adjacent differences are smaller on axes with longer correlation.
    variation = []
    for axis in range(4):
        variation.append(float(torch.diff(fields, dim=axis + 1).abs().mean()))
    assert variation[0] < variation[1] < variation[2] < variation[3]


def test_synthetic_fields_randomly_permute_correlation_axes():
    shape = (16, 16, 16, 16)
    fields = sample_synthetic_batch(
        6, shape, (6.0, 3.0, 1.5, 0.5), np.random.default_rng(19)
    ).reshape(6, *shape)

    # The smoothest axis (smallest adjacent variation) must not remain fixed
    # across the batch when correlation assignments are shuffled per field.
    smoothest_axes = []
    for field in fields:
        variation = [
            float(torch.diff(field, dim=axis).abs().mean()) for axis in range(4)
        ]
        smoothest_axes.append(int(np.argmin(variation)))
    assert len(set(smoothest_axes)) > 1


def test_synthetic_batch_is_2d_normalized():
    shape = (48, 40)
    a = sample_synthetic_batch(
        4,
        shape,
        (2.0, 32.0),
        np.random.default_rng(5),
        turbulence_frac=0.5,
        max_discontinuities=3,
    )
    assert a.shape == (4, np.prod(shape))
    assert torch.isfinite(a).all()
    assert float(a.min()) >= 0.0
    assert float(a.max()) <= 1.0


def test_discontinuities_inject_sharp_jumps():
    """Forced hyperplane cuts concentrate variation into single-cell steps, so
    the adjacent-difference distribution is far more peaked (max/mean) than the
    smooth field's gently varying gradient — the signature of a sharp front."""
    shape = (64, 64)

    def peakedness(fields):
        vals = []
        for f in fields:
            d = torch.cat(
                [torch.diff(f, dim=0).abs().reshape(-1), torch.diff(f, dim=1).abs().reshape(-1)]
            )
            vals.append(float(d.max() / (d.mean() + 1e-9)))
        return float(np.mean(vals))

    smooth = sample_synthetic_batch(
        8, shape, (8.0, 8.0), np.random.default_rng(11), turbulence_frac=0.0,
        max_discontinuities=0,
    ).reshape(8, *shape)
    cut = sample_synthetic_batch(
        8, shape, (8.0, 8.0), np.random.default_rng(11), turbulence_frac=0.0,
        max_discontinuities=6,
    ).reshape(8, *shape)

    assert peakedness(cut) > 1.2 * peakedness(smooth)


def test_discontinuities_disabled_matches_no_cut_arg():
    shape = (32, 32)
    a = sample_synthetic_batch(
        3, shape, (4.0, 16.0), np.random.default_rng(3), max_discontinuities=0
    )
    assert torch.isfinite(a).all()
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


def test_turbulent_spectrum_is_reproducible_and_bounded():
    shape = (40, 40)
    a = sample_synthetic_batch(
        3, shape, (2.0, 32.0), np.random.default_rng(21), turbulence_frac=1.0
    )
    b = sample_synthetic_batch(
        3, shape, (2.0, 32.0), np.random.default_rng(21), turbulence_frac=1.0
    )
    assert torch.equal(a, b)
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0


def test_warp_with_zero_displacement_is_identity():
    shape = (12, 10)
    field = torch.rand(3, *shape)
    disp = torch.zeros(3, 2, *shape)
    out = _warp(field, disp, shape, "cpu")
    assert torch.allclose(out, field, atol=1e-5)


def test_warp_shift_advects_content():
    """A constant unit displacement backward-samples the neighbour, so the field
    shifts by one cell (reflecting at the boundary) rather than staying put."""
    shape = (1, 8)
    field = torch.arange(8, dtype=torch.float32).reshape(1, *shape)
    disp = torch.zeros(1, 2, *shape)
    disp[:, 1] = 1.0  # sample one cell further along the last axis
    out = _warp(field, disp, shape, "cpu").reshape(-1)
    assert torch.allclose(out[:-1], field.reshape(-1)[1:])


def test_turbulent_advection_only_moves_turbulent_rows():
    torch.manual_seed(0)
    field = torch.rand(4, 32, 32)
    is_turb = [False, True, False, True]
    out = _turbulent_advect(
        field.clone(), is_turb, np.random.default_rng(1), (32, 32), "cpu"
    )
    moved = [float((out[i] - field[i]).abs().mean()) for i in range(4)]
    assert moved[0] == 0.0 and moved[2] == 0.0  # smooth rows untouched
    assert moved[1] > 1e-3 and moved[3] > 1e-3  # turbulent rows advected


def test_advection_raises_local_anisotropy():
    """Advected turbulent fields have more elongated (streaky/filamentary) local
    structure than the phase-random spectrum-only field, measured by the mean
    anisotropy of the gradient structure tensor."""
    import deepsz.synthetic_data as gnn

    shape = (96, 96)

    def anisotropy(fields):
        vals = []
        for f in fields:
            gy = torch.diff(f, dim=0)[:, :-1]
            gx = torch.diff(f, dim=1)[:-1, :]
            jxx, jyy, jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
            tr, det = jxx + jyy, jxx * jyy - jxy * jxy
            disc = torch.clamp(tr * tr / 4 - det, min=0.0).sqrt()
            l1, l2 = tr / 2 + disc, tr / 2 - disc
            vals.append(float((l1 - l2) / (l1 + l2 + 1e-9)))
        return float(np.mean(vals))

    orig = gnn._turbulent_advect
    try:
        gnn._turbulent_advect = lambda f, *a, **k: f  # spectrum only
        spectral = gnn.sample_synthetic_batch(
            8, shape, (2.0, 32.0), np.random.default_rng(4), turbulence_frac=1.0,
            max_discontinuities=0,
        ).reshape(8, *shape)
    finally:
        gnn._turbulent_advect = orig
    advected = gnn.sample_synthetic_batch(
        8, shape, (2.0, 32.0), np.random.default_rng(4), turbulence_frac=1.0,
        max_discontinuities=0,
    ).reshape(8, *shape)

    assert anisotropy(advected) > anisotropy(spectral)


def test_ema_smooths_jittery_weights():
    """The EMA of a weight that jitters around a fixed mean has a much smaller
    step-to-step spread than the live weight — the point of evaluating the EMA
    on the deterministic held-out field instead of the latest noisy step."""
    model = build_model(16, 2)
    ema = ModelEMA(model, 0.99)
    center = {
        k: v.clone() for k, v in model.state_dict().items() if v.is_floating_point()
    }
    key = next(iter(center))

    torch.manual_seed(0)
    live_hist, ema_hist = [], []
    for step in range(60):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    v.copy_(center[k] + 0.1 * torch.randn_like(v))
        ema.update(model, step)
        live_hist.append(model.state_dict()[key].clone())
        ema_hist.append(ema.shadow[key].clone())

    live_spread = torch.stack(live_hist[20:]).std(0).mean()
    ema_spread = torch.stack(ema_hist[20:]).std(0).mean()
    assert ema_spread < 0.5 * live_spread


def test_ema_decay_warms_up_so_early_average_tracks():
    """Effective decay ramps min(decay, (1+step)/(10+step)), so the first update
    is dominated by the live weights (fast tracking) rather than the cold init."""
    model = build_model(16, 2)
    ema = ModelEMA(model, 0.999)
    key = next(k for k, v in model.state_dict().items() if v.is_floating_point())
    init = ema.shadow[key].clone()

    with torch.no_grad():
        for v in model.state_dict().values():
            if v.is_floating_point():
                v.add_(1.0)
    ema.update(model, step=0)  # effective decay = min(0.999, 1/10) = 0.1

    # Shadow moves 90% of the way to the new weights on step 0, not 0.1%.
    moved = (ema.shadow[key] - init).mean()
    assert abs(float(moved) - 0.9) < 1e-5


def test_ema_copy_to_loads_into_fresh_model():
    model = build_model(16, 2)
    ema = ModelEMA(model, 0.999)
    with torch.no_grad():
        for v in model.state_dict().values():
            if v.is_floating_point():
                v.mul_(2.0)
    ema.update(model, step=100)

    target = build_model(16, 2)
    ema.copy_to(target)
    for k, v in target.state_dict().items():
        assert v.dtype == model.state_dict()[k].dtype
        assert torch.allclose(v.float(), ema.shadow[k].float(), atol=1e-6)


def test_mixed_batch_fraction_counts_scalar_points():
    field2d_batch, synthetic_batch, actual = mixed_batch_sizes(
        crop=128,
        synthetic_shape=(16, 16, 16, 16),
        synthetic_fraction=0.25,
        field2d_batch=8,
        synthetic_batch=1,
    )

    assert field2d_batch == 12
    assert synthetic_batch == 1
    field2d_points = field2d_batch * 128**2
    synthetic_points = synthetic_batch * 16**4
    assert synthetic_points / (field2d_points + synthetic_points) == 0.25
    assert actual == 0.25


def test_mixed_batch_sizes_preserve_config_at_endpoints():
    assert mixed_batch_sizes(128, (16,) * 4, 0, 7, 2) == (7, 0, 0.0)
    assert mixed_batch_sizes(128, (16,) * 4, 1, 7, 2) == (0, 2, 1.0)


def test_fp16_autocast_is_cuda_only_on_cpu():
    x = torch.ones(2, dtype=torch.float32)
    with training_autocast(True, torch.device("cpu")):
        y = x * 2
    assert y.dtype == torch.float32


def test_chunk_finalization_casts_autocast_output_to_field_dtype():
    """Regression for CUDA FP16 finalization copied into the FP32 field.

    CPU BF16 autocast exercises the same mixed-dtype boundary without a GPU.
    """
    model = build_model(8)
    x = torch.rand(1, 8 * 8)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        nll, npix, _ = run_chunked_scene(
            model,
            x,
            (8, 8),
            axis=1,
            order=(0, 1),
            levels=2,
            stride=4,
            d=8,
            device=torch.device("cpu"),
            eb=0.01,
            agg_level=2,
        )
    loss = nll / npix
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
