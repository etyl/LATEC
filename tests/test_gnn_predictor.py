"""Tests for the dimension-agnostic GNN predictor (untrained-net smoke tests)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from latec.gnn_predictor import (
    CKPT_VERSION,
    ChunkedGNNPredictor,
    build_model,
    build_stage_geoms,
    half_directions,
    stage_forward,
)
from latec.levels import stage_masks


def _shift(arr, offset, fill):
    """result[p] = arr[p + offset], out-of-bounds filled with `fill`."""
    out = np.full_like(arr, fill)
    src, dst = [], []
    for o, n in zip(offset, arr.shape):
        if o >= 0:
            src.append(slice(o, n))
            dst.append(slice(0, n - o))
        else:
            src.append(slice(0, n + o))
            dst.append(slice(-o, n))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def _nearest_in_dir(known, flat, dvec, max_radius):
    """Step-by-step scan for the nearest known sample along +dvec, the
    reference the closed-form lattice geometry has to reproduce."""
    found_idx = np.zeros(known.shape, np.int64)
    found_dist = np.ones(known.shape, np.float32)
    found = np.zeros(known.shape, bool)
    if not known.any():
        return found_idx, found_dist, found
    limit = min(max_radius, max(known.shape))
    for step in range(1, limit + 1):
        off = tuple(step * c for c in dvec)
        new = _shift(known, off, False) & ~found
        if new.any():
            found_idx[new] = _shift(flat, off, -1)[new]
            found_dist[new] = step
            found |= new
            if found.all():
                break
    return found_idx, found_dist, found


def _scan_lines(known, qidx, max_radius):
    """(ip, in_, dp, dn, vp, vn) at the query points, by exhaustive scan."""
    flat = np.arange(known.size, dtype=np.int64).reshape(known.shape)
    cols = {k: [] for k in ("ip", "in_", "dp", "dn", "vp", "vn")}
    for h in half_directions(known.ndim):
        neg = tuple(-c for c in h)
        ip, dp, vp = _nearest_in_dir(known, flat, h, max_radius)
        in_, dn, vn = _nearest_in_dir(known, flat, neg, max_radius)
        for name, a in zip(cols, (ip, in_, dp, dn, vp, vn)):
            cols[name].append(a.reshape(-1)[qidx])
    return tuple(np.stack(cols[k], axis=0) for k in ("ip", "in_", "dp", "dn", "vp", "vn"))


@pytest.mark.parametrize(
    "shape,levels,stride,block,max_radius",
    [
        ((64, 64), 4, 16, 4, 64),
        ((64, 64), 4, 16, 1, 8),  # small radius exercises the `<= limit` gate
        ((70, 90), 3, 8, 1, 64),  # non-multiple-of-stride shape (edge clamping)
        ((16, 16, 16), 2, 4, 1, 16),  # 3-D
    ],
)
def test_geometry_matches_scan(shape, levels, stride, block, max_radius):
    """The load-bearing test: the closed-form lattice geometry must reproduce,
    bit for bit, the step-by-step neighbour scan (`_scan_lines`) it replaced
    — per stage, per half-direction, for (idx, dist, valid) on both sides."""
    masks = stage_masks(shape, levels, stride, block)
    geoms, _ = build_stage_geoms(shape, levels, stride, block, max_radius, torch)
    cum = np.zeros(shape, bool)  # known before the current stage
    gi = 0
    for mask in masks:
        if mask.any():
            g = geoms[gi]
            qidx = np.ravel_multi_index(np.nonzero(mask), shape)
            want = _scan_lines(cum, qidx, max_radius)
            assert np.array_equal(g.query_idx.numpy(), qidx)
            # `_StageGeom` stores steps and rebuilds (idx, dist) in `lines`; the
            # comparison is against the values, so it covers the derivation too.
            got = g.lines(0, g.M)
            for k, a, b in zip(("ip", "in_", "dp", "dn", "vp", "vn"), got, want):
                assert np.array_equal(a.numpy(), b), (k, gi)
            dirs = np.asarray(half_directions(len(shape)), np.float32)
            nnz = np.count_nonzero(dirs, axis=1).astype(np.float32)
            expected_cos = dirs / np.sqrt(nnz)[:, None]
            expected_lognnz = (0.5 * np.log2(nnz))[:, None]
            assert np.allclose(g.cos.numpy(), expected_cos)
            assert np.allclose(g.lognnz.numpy(), expected_lognnz)
            assert g.ndim == len(shape)
            gi += 1
        cum |= mask
    assert gi == len(geoms)


def test_half_directions_count():
    # one representative per line = (3^n - 1) / 2
    assert len(half_directions(2)) == 4
    assert len(half_directions(3)) == 13


def _run(model, recon, shape, levels, stride, block=1, max_radius=64):
    """One codec stage: reveal the anchors, then predict the next stage
    (mirrors the codec's first predict call)."""
    c = recon.shape[0]
    geoms, _ = build_stage_geoms(shape, levels, stride, block, max_radius, torch)
    gp, gh = geoms[0], geoms[1]
    x = torch.from_numpy(recon.reshape(c, -1).astype(np.float32))
    E = torch.zeros(c, int(np.prod(shape)), len(shape), model.d)
    with torch.no_grad():
        (values, log_b), _, _ = stage_forward(
            model, E, gp, gh, x[:, gp.idx_np], torch, eb=0.01
        )
    assert log_b.shape == values.shape
    return values.numpy()


def test_2d_smoke_and_determinism():
    model = build_model(d=16).eval()
    masks = stage_masks((64, 64), 4, 16, anchor_block=4)  # levels==log2(stride)
    known = masks[0]  # anchors known, predict the next stage
    recon = np.random.RandomState(0).rand(2, 64, 64).astype(np.float32)
    recon = recon * known[None]  # only known positions meaningful
    out = _run(model, recon, (64, 64), 4, 16, block=4)
    assert out.shape == (2, int(masks[1].sum()))
    assert np.isfinite(out).all()
    # deterministic
    assert np.array_equal(out, _run(model, recon, (64, 64), 4, 16, block=4))


def test_3d_dimension_agnostic():
    """Same weights (trained on 2-D) evaluate on a 3-D grid -> proves the
    network is dimension-generic."""
    model = build_model(d=16).eval()
    masks = stage_masks((16, 16, 16), 2, 4, 1)
    recon = np.random.RandomState(1).rand(1, 16, 16, 16).astype(np.float32)
    recon = recon * masks[0][None]  # sparse anchors
    out = _run(model, recon, (16, 16, 16), 2, 4, max_radius=16)
    assert out.shape == (1, int(masks[1].sum()))
    assert np.isfinite(out).all()


def test_no_neighbour_is_finite():
    """The anchor stage, predicted with nothing known, is finite (null token)."""
    model = build_model(d=16).eval()
    geoms, _ = build_stage_geoms((8, 8), 2, 4, 1, 64, torch)
    E = torch.zeros(1, 8 * 8, 2, model.d)
    with torch.no_grad():
        (values, _), _, _ = stage_forward(model, E, None, geoms[0], None, torch, eb=0.01)
    assert np.isfinite(values.numpy()).all()


def test_init_embed_uses_value_only():
    """The revealed-point embedding is conditioned only on the known value."""
    torch.manual_seed(0)
    model = build_model(d=16).eval()
    v = torch.full((2, 5, 1), 0.5)
    out = model.init(v)
    assert out.shape == (2, 5, model.d)
    assert model.init.net[0].in_features == 1


@pytest.mark.parametrize("version", [None, 2, 3, 4, 5, 6])
def test_predictor_rejects_old_checkpoint(tmp_path, version):
    model = build_model(d=8).eval()
    path = tmp_path / f"v{version or 1}.pt"
    checkpoint = {"d": model.d, "state_dict": model.state_dict()}
    if version is not None:
        checkpoint["version"] = version
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="format v7"):
        ChunkedGNNPredictor(path, 0.0, 1.0, levels=2, anchor_stride=4, anchor_block=1)


def test_predictors_share_loaded_inference_model(tmp_path):
    path = tmp_path / "v3.pt"
    torch.save(
        {
            "version": CKPT_VERSION,
            "d": 8,
            "agg_level": 2,
            "state_dict": build_model(8).state_dict(),
        },
        path,
    )

    first = ChunkedGNNPredictor(
        path, 0.0, 1.0, levels=2, anchor_stride=4, anchor_block=1
    )
    second = ChunkedGNNPredictor(
        path, 0.0, 2.0, levels=2, anchor_stride=4, anchor_block=1
    )

    assert first.model is second.model


def test_finalize_ctx_reuse_equivalence():
    torch.manual_seed(0)
    model = build_model(d=8).eval()
    geoms, _ = build_stage_geoms((16, 16), 2, 4, 1, 16, torch)
    gp, gh = geoms[:2]
    E = torch.zeros(2, 16 * 16, gp.ndim, model.d)
    vals = torch.rand(2, gp.M)

    with torch.no_grad():
        finalize_ctx = model.embed(E, gp)
        implicit, E_implicit, head_implicit = stage_forward(
            model, E, gp, gh, vals, torch, eb=0.01
        )
        explicit, E_explicit, head_explicit = stage_forward(
            model, E, gp, gh, vals, torch, finalize_ctx=finalize_ctx, eb=0.01
        )

    assert finalize_ctx.shape == (2, gp.M, gp.ndim, model.d)
    assert torch.equal(E_implicit, E_explicit)
    assert torch.equal(head_implicit, head_explicit)
    assert all(torch.equal(a, b) for a, b in zip(implicit, explicit))


def test_dir_state_axis_symmetry():
    """Axes differ *only* through the concatenated direction-state embedding.
    In the closed loop the propagating field is identical across axes (it starts
    at zeros), so with the direction-state table zeroed every axis context stays
    identical; with the learned table the per-axis state breaks the symmetry."""
    torch.manual_seed(0)
    model = build_model(d=8).eval()
    geoms, _ = build_stage_geoms((16, 16), 2, 4, 1, 16, torch)
    gp = geoms[1]  # a stage with valid neighbours
    # Field identical across the axis dim, as it is in the real loop.
    E = torch.randn(2, 16 * 16, 1, model.d).expand(-1, -1, gp.ndim, -1)
    E = E.contiguous()

    with torch.no_grad():
        _, valid = model._line_messages(E, gp)
        assert bool(valid.any())  # guard: the test is meaningful

        model.dir_state.table.zero_()
        ctx = model.embed(E, gp)  # (B, M, ndim, d)
        assert torch.allclose(ctx[..., 0, :], ctx[..., 1, :])

        model.dir_state.table.copy_(build_model(d=8).dir_state.table)
        ctx = model.embed(E, gp)
        assert not torch.allclose(ctx[..., 0, :], ctx[..., 1, :])
