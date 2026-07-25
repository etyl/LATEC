"""Chunked GNN codec: bounded-memory path for large / high-dim tensors.

Covers n-D + integer roundtrips within the error bound, encoder determinism, auto vs
forced chunk selection, chunked-vs-whole equivalence of the guarantee, and the
halo geometry (that out-of-chunk neighbours become live only once their chunk is
coded). The error bound holds regardless of predictor quality — it is the
quantizer's guarantee — so a tiny random checkpoint suffices.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("constriction")  # rANS backend; skip if unavailable

from deepsz import GNNCompressorCodec
import deepsz.gnn_predictor as gp
from deepsz.gnn_codec import _chunk_device_plan
from deepsz.gnn_predictor import (
    CKPT_VERSION,
    ChunkedGNNPredictor,
    _CompactFrame,
    build_chunk_geoms,
    build_model,
    chunk_halo_info,
)
from deepsz.levels import stage_plan

STRIDE = 4
LEVELS = 2


@pytest.fixture()
def current_ckpt(tmp_path):
    torch.manual_seed(0)
    model = build_model(d=8).eval()
    path = tmp_path / "gnn_v6.pt"
    torch.save(
        {
            "d": model.d,
            "agg_level": 2,
            "state_dict": model.state_dict(),
            "version": CKPT_VERSION,
        },
        path,
    )
    return path


def _codec(path, *, eb=1e-2, chunk_size):
    return GNNCompressorCodec(
        path,
        error_bound=eb,
        levels=LEVELS,
        chunk_size=chunk_size,
        fp16=False,
        compile=False,
        gate=False,
    )


def _maxerr(y, x):
    return float(torch.max(torch.abs(y.float() - torch.as_tensor(x).float())))


def test_gate_roundtrip_and_header(current_ckpt):
    """Implicit scale gates roundtrip and use compact variable byte records."""
    from deepsz.gnn_codec import (
        _gate_unpack_t,
        _read_stream,
        _split_gate_payload,
    )

    rng = np.random.RandomState(7)
    gx, gy = np.meshgrid(
        np.linspace(0, 4, 16, dtype=np.float32),
        np.linspace(0, 4, 16, dtype=np.float32),
        indexing="ij",
    )
    f = np.sin(gx) * np.cos(gy) + rng.rand(16, 16).astype(np.float32) * 0.01
    eb = 1e-6
    on = GNNCompressorCodec(
        current_ckpt,
        error_bound=eb,
        levels=LEVELS,
        chunk_size=STRIDE,
        fp16=False,
        compile=False,
        gate=True,
        classical_fallback=False,
    )
    off = _codec(current_ckpt, eb=eb, chunk_size=STRIDE)
    s_on, s_off = on.compress(f), off.compress(f)
    meta, payload = _read_stream(s_on)
    gates, _stage_payload = _split_gate_payload(meta, payload)
    for redundant in (
        "codec",
        "coded_shape",
        "anchor_stride",
        "anchor_block",
        "max_radius",
        "agg_level",
        "entropy_coder",
        "chunk_batch",
    ):
        assert redundant not in meta
    if gates is None:
        assert s_on == s_off
    else:
        assert "gates" not in meta
        assert meta["gate_count"] == len(gates)
        assert len(payload) >= len(gates)
        decoded = []
        i = 0
        while i < len(gates):
            kind, threshold = _gate_unpack_t(
                torch.tensor(gates[i], dtype=torch.int64)
            )
            i += 1
            scale_level = None
            low_side = None
            if int(kind):
                scale_level = gates[i] & 63
                low_side = (gates[i] >> 6) & 1
                i += 1
            decoded.append((int(kind), int(threshold), scale_level, low_side))
        assert all(kind in (0, 1, 2, 3) for kind, _, _, _ in decoded)
        assert all(0 <= threshold < 64 for _, threshold, _, _ in decoded)
        assert all(
            scale_level is None or 0 <= scale_level < 64
            for _, _, scale_level, _ in decoded
        )
        assert all(low_side is None or low_side in (0, 1) for *_, low_side in decoded)
        assert any(kind > 0 for kind, _, _, _ in decoded)
    abs_eb = eb * float(f.max() - f.min())  # eb is relative to the data range
    for s in (s_on, s_off):
        # The codec enforces the bound in normalized coordinates; restoring the
        # original float32 range can add one final-output ULP.
        restore_ulp = float(np.spacing(np.max(np.abs(f))))
        assert _maxerr(off.uncompress(s), f) <= abs_eb + restore_ulp


@pytest.mark.parametrize("winner", [1, 2, 3])
def test_gate_rate_selector_can_choose_each_predictor(winner):
    """The chunk-stage rate proxy selects among multiple classical candidates."""
    from deepsz.gnn_codec import _gate_select_t

    eb = 1e-3
    scale = torch.full((32,), eb)
    gnn_residual = torch.full((1, 32), 10 * eb)
    candidates = [torch.full((1, 32), 5 * eb) for _ in range(3)]
    candidates[winner - 1] = torch.zeros((1, 32))

    kind, threshold, scale_level, low_side = _gate_select_t(
        torch,
        gnn_residual,
        tuple(candidates),
        scale,
        eb,
    )

    assert int(kind) == winner
    assert 0 <= int(threshold) < 64
    assert 0 <= int(scale_level) < 64
    assert int(low_side) in (0, 1)


def test_implicit_gate_switches_only_high_scale_points():
    from deepsz.gnn_codec import _gate_apply_t, _gate_select_t

    eb = 1e-3
    scale = torch.tensor([eb / 16, eb / 16, 4 * eb, 4 * eb]).repeat(128)
    gnn_residual = torch.tensor([[0.0, 0.0, 20 * eb, 20 * eb]]).repeat(1, 128)
    classical = torch.tensor([[20 * eb, 20 * eb, 0.0, 0.0]]).repeat(1, 128)
    poor = torch.full_like(classical, 20 * eb)

    kind, threshold, scale_level, low_side = _gate_select_t(
        torch,
        gnn_residual,
        (classical, poor, poor),
        scale,
        eb,
    )
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0]).repeat(128)
    interp = torch.tensor([[10.0, 20.0, 30.0, 40.0]]).repeat(1, 128)
    selected = _gate_apply_t(
        torch, pred, interp, scale, eb, kind, threshold, low_side
    )

    assert int(kind) == 1
    assert 0 < int(threshold) < 63
    assert 0 <= int(scale_level) < 64
    assert int(low_side) == 0
    expected = torch.tensor([[1.0, 2.0, 30.0, 40.0]]).repeat(1, 128)
    assert torch.equal(selected, expected)


def test_implicit_gate_can_be_ignored_when_savings_do_not_pay_for_parameter():
    from deepsz.gnn_codec import _gate_select_t

    eb = 1e-3
    scale = torch.full((2,), eb)
    residual = torch.tensor([[eb, eb]])

    kind, threshold, scale_level, low_side = _gate_select_t(
        torch,
        residual,
        (torch.zeros_like(residual),) * 3,
        scale,
        eb,
    )

    assert (
        int(kind),
        int(threshold),
        int(scale_level),
        int(low_side),
    ) == (0, 0, 0, 0)


def test_gate_rate_proxy_charges_raw_float_for_outlier():
    from deepsz.gnn_codec import _laplace_bits_t

    eb = 1e-3
    radius = 8
    scale = torch.tensor([eb])
    regular = _laplace_bits_t(torch, torch.tensor([2 * eb * (radius - 1)]), scale, eb, radius)
    outlier = _laplace_bits_t(torch, torch.tensor([2 * eb * radius]), scale, eb, radius)

    assert float(regular[0]) <= 24.0
    assert float(outlier[0]) == 56.0


def test_per_chunk_fallback_can_store_mixed_modes(current_ckpt, monkeypatch):
    """Each independent block keeps its smaller valid codec stream."""
    from deepsz.gnn_codec import (
        _chunk_slices,
        _read_stream,
        _unpack_rate_selected_chunk,
    )

    x = np.zeros((8, 8), np.float64)
    x[:, 4:] = 1.0
    codec = GNNCompressorCodec(
        current_ckpt,
        error_bound=1e-3,
        levels=LEVELS,
        chunk_size=STRIDE,
        fp16=False,
        compile=False,
        gate=True,
        classical_fallback=True,
    )
    original_gnn = codec._compress_gnn_block
    original_interp = codec._compress_interp_block
    losing = np.random.RandomState(0).bytes(1 << 16)
    monkeypatch.setattr(
        codec,
        "_compress_gnn_block",
        lambda block, eb, levels: (
            original_gnn(block, eb, levels) if float(block.mean()) < 0.5 else losing
        ),
    )
    monkeypatch.setattr(
        codec,
        "_compress_interp_block",
        lambda block, eb, levels, stride: (
            losing
            if float(block.mean()) < 0.5
            else original_interp(block, eb, levels, stride)
        ),
    )

    stream = codec.compress(x)
    meta, payload = _read_stream(stream)
    edges = tuple(meta["select_chunks"])
    modes = []
    off = 0
    for _ in _chunk_slices(x.shape, edges):
        mode, _chunk, off = _unpack_rate_selected_chunk(payload, off)
        modes.append(mode)

    assert meta["fallback"] == "rate-selected-chunks"
    assert modes == [1, 0, 1, 0]
    assert "gates" not in meta
    assert _maxerr(codec.uncompress(stream), x) <= codec.error_bound


def test_single_float32_classical_fallback_uses_native_stream(
    current_ckpt, monkeypatch
):
    """A single float32 interpolation chunk needs no outer GNN wrapper."""
    from deepsz.bitstream import MAGIC

    x = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    codec = GNNCompressorCodec(
        current_ckpt,
        error_bound=1e-3,
        levels=LEVELS,
        chunk_size=2 * STRIDE,
        fp16=False,
        compile=False,
        gate=True,
        classical_fallback=True,
    )
    incompressible = np.random.RandomState(1).bytes(1 << 16)
    monkeypatch.setattr(
        codec,
        "_compress_gnn_block",
        lambda *args, **kwargs: incompressible,
    )

    stream = codec.compress(x)

    assert stream.startswith(MAGIC)
    assert _maxerr(codec.uncompress(stream), x) <= codec.error_bound


@pytest.mark.parametrize("payload", [b"", b"\x02" + b"\0" * 8, b"\x00\x01" + b"\0" * 7])
def test_rate_selected_chunk_rejects_malformed_payload(payload):
    from deepsz.gnn_codec import _unpack_rate_selected_chunk

    with pytest.raises(ValueError):
        _unpack_rate_selected_chunk(payload, 0)


# --- roundtrip within the error bound --------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (8, 8),  # 2D, 2x2 chunks
        (12, 8),  # 2D, ragged along axis 0 (12 = 3 chunks; even)
        (8, 8, 8),  # 3D, 2x2x2
        (8, 8, 8, 8),  # 4D
    ],
)
def test_chunked_roundtrip_float(current_ckpt, shape):
    rng = np.random.RandomState(len(shape))
    # smooth-ish field so the predictor has something to do (bound holds anyway)
    x = np.zeros(shape, np.float32)
    for k, s in enumerate(shape):
        wave = np.cos(np.linspace(0, 2 * np.pi, s, dtype=np.float32))
        x = x + wave.reshape([-1 if i == k else 1 for i in range(len(shape))])
    x += rng.rand(*shape).astype(np.float32) * 0.05
    eb = 0.02  # relative to (x.max() - x.min())
    codec = _codec(current_ckpt, eb=eb, chunk_size=STRIDE)

    y = codec.uncompress(codec.compress(x))

    assert tuple(y.shape) == shape
    abs_eb = eb * float(x.max() - x.min())
    assert _maxerr(y, x) <= abs_eb


def test_chunked_roundtrip_integer(current_ckpt):
    rng = np.random.RandomState(7)
    x = (rng.rand(8, 8) * 50).astype(np.int32)
    # eb is relative to (max - min) ~= 49; ~1.0 raw units, tight enough to
    # exercise the integer-rounding check (quantize's round_output=(span, offset)).
    eb = 1.0 / float(x.max() - x.min())
    codec = _codec(current_ckpt, eb=eb, chunk_size=STRIDE)

    y = codec.uncompress(codec.compress(x))

    assert np.issubdtype(np.dtype(y.numpy().dtype), np.integer)
    assert tuple(y.shape) == x.shape
    assert _maxerr(y, x) <= 1.0


# --- determinism ------------------------------------------------------------


@pytest.mark.parametrize("shape", [(8, 8), (8, 8, 8)])
def test_chunked_encoder_deterministic(current_ckpt, shape):
    rng = np.random.RandomState(3)
    x = rng.rand(*shape).astype(np.float32)
    codec = _codec(current_ckpt, chunk_size=STRIDE)

    a = codec.compress(x)
    b = codec.compress(x)

    assert a == b  # byte-identical: closed loop is deterministic incl. coarse table


# --- chunked vs whole -------------------------------------------------------


def test_chunked_matches_whole_bound(current_ckpt):
    """Same tensor both ways: each path honours the bound; a small tensor codes
    identically small under either (sanity that the pipeline, not luck, is wired).
    """
    rng = np.random.RandomState(11)
    x = rng.rand(8, 12).astype(np.float32)

    whole = _codec(current_ckpt, chunk_size=0)  # force whole-tensor
    chunk = _codec(current_ckpt, chunk_size=STRIDE)  # force chunked

    yw = whole.uncompress(whole.compress(x))
    yc = chunk.uncompress(chunk.compress(x))

    assert _maxerr(yw, x) <= whole.error_bound
    assert _maxerr(yc, x) <= chunk.error_bound


def test_gate_applies_on_whole_tensor_path(current_ckpt):
    """gate=True with chunk_size=0 (whole-tensor): the gate is device-only, so a
    gated whole-tensor encode is realised as a single chunk covering the shape
    (grid 1 per axis). The stream then carries chunk metadata and honours the
    bound; the ungated whole-tensor encode stays on the plain numpy path."""
    from deepsz.gnn_codec import _read_stream

    rng = np.random.RandomState(3)
    x = rng.rand(20, 24).astype(np.float32)

    gated = GNNCompressorCodec(
        current_ckpt, error_bound=1e-4, levels=LEVELS, chunk_size=0,
        fp16=False, compile=False, gate=True, classical_fallback=False,
    )
    plain = GNNCompressorCodec(
        current_ckpt, error_bound=1e-4, levels=LEVELS, chunk_size=0,
        fp16=False, compile=False, gate=False,
    )

    sg = gated.compress(x)
    meta_g = _read_stream(sg)[0]
    # gate routes the whole-tensor case through a single chunk covering the shape
    assert meta_g["chunks"] == [20, 24]  # ceil(shape, anchor_stride=1<<LEVELS)
    assert "chunks" not in _read_stream(plain.compress(x))[0]  # numpy whole path
    assert _maxerr(gated.uncompress(sg), x) <= gated.error_bound


def test_auto_chunk_selection(current_ckpt):
    """chunk_size=None: whole-tensor for small inputs, chunked past the
    threshold; forced int must be a multiple of anchor_stride."""
    codec = _codec(current_ckpt, chunk_size=None)
    assert codec._chunk_edges((16, 16), STRIDE) is None  # small -> whole
    big = (1 << 12, 1 << 12)  # 16.7M points -> chunked
    edges = codec._chunk_edges(big, STRIDE)
    assert edges is not None
    assert all(e % STRIDE == 0 and e > 0 for e in edges)
    assert np.prod([min(e, n) for e, n in zip(edges, big)]) <= 1 << 21
    assert np.prod([min(e + STRIDE, n) for e, n in zip(edges, big)]) > 1 << 21

    elongated = codec._chunk_edges((1 << 20, 16), STRIDE)
    assert elongated[1] >= 16
    assert elongated[0] > edges[0]  # short axis leaves room for a longer chunk

    bad = _codec(current_ckpt, chunk_size=STRIDE + 1)  # not a multiple
    with pytest.raises(ValueError):
        bad.compress(np.zeros((8, 8), np.float32))


def test_chunk_device_plan_uses_flat_integer_indices():
    """Stage indices select the same points as the schedule masks, both within
    a contiguous chunk block and within the flattened full reconstruction."""
    cshape = (4, 3)
    full_shape = (8, 7)
    _full, counts, positions, recon_offsets, _, _, _ = _chunk_device_plan(
        torch, "cpu", cshape, full_shape, LEVELS, STRIDE, 1
    )
    plan = stage_plan(cshape, LEVELS, STRIDE, 1)
    origin = (4, 3)
    origin_base = np.ravel_multi_index(origin, full_shape)

    for count, pos, recon_off, (mask, _, _) in zip(
        counts, positions, recon_offsets, plan
    ):
        expected_pos = np.flatnonzero(mask)
        np.testing.assert_array_equal(pos.numpy(), expected_pos)
        assert pos.dtype == torch.int64
        assert count == expected_pos.size

        coords = np.unravel_index(expected_pos, cshape)
        expected_global = np.ravel_multi_index(
            tuple(c + o for c, o in zip(coords, origin)), full_shape
        )
        np.testing.assert_array_equal(recon_off.numpy() + origin_base, expected_global)


def test_query_only_nearest_search_matches_period_tile_lookup():
    rng = np.random.RandomState(17)
    pat = rng.rand(4, 4, 4) > 0.7
    q = np.stack(np.nonzero(rng.rand(4, 4, 4) > 0.4), axis=1)
    res = tuple(q[:, k] for k in range(q.shape[1]))
    direction = (1, -1, 0)

    tiled = gp._nearest_steps_at(pat, direction, 4, res)
    query_only = gp._nearest_steps_at(pat, direction, 4, res, query_only=True)

    np.testing.assert_array_equal(query_only, tiled)


def test_chunk_geometry_uses_query_only_search_and_reports_progress(monkeypatch):
    """Chunk schedules must not rebuild a full period tile for every stage and
    direction.  That path effectively hangs for a 32^4 chunk (76 stages)."""
    gp._CHUNK_GEOM_CACHE.clear()
    seen = []
    original = gp._nearest_steps_at

    def spy(*args, **kwargs):
        seen.append(kwargs.get("query_only", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(gp, "_nearest_steps_at", spy)
    updates = []
    geom = gp.build_chunk_geoms(
        (8, 8), LEVELS, STRIDE, 1, torch, None, 2, updates.append
    )

    assert seen and all(seen)
    assert sum(updates) == len(geom.geoms)

    # A cache hit still completes a caller's setup bar immediately.
    cached_updates = []
    assert (
        gp.build_chunk_geoms(
            (8, 8), LEVELS, STRIDE, 1, torch, None, 2, cached_updates.append
        )
        is geom
    )
    assert sum(cached_updates) == len(geom.geoms)


def test_field_budget_estimate_warns_instead_of_aborting(current_ckpt):
    predictor = ChunkedGNNPredictor(
        current_ckpt, 0.0, 1.0, levels=LEVELS, anchor_stride=STRIDE
    )
    predictor.shape = (8, 8)
    predictor.edges = (8, 8)
    predictor.d = 1 << 30  # force the static estimate beyond the CPU budget

    with pytest.warns(RuntimeWarning, match="estimate is advisory"):
        predictor._check_field_budget(ndim=2, channels=1)


def test_cuda_budget_includes_reusable_allocator_cache():
    class FakeCuda:
        @staticmethod
        def mem_get_info(device):
            return 2_000, 10_000

        @staticmethod
        def memory_reserved(device):
            return 5_000

        @staticmethod
        def memory_allocated(device):
            return 1_000

    class FakeTorch:
        cuda = FakeCuda()

    # 2,000 driver-free + 4,000 reserved-but-unused, with the 80% margin.
    assert gp._cuda_working_budget(FakeTorch(), "cuda") == 4_800


def test_fp16_flag_roundtrips_and_persists(current_ckpt):
    """fp16=True round-trips within the bound and the flag rides in the stream so
    decode replays the same float path. (autocast only bites on cuda; on cpu this
    checks the plumbing + that enabling it doesn't break the closed loop.)"""
    from deepsz.gnn_codec import _read_stream

    rng = np.random.RandomState(9)
    x = rng.rand(8, 8).astype(np.float32)
    codec = GNNCompressorCodec(
        current_ckpt,
        error_bound=0.02,
        levels=LEVELS,
        chunk_size=STRIDE,
        fp16=True,
        compile=False,
        classical_fallback=False,
    )

    stream = codec.compress(x)
    meta, _ = _read_stream(bytes(stream))
    assert meta.get("fp16") is True
    assert _maxerr(codec.uncompress(stream), x) <= 0.02


def test_compile_flag_roundtrips_and_persists(current_ckpt, monkeypatch):
    """compile=True round-trips within the bound and the flag rides in the stream
    so decode replays the same compiled float path. Small workloads skip compile
    (dynamo warmup never amortizes) and record compiled=False."""
    import deepsz.gnn_codec as gc
    from deepsz.gnn_codec import _read_stream

    rng = np.random.RandomState(11)
    x = rng.rand(8, 8).astype(np.float32)
    codec = GNNCompressorCodec(
        current_ckpt,
        error_bound=0.02,
        levels=LEVELS,
        chunk_size=STRIDE,
        fp16=False,
        compile=True,
        classical_fallback=False,
    )

    stream = codec.compress(x)
    meta, _ = _read_stream(bytes(stream))
    assert meta.get("compiled") is False  # 4 chunks: below the gate

    monkeypatch.setattr(gc, "_COMPILE_MIN_CHUNKS", 1)
    stream = codec.compress(x)
    meta, _ = _read_stream(bytes(stream))
    assert meta.get("compiled") is True
    assert _maxerr(codec.uncompress(stream), x) <= 0.02


def test_compile_auto_defers_to_crossover(current_ckpt, monkeypatch):
    """compile='auto' (the default) never compiles while _COMPILE_AUTO_CROSSOVER is
    None (no measured crossover); setting the crossover turns it on past that many
    chunks -- independently of the explicit-compile floor _COMPILE_MIN_CHUNKS."""
    import deepsz.gnn_codec as gc
    from deepsz.gnn_codec import _read_stream

    rng = np.random.RandomState(12)
    x = rng.rand(8, 8).astype(np.float32)  # 4 chunks at chunk_size=STRIDE
    codec = GNNCompressorCodec(
        current_ckpt, error_bound=0.02, levels=LEVELS, chunk_size=STRIDE,
        fp16=False, compile="auto", classical_fallback=False,
    )
    assert codec.auto_compile is True and codec.compile is False

    # crossover None -> auto stays off even with the forced floor lowered
    monkeypatch.setattr(gc, "_COMPILE_MIN_CHUNKS", 1)
    meta, _ = _read_stream(bytes(codec.compress(x)))
    assert meta.get("compiled") is False

    # a low crossover flips auto on (4 chunks >= 2), gating on its own constant
    monkeypatch.setattr(gc, "_COMPILE_AUTO_CROSSOVER", 2)
    meta, _ = _read_stream(bytes(codec.compress(x)))
    assert meta.get("compiled") is True


def test_bad_compile_string_rejected(current_ckpt):
    with pytest.raises(ValueError, match="bool or 'auto'"):
        GNNCompressorCodec(current_ckpt, compile="yes")


# --- halo geometry: out-of-chunk neighbours go live only once coded ---------


def test_halo_links_activate_when_neighbour_coded():
    """Vertical (2,1) chunk grid: chunk 1 (bottom) sees chunk 0 (top) across the
    border. Anchors (level 0) are always usable context; the finer halo cells
    become valid neighbours only after the top chunk is coded. Uses the (2,1)
    orientation because coded-neighbour links into the negative-side halo are
    structurally richer there than in (1,2)."""
    stride, levels = 8, 3
    edges = (16, 16)
    shape = (32, 16)  # two stacked 16x16 chunks
    grid = (2, 1)
    cg = build_chunk_geoms(edges, levels, stride, 1, torch, None)
    origin = (16, 0)  # chunk 1 (bottom)

    def halo_valid_links(coded):
        # compact halo rows are the trailing block (row index > n_interior); a
        # valid line into one is a live cross-border neighbour.
        frame = _CompactFrame(cg, origin, shape, edges, grid, coded, torch, None)
        total = 0
        for s in cg.chain[1:]:  # refinement stages only
            g = frame.geoms[s]
            for ip, v in ((g.ip, g.vp), (g.in_, g.vn)):
                in_halo = ip > frame.n_interior
                total += int((v & in_halo).sum())
        return total

    present_uncoded = chunk_halo_info(
        cg, origin, shape, edges, grid, np.array([False, False])
    )[0]
    present_coded = chunk_halo_info(
        cg, origin, shape, edges, grid, np.array([True, False])
    )[0]

    # more halo cells usable once the neighbour is coded
    assert len(present_coded) > len(present_uncoded)
    # Coding the neighbour is what creates live cross-border links: the periodic
    # nearest step upward from interior points otherwise lands on in-chunk
    # lattice cells, so an uncoded top halo contributes none (the (2,1) negative-
    # side asymmetry). This guards the halo wiring being live, not dead.
    assert halo_valid_links(np.array([True, False])) > halo_valid_links(
        np.array([False, False])
    )
    assert halo_valid_links(np.array([True, False])) > 0


def test_compact_geometry_precomputes_message_selections(monkeypatch):
    """The repeated embed path must consume cached geometry metadata without
    CUDA-style data-dependent selections or distance transforms."""
    stride, levels = 4, 2
    edges = shape = (8, 8)
    grid = (1, 1)
    cg = build_chunk_geoms(edges, levels, stride, 1, torch, None)
    frame = _CompactFrame(
        cg, (0, 0), shape, edges, grid, np.array([False]), torch, None
    )
    stage = cg.chain[1]
    geom = frame.geoms[stage]
    assert cg.geoms[stage].message_blocks is None

    block = geom.message_blocks[0]
    valid = geom.vp | geom.vn
    live = valid.reshape(-1).nonzero(as_tuple=True)[0]
    np.testing.assert_array_equal(block.valid.numpy(), valid.numpy())
    np.testing.assert_array_equal(block.live_idx.numpy(), live.numpy())
    np.testing.assert_array_equal(block.ip.numpy(), geom.ip.reshape(-1)[live].numpy())

    def unexpected(*args, **kwargs):
        raise AssertionError("embed recomputed static geometry metadata")

    monkeypatch.setattr(torch.Tensor, "nonzero", unexpected)
    monkeypatch.setattr(torch, "log2", unexpected)
    model = build_model(d=8).eval()
    field = torch.zeros(1, frame.n_compact, geom.ndim, model.d)
    with torch.no_grad():
        ctx = model.embed(field, geom)
    assert ctx.shape == (1, geom.M, geom.ndim, model.d)


def test_out_of_tensor_halo_never_usable():
    """A corner chunk's halo that falls outside the tensor is never usable,
    regardless of coded flags."""
    stride, levels = 8, 3
    edges = (16, 16)
    shape = (32, 16)
    grid = (2, 1)
    cg = build_chunk_geoms(edges, levels, stride, 1, torch, None)
    # chunk 0 (top): its top halo has global row < 0 -> out of tensor
    present, *_ = chunk_halo_info(
        cg, (0, 0), shape, edges, grid, np.array([True, True])
    )
    gc = cg.ref_halo_coords + np.array([0, 0])
    out = np.any((gc < 0) | (gc >= np.array(shape)), axis=1)
    out_flat = cg.ref_halo_flat[out]
    assert not np.isin(out_flat, present).any()
