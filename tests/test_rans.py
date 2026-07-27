import numpy as np
import pytest

from deepsz.rans import (
    N_SHAPES,
    SHAPES,
    build_laplace_tables,
    choose_kexp,
    choose_shape,
    full_kexp,
    model_bits,
    rans_decode,
    rans_encode,
    scale_to_level,
)


def _roundtrip(codes, levels, tables):
    blob = rans_encode(codes, levels, tables)
    got = rans_decode(blob, levels, tables)
    assert np.array_equal(got, np.asarray(codes, np.uint32).ravel())
    return blob


def test_random_codes_random_scale_levels_roundtrip():
    rng = np.random.RandomState(0)
    tables = build_laplace_tables(0.01, radius=32, precision=14)
    levels = rng.randint(0, 64, 4096).astype(np.uint8)
    codes = rng.randint(0, 64, len(levels)).astype(np.uint32)

    blob = _roundtrip(codes, levels, tables)

    assert len(blob) * 8 <= model_bits(codes, levels, tables) * 1.02 + 64


def test_skewed_distribution_tracks_model_bits():
    rng = np.random.RandomState(1)
    eb = 0.01
    radius = 64
    tables = build_laplace_tables(eb, radius=radius, precision=15)
    levels = np.full(20000, scale_to_level(np.asarray([eb]), eb)[0], np.uint8)
    residual = rng.laplace(0.0, eb, len(levels))
    q = np.rint(residual / (2 * eb)).astype(np.int64)
    codes = np.clip(q + radius, 1, 2 * radius - 1).astype(np.uint32)

    blob = _roundtrip(codes, levels, tables)

    assert abs(len(blob) * 8 - model_bits(codes, levels, tables)) / len(blob) / 8 < 0.01


def test_empty_stage_roundtrip():
    tables = build_laplace_tables(0.01, radius=8, precision=10)

    blob = _roundtrip(np.zeros(0, np.uint32), np.zeros(0, np.uint8), tables)

    assert blob == b""


def test_alphabet_edge_outlier_code_roundtrip():
    tables = build_laplace_tables(0.01, radius=16, precision=12)
    codes = np.asarray([0, 1, 31, 0, 16, 31], np.uint32)
    levels = np.asarray([0, 4, 12, 63, 32, 1], np.uint8)

    _roundtrip(codes, levels, tables)


def test_rejects_invalid_scale_level():
    tables = build_laplace_tables(0.01, radius=8, n_levels=4, precision=10)
    levels = np.asarray([4], np.uint8)

    with pytest.raises(ValueError, match="scale level"):
        rans_encode(np.asarray([1], np.uint32), levels, tables)
    with pytest.raises(ValueError, match="scale level"):
        rans_decode(b"\0\0\0\0", levels, tables)


def test_narrowed_alphabet_roundtrips_at_every_window():
    radius = 1 << 15
    tables = build_laplace_tables(0.01, radius)
    rng = np.random.RandomState(3)
    for kexp in range(1, full_kexp(radius) + 1):
        span = (1 << kexp) - 1
        # 0 is the outlier marker; both window edges must survive
        codes = np.concatenate(
            [
                [0, radius, radius + span, radius - span],
                rng.randint(radius - span, radius + span + 1, 256),
            ]
        ).astype(np.uint32)
        levels = rng.randint(0, 64, len(codes)).astype(np.uint8)

        assert choose_kexp(codes, radius) == kexp
        blob = rans_encode(codes, levels, tables, kexp=kexp)
        assert np.array_equal(rans_decode(blob, levels, tables, kexp=kexp), codes)


def test_full_window_is_the_default_and_covers_the_radius():
    radius = 1 << 15
    tables = build_laplace_tables(0.01, radius)
    codes = np.asarray([0, 1, radius, 2 * radius - 1], np.uint32)
    levels = np.asarray([0, 17, 33, 63], np.uint8)

    assert choose_kexp(codes, radius) == full_kexp(radius)
    explicit = rans_encode(codes, levels, tables, kexp=full_kexp(radius))
    assert rans_encode(codes, levels, tables) == explicit
    assert np.array_equal(rans_decode(explicit, levels, tables), codes)


def test_sizing_the_alphabet_never_costs_bits():
    """A confident stage pays the per-symbol PMF floor over unused bins."""
    radius = 1 << 15
    tables = build_laplace_tables(0.01, radius)
    codes = np.full(4096, radius, np.uint32)  # every residual quantizes to zero
    levels = np.zeros(len(codes), np.uint8)  # most confident scale level

    sized = choose_kexp(codes, radius)
    assert sized == 1
    narrow = model_bits(codes, levels, tables, kexp=sized)
    wide = model_bits(codes, levels, tables, kexp=full_kexp(radius))
    assert narrow < 0.5 * wide
    assert len(rans_encode(codes, levels, tables, kexp=sized)) < len(
        rans_encode(codes, levels, tables)
    )


def test_rejects_code_outside_the_narrowed_window():
    radius = 1 << 15
    tables = build_laplace_tables(0.01, radius)
    codes = np.asarray([radius + 8], np.uint32)

    with pytest.raises(ValueError, match="narrowed"):
        rans_encode(codes, np.zeros(1, np.uint8), tables, kexp=2)


def test_rejects_unaligned_payload():
    tables = build_laplace_tables(0.01, radius=8, precision=10)

    with pytest.raises(ValueError, match="aligned"):
        rans_decode(b"\0", np.asarray([0], np.uint8), tables)


def test_every_shape_roundtrips_at_every_window():
    """The dictionary is decodable over any alphabet the sizing can produce."""
    rng = np.random.RandomState(3)
    radius = 1 << 8
    tables = build_laplace_tables(0.01, radius=radius, precision=20)
    for kexp in (1, 4, full_kexp(radius)):
        span = (1 << kexp) - 1
        codes = np.concatenate(
            [
                rng.randint(radius - span, radius + span + 1, 512),
                [0, radius - span, radius + span],
            ]
        ).astype(np.uint32)
        levels = rng.randint(0, 64, len(codes)).astype(np.uint8)
        for shape in range(N_SHAPES):
            blob = rans_encode(codes, levels, tables, kexp=kexp, shape=shape)
            got = rans_decode(blob, levels, tables, kexp=kexp, shape=shape)
            assert np.array_equal(got, codes), (kexp, shape)


def test_shape_zero_is_the_pure_laplace():
    """Entry 0 must reproduce the unshaped coder bit for bit."""
    rng = np.random.RandomState(4)
    radius = 1 << 8
    tables = build_laplace_tables(0.01, radius=radius, precision=20)
    codes = rng.randint(radius - 40, radius + 40, 4096).astype(np.uint32)
    levels = rng.randint(0, 64, len(codes)).astype(np.uint8)
    kexp = choose_kexp(codes, radius)
    assert SHAPES[0] == (0, 0.0)
    assert rans_encode(codes, levels, tables, kexp=kexp, shape=0) == rans_encode(
        codes, levels, tables, kexp=kexp
    )


def test_selected_shape_never_costs_more_than_laplace():
    """Selection is over a dictionary containing the Laplace, so it cannot lose.

    Scored on the full stage so the check is exact rather than subject to the
    encoder's sampling stride.
    """
    rng = np.random.RandomState(5)
    radius = 1 << 10
    tables = build_laplace_tables(0.01, radius=radius, precision=22)
    # A scale mixture inside one grid level: most points at the level's own
    # scale, a tenth drawn from a much broader one. Scales come from the grid and
    # are converted to bin-index units (a bin spans 2*eb), which is what the
    # coder's tables are expressed in.
    n = 20000
    eb = tables.eb
    core = tables.scale_grid[20] / (2.0 * eb)
    tail = tables.scale_grid[36] / (2.0 * eb)
    scale = np.where(rng.rand(n) < 0.1, tail, core)
    q = np.rint(rng.laplace(0.0, scale)).astype(np.int64)
    codes = np.clip(q + radius, 1, 2 * radius - 1).astype(np.uint32)
    levels = np.full(n, 20, np.uint8)
    kexp = choose_kexp(codes, radius)

    shape = choose_shape(codes, levels, tables, kexp, sample=1.0)
    assert shape != 0, "a scale mixture should not select the pure Laplace"
    picked = model_bits(codes, levels, tables, kexp=kexp, shape=shape)
    laplace = model_bits(codes, levels, tables, kexp=kexp, shape=0)
    assert picked < laplace


def test_quantize_pmf_rows_matches_the_scalar_path():
    """The batched quantizer is a pure speedup, not a different table."""
    from deepsz.rans import _quantize_pmf, _quantize_pmf_rows

    rng = np.random.RandomState(6)
    total = 1 << 20
    for alphabet in (4, 64, 1024):
        w = rng.rand(64, alphabet) ** 6 + 1e-12
        rows = _quantize_pmf_rows(w, total)
        assert (rows.sum(1) == total).all()
        for i in range(64):
            assert np.array_equal(rows[i], _quantize_pmf(w[i], total))


def test_model_cache_respects_its_byte_budget():
    """Many (window, shape) pairs must not pin unbounded table memory.

    One field can want a dozen shapes at the widest window, which is ~160 MB of
    tables on its own, so the cache is bounded by bytes rather than entries.
    """
    from deepsz import rans

    rans._clear_model_cache()
    budget = rans._MODEL_BUDGET_BYTES
    try:
        rans._MODEL_BUDGET_BYTES = 1 << 20  # 1 MB: a few small windows' worth
        for shape in range(N_SHAPES):
            rans._ans_models(10, shape, 64, 24)
        assert rans._model_bytes <= max(rans._MODEL_BUDGET_BYTES, 64 * 2048 * 4)
        assert len(rans._model_cache) < N_SHAPES, "nothing was evicted"
        # eviction is transparent: a dropped entry rebuilds to the same tables
        again = rans._ans_models(10, 0, 64, 24)
        assert len(again) == 64
    finally:
        rans._MODEL_BUDGET_BYTES = budget
        rans._clear_model_cache()


def test_wide_windows_offer_only_the_cheap_entries():
    """Encoder-side only: the id space and the decoder are unchanged.

    Every entry must still decode at a wide window even though the encoder will
    not select it, so an id restriction can never make a stream unreadable.
    """
    from deepsz.rans import _WIDE_KEXP, _shape_ids

    assert _shape_ids(_WIDE_KEXP - 1) == tuple(range(N_SHAPES))
    wide = _shape_ids(_WIDE_KEXP)
    assert wide[0] == 0 and len(wide) < N_SHAPES
    assert all(SHAPES[i][1] == 0.02 for i in wide[1:])

    radius = 1 << 15
    tables = build_laplace_tables(0.01, radius=radius)
    kexp = _WIDE_KEXP
    span = (1 << kexp) - 1
    codes = np.asarray([0, radius, radius + span, radius - span], np.uint32)
    levels = np.asarray([0, 17, 33, 63], np.uint8)
    excluded = next(i for i in range(N_SHAPES) if i not in _shape_ids(kexp))
    blob = rans_encode(codes, levels, tables, kexp=kexp, shape=excluded)
    got = rans_decode(blob, levels, tables, kexp=kexp, shape=excluded)
    assert np.array_equal(got, codes)
