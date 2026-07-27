import numpy as np
import pytest

from deepsz.rans import (
    build_laplace_tables,
    choose_kexp,
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
