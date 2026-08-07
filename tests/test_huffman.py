import numpy as np
import pytest

from latec.huffman import huffman_decode, huffman_encode


@pytest.mark.parametrize(
    "name,symbols",
    [
        ("uniform", np.random.RandomState(0).randint(0, 256, 5000)),
        ("skewed", np.random.RandomState(1).geometric(0.3, 5000) - 1),
        ("single", np.full(1000, 42)),
        ("empty", np.zeros(0, np.uint32)),
        ("two", np.array([7, 7, 7, 9])),
        ("big_alphabet", np.random.RandomState(2).randint(0, 65536, 20000)),
    ],
)
def test_roundtrip(name, symbols):
    symbols = symbols.astype(np.uint32)
    blob = huffman_encode(symbols)
    out = huffman_decode(blob)
    assert np.array_equal(out, symbols), name


def test_size_near_entropy():
    rng = np.random.RandomState(3)
    symbols = (rng.geometric(0.05, 200000) - 1).astype(np.uint32)
    freqs = np.bincount(symbols)
    p = freqs[freqs > 0] / len(symbols)
    entropy_bits = -(p * np.log2(p)).sum() * len(symbols)
    blob = huffman_encode(symbols)
    # header/table overhead + Huffman's <1 bit/symbol slack
    assert len(blob) * 8 < entropy_bits * 1.05 + 8 * (20 + len(freqs))


def test_explicit_alphabet_size():
    symbols = np.array([1, 2, 3], np.uint32)
    blob = huffman_encode(symbols, alphabet_size=1 << 16)
    assert np.array_equal(huffman_decode(blob), symbols)
