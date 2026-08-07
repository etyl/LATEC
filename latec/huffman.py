"""Canonical Huffman coding of non-negative integer symbols (classic SZ stage 3).

Blob layout (little-endian):
  [n_symbols u64][alphabet_size u32][code lengths: alphabet_size x u8]
  [n_bits u64][packed bits, MSB-first]

Only code lengths are serialized; both sides rebuild identical canonical codes
(symbols sorted by (length, symbol)). The lengths array is mostly zeros and the
whole stream passes through zstd afterwards, so no further table compression.
"""

from __future__ import annotations

import heapq
import struct

import numpy as np

MAX_CODE_LEN = 32


def code_lengths(freqs: np.ndarray) -> np.ndarray:
    """Huffman code length per symbol (0 for absent symbols)."""
    present = np.flatnonzero(freqs)
    lengths = np.zeros(len(freqs), np.uint8)
    if len(present) == 0:
        return lengths
    if len(present) == 1:
        lengths[present[0]] = 1
        return lengths
    # heap items: (freq, tiebreak, list of (symbol, depth))
    heap = [(int(freqs[s]), int(s), [(int(s), 0)]) for s in present]
    heapq.heapify(heap)
    tie = len(freqs)
    while len(heap) > 1:
        fa, _, a = heapq.heappop(heap)
        fb, _, b = heapq.heappop(heap)
        merged = [(s, d + 1) for s, d in a] + [(s, d + 1) for s, d in b]
        heapq.heappush(heap, (fa + fb, tie, merged))
        tie += 1
    for s, d in heap[0][2]:
        lengths[s] = d
    if lengths.max() > MAX_CODE_LEN:
        raise ValueError(
            f"Huffman code length {lengths.max()} exceeds {MAX_CODE_LEN}; "
            "input distribution too skewed for this coder"
        )
    return lengths


def canonical_codes(lengths: np.ndarray) -> np.ndarray:
    """Canonical code value per symbol (uint64; valid only where length > 0)."""
    codes = np.zeros(len(lengths), np.uint64)
    present = np.flatnonzero(lengths)
    if len(present) == 0:
        return codes
    order = present[np.lexsort((present, lengths[present]))]
    code = 0
    prev_len = int(lengths[order[0]])
    for s in order:
        ln = int(lengths[s])
        code <<= ln - prev_len
        codes[s] = code
        code += 1
        prev_len = ln
    return codes


def huffman_encode(symbols: np.ndarray, alphabet_size: int | None = None) -> bytes:
    symbols = np.asarray(symbols, dtype=np.uint32).ravel()
    if alphabet_size is None:
        alphabet_size = int(symbols.max()) + 1 if len(symbols) else 1
    if len(symbols) == 0:
        lengths = np.zeros(alphabet_size, np.uint8)
        return _pack_blob(0, lengths, 0, b"")

    freqs = np.bincount(symbols, minlength=alphabet_size)
    lengths = code_lengths(freqs)
    codes = canonical_codes(lengths)

    sym_lens = lengths[symbols].astype(np.int64)
    sym_codes = codes[symbols]
    n_bits = int(sym_lens.sum())

    # vectorized bit expansion: for each occurrence, emit its bits MSB-first
    offsets = np.cumsum(sym_lens) - sym_lens
    rep = np.repeat(np.arange(len(symbols)), sym_lens)
    intra = np.arange(n_bits) - np.repeat(offsets, sym_lens)
    shifts = (sym_lens[rep] - 1 - intra).astype(np.uint64)
    bits = ((sym_codes[rep] >> shifts) & np.uint64(1)).astype(np.uint8)
    packed = np.packbits(bits).tobytes()
    return _pack_blob(len(symbols), lengths, n_bits, packed)


def huffman_decode(blob: bytes) -> np.ndarray:
    n_symbols, lengths, n_bits, packed = _unpack_blob(blob)
    if n_symbols == 0:
        return np.zeros(0, np.uint32)

    present = np.flatnonzero(lengths)
    if len(present) == 1:
        return np.full(n_symbols, present[0], np.uint32)

    codes = canonical_codes(lengths)
    order = present[np.lexsort((present, lengths[present]))]
    # canonical decode tables per length: first code value and first symbol index
    max_len = int(lengths[present].max())
    first_code = np.zeros(max_len + 1, np.int64)
    first_idx = np.zeros(max_len + 1, np.int64)
    count = np.bincount(lengths[present].astype(np.int64), minlength=max_len + 1)
    idx = 0
    for ln in range(1, max_len + 1):
        if count[ln]:
            first_code[ln] = int(codes[order[idx]])
            first_idx[ln] = idx
            idx += count[ln]
        else:
            first_code[ln] = -1

    bits = np.unpackbits(np.frombuffer(packed, np.uint8), count=n_bits)
    out = np.empty(n_symbols, np.uint32)
    pos = 0
    code = 0
    ln = 0
    for i in range(n_symbols):
        code = 0
        ln = 0
        while True:
            code = (code << 1) | int(bits[pos])
            pos += 1
            ln += 1
            fc = first_code[ln] if ln <= max_len else -1
            if fc >= 0 and code - fc < count[ln]:
                out[i] = order[first_idx[ln] + code - fc]
                break
            if ln > max_len:
                raise ValueError("corrupt Huffman stream")
    return out


def _pack_blob(
    n_symbols: int, lengths: np.ndarray, n_bits: int, packed: bytes
) -> bytes:
    head = struct.pack("<QI", n_symbols, len(lengths))
    return (
        head + lengths.astype(np.uint8).tobytes() + struct.pack("<Q", n_bits) + packed
    )


def _unpack_blob(blob: bytes):
    n_symbols, alphabet = struct.unpack_from("<QI", blob, 0)
    off = 12
    lengths = np.frombuffer(blob, np.uint8, count=alphabet, offset=off)
    off += alphabet
    (n_bits,) = struct.unpack_from("<Q", blob, off)
    off += 8
    return n_symbols, lengths, n_bits, blob[off:]
