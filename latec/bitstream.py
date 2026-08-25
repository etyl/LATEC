"""LATEC container format.

File = a fixed little-endian header, the spatial shape, and one zstd frame.

The stage payload (produced by the codec, opaque here) contains, per stage:
  [n_codes u32][shape u5 | kexp u4 | blob len u55][entropy blob]
  [n_outliers u32][outliers f32...]

The entropy blob is canonical Huffman unless the header sets ``FLAG_RANS``, in
which case it uses scale-conditioned context coding over the same code array.
``kexp`` is the rANS coder window for that stage and ``shape`` its entry in the
mixture dictionary (both 0 on the Huffman path). They share the length word so
that per-stage alphabet sizing and shape selection cost no bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import zstandard

MAGIC = b"LATEC001"
VERSION = 3


FLAG_MOCK = 1 << 0
FLAG_GRAY = 1 << 1
FLAG_INTERP = 1 << 3  # SZ-style interpolation baseline (torch-free)
FLAG_CUBIC = 1 << 4  # interp order: set = cubic, clear = linear
FLAG_NOTILE = 1 << 5  # whole image is one tile (no padding, no seam)
FLAG_RANS = 1 << 6  # per-symbol scale-conditioned coder for stage bins
# interp line-end handling, predictor.END_QUAD / END_EXTRAP shifted into flags
FLAG_END_QUAD = 1 << 9
FLAG_END_EXTRAP = 1 << 10
END_MODE_SHIFT = 9  # flags >> this, masked to 2 bits, is the predictor end_mode

# magic, version, flags, channels, dtype, scheduling parameters, predictor
# parameters, value range, checkpoint fingerprint, and per-level EB ratio.
# Spatial dimensions follow this fixed portion as ``[ndim u8][ndim * u32]``.
_HEADER_FMT = "<8sHHIIBBBBIHhddd16sd"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

DTYPE_CODES = {0: np.uint8, 1: np.float32}
DTYPE_IDS = {np.dtype(np.uint8): 0, np.dtype(np.float32): 1}


@dataclass
class Header:
    channels: int
    src_dtype: int  # key into DTYPE_CODES
    spatial: tuple[int, ...]
    eb: float  # absolute error bound, original data units
    levels: int
    anchor_stride: int
    anchor_block: int
    radius: int
    max_radius: int
    agg_level: int  # -1 means the full GNN neighbourhood
    vmin: float
    vmax: float
    ckpt_hash: bytes = b"\0" * 16  # sha256 prefix; zeros for interpolation
    flags: int = 0
    interp_center: int = 0  # interp multi-axis mode: 0=avg both, 1=axis0, 2=axis1
    eb_ratio: float = 1.0  # per-level error-bound decay (coarse tighter); 1=flat
    version: int = VERSION

    def pack(self) -> bytes:
        fixed = struct.pack(
            _HEADER_FMT,
            MAGIC,
            self.version,
            self.flags,
            self.channels,
            self.src_dtype,
            self.levels,
            self.anchor_stride,
            self.anchor_block,
            self.interp_center,
            self.radius,
            self.max_radius,
            self.agg_level,
            self.eb,
            self.vmin,
            self.vmax,
            self.ckpt_hash,
            self.eb_ratio,
        )
        if not self.spatial or len(self.spatial) > 255:
            raise ValueError("spatial shape must contain 1..255 dimensions")
        return fixed + struct.pack(
            f"<B{len(self.spatial)}I", len(self.spatial), *self.spatial
        )

    @classmethod
    def unpack(cls, buf: bytes) -> "Header":
        (
            magic,
            version,
            flags,
            channels,
            src_dtype,
            levels,
            anchor_stride,
            anchor_block,
            interp_center,
            radius,
            max_radius,
            agg_level,
            eb,
            vmin,
            vmax,
            ckpt_hash,
            eb_ratio,
        ) = struct.unpack_from(_HEADER_FMT, buf, 0)
        if magic != MAGIC:
            raise ValueError(f"not a LATEC stream (bad magic {magic!r})")
        if version != VERSION:
            raise ValueError(f"unsupported version {version}")
        (ndim,) = struct.unpack_from("<B", buf, _HEADER_SIZE)
        if not ndim:
            raise ValueError("stream spatial shape is empty")
        spatial = struct.unpack_from(f"<{ndim}I", buf, _HEADER_SIZE + 1)
        return cls(
            channels=channels,
            src_dtype=src_dtype,
            spatial=spatial,
            eb=eb,
            levels=levels,
            anchor_stride=anchor_stride,
            anchor_block=anchor_block,
            radius=radius,
            max_radius=max_radius,
            agg_level=agg_level,
            vmin=vmin,
            vmax=vmax,
            ckpt_hash=ckpt_hash,
            flags=flags,
            interp_center=interp_center,
            eb_ratio=eb_ratio,
            version=version,
        )


def write_stream(header: Header, payload: bytes, zstd_level: int = 9) -> bytes:
    body = zstandard.ZstdCompressor(level=zstd_level).compress(payload)
    return header.pack() + body


def read_stream(data: bytes) -> tuple[Header, bytes]:
    header = Header.unpack(data)
    off = _HEADER_SIZE + 1 + 4 * len(header.spatial)
    payload = zstandard.ZstdDecompressor().decompress(data[off:])
    return header, payload


# ---- stage-record helpers used by codec ----


def pack_stage(
    codes: np.ndarray,
    outliers: np.ndarray,
    *,
    rans_levels: np.ndarray | None = None,
    rans_tables=None,
) -> bytes:
    blob_codes = np.asarray(codes, np.uint32)
    out = np.asarray(outliers, np.float32)
    kexp = shape = 0  # unused on the Huffman path
    if rans_levels is None:
        from .huffman import huffman_encode

        hblob = huffman_encode(blob_codes)
    else:
        from .rans import choose_kexp, choose_shape, rans_encode

        if rans_tables is None:
            raise ValueError("rans_tables are required with rans_levels")
        # Size the alphabet to the window this stage actually uses, then pick the
        # mixture shape that codes it shortest. The decoder needs both to select
        # the same tables, and they ride in the spare high bits of the
        # blob-length field rather than costing bytes of their own: dedicated
        # bytes would swamp the near-empty coarse stages, where a whole schedule
        # level can be a few dozen points spread over hundreds of stages
        # (measured +72% on one such level for the window alone, and +52% for
        # the shape id at 5 bits).
        kexp = choose_kexp(blob_codes, rans_tables.radius)
        shape = choose_shape(blob_codes, rans_levels, rans_tables, kexp)
        hblob = rans_encode(
            blob_codes, rans_levels, rans_tables, kexp=kexp, shape=shape
        )
    return (
        struct.pack("<IQ", len(blob_codes), _pack_hlen(len(hblob), kexp, shape))
        + hblob
        + struct.pack("<I", len(out))
        + out.tobytes()
    )


# The blob length occupies the low bits; the coder window and the shape id share
# the top 9, which the length word had going spare (55 bits still frames 32 PB).
_HLEN_BITS = 55
_KEXP_BITS = 4
_KEXP_SHIFT = _HLEN_BITS
_SHAPE_SHIFT = _HLEN_BITS + _KEXP_BITS


def _pack_hlen(hlen: int, kexp: int, shape: int) -> int:
    if hlen >= 1 << _HLEN_BITS:
        raise ValueError("stage entropy blob is too large to frame")
    if not 0 <= int(kexp) < 1 << _KEXP_BITS:
        raise ValueError("rANS coder window does not fit the stage frame")
    if not 0 <= int(shape) < 1 << 5:
        raise ValueError("rANS shape id does not fit the stage frame")
    return hlen | (int(kexp) << _KEXP_SHIFT) | (int(shape) << _SHAPE_SHIFT)


def unpack_stage(
    buf: bytes,
    off: int,
    *,
    rans_levels: np.ndarray | None = None,
    rans_tables=None,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_codes, framed = struct.unpack_from("<IQ", buf, off)
    hlen = framed & ((1 << _HLEN_BITS) - 1)
    kexp = (framed >> _KEXP_SHIFT) & ((1 << _KEXP_BITS) - 1)
    shape = framed >> _SHAPE_SHIFT
    off += 12
    if rans_levels is None:
        from .huffman import huffman_decode

        codes = huffman_decode(buf[off : off + hlen])
    else:
        from .rans import rans_decode

        if rans_tables is None:
            raise ValueError("rans_tables are required with rans_levels")
        if not kexp:
            raise ValueError("rANS stage record has no coder window")
        codes = rans_decode(
            buf[off : off + hlen], rans_levels, rans_tables, kexp=kexp, shape=shape
        )
    if len(codes) != n_codes:
        raise ValueError("stage code count mismatch")
    off += hlen
    (n_out,) = struct.unpack_from("<I", buf, off)
    off += 4
    outliers = np.frombuffer(buf, np.float32, count=n_out, offset=off).copy()
    off += 4 * n_out
    return codes, outliers, off
