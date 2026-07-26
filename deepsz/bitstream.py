"""DeepSZ container format.

File = a fixed little-endian header, the spatial shape, and one zstd frame.

The stage payload (produced by the codec, opaque here) contains, per stage:
  [n_codes u32][entropy blob len u64][entropy blob][n_outliers u32][outliers f32...]

The entropy blob is canonical Huffman unless the header sets ``FLAG_RANS``, in
which case it uses scale-conditioned context coding over the same code array.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import zstandard

MAGIC = b"DEEPSZ01"
VERSION = 1


FLAG_MOCK = 1 << 0
FLAG_GRAY = 1 << 1
FLAG_GNN = 1 << 2
FLAG_INTERP = 1 << 3  # SZ-style interpolation baseline (torch-free)
FLAG_CUBIC = 1 << 4  # interp order: set = cubic, clear = linear
FLAG_NOTILE = 1 << 5  # whole image is one tile (no padding, no seam)
FLAG_RANS = 1 << 6  # per-symbol scale-conditioned coder for stage bins
FLAG_FP16 = 1 << 7  # GNN message pass used fp16 autocast
FLAG_COMPILED = 1 << 8  # GNN message pass used torch.compile
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
            raise ValueError(f"not a DeepSZ stream (bad magic {magic!r})")
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
    if rans_levels is None:
        from .huffman import huffman_encode

        hblob = huffman_encode(blob_codes)
    else:
        from .rans import rans_encode

        if rans_tables is None:
            raise ValueError("rans_tables are required with rans_levels")
        hblob = rans_encode(blob_codes, rans_levels, rans_tables)
    return (
        struct.pack("<IQ", len(blob_codes), len(hblob))
        + hblob
        + struct.pack("<I", len(out))
        + out.tobytes()
    )


def unpack_stage(
    buf: bytes,
    off: int,
    *,
    rans_levels: np.ndarray | None = None,
    rans_tables=None,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_codes, hlen = struct.unpack_from("<IQ", buf, off)
    off += 12
    if rans_levels is None:
        from .huffman import huffman_decode

        codes = huffman_decode(buf[off : off + hlen])
    else:
        from .rans import rans_decode

        if rans_tables is None:
            raise ValueError("rans_tables are required with rans_levels")
        codes = rans_decode(buf[off : off + hlen], rans_levels, rans_tables)
    if len(codes) != n_codes:
        raise ValueError("stage code count mismatch")
    off += hlen
    (n_out,) = struct.unpack_from("<I", buf, off)
    off += 4
    outliers = np.frombuffer(buf, np.float32, count=n_out, offset=off).copy()
    off += 4 * n_out
    return codes, outliers, off
