"""LATEC container format.

File = a fixed little-endian header, the spatial shape, and one zstd frame.

The stage payload (produced by the codec, opaque here) contains, per stage:
  [n_codes u32][lit u1 | owx u3 | level u6 | shape u5 | kexp u4 | len u45][blob]
  [n_outliers u32][outliers f32...]

The entropy blob is always rANS over the same Laplace-mixture dictionary; the
header's ``FLAG_RANS`` selects how the scale level is obtained. With the flag
the predictor supplies a per-point scale and the decoder re-derives the levels
itself (``level`` is 0 and unused); without it the encoder fits one level to the
stage's own histogram and sends it here, together with ``owx``, the weight its
outlier marker is coded at -- unless ``lit`` is set, which replaces the whole
stage with its narrowed symbols deflated by zstd, for stages whose codes are
correlated enough that a memoryless coder loses to an LZ pass. ``kexp`` is the
coder window for that stage and ``shape`` its entry in the mixture dictionary.
They all share the length word, so per-stage alphabet sizing, shape selection,
the fitted model and the back-end choice cost no bytes of their own.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import zstandard

from .quantizer import DEFAULT_RADIUS
from .rans import OUTLIER_BITS

MAGIC = b"LATEC001"
VERSION = 4  # stage frame: rANS everywhere, fitted level + outlier weight


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

# Stages smaller than this keep the rANS coder unconditionally: a deflated
# literal cannot pay its own frame back at a few hundred symbols, and the scout
# pass would cost more than the stage.
_LITERAL_MIN = 1 << 12
# The scout deflate runs at level 1; measured stages land ~15% under it at level
# 9, so this is how much cheaper the scout has to be before the real pass runs.
_LITERAL_SCOUT_SLACK = 0.85


def pack_stage(
    codes: np.ndarray,
    outliers: np.ndarray,
    *,
    rans_levels: np.ndarray | None = None,
    rans_tables=None,
    radius: int = DEFAULT_RADIUS,
    zstd_level: int = 9,
) -> bytes:
    blob_codes = np.asarray(codes, np.uint32)
    out = np.asarray(outliers, np.float32)
    level = owx = 0
    literal = False
    if rans_levels is None:
        # No per-point scale: fit one scale level to this stage's histogram and
        # signal it. Same coder, same dictionary, one bulk call.
        from .rans import (
            choose_const_level,
            choose_kexp,
            const_tables,
            narrow_literal,
            rans_encode_const,
        )

        tables = rans_tables if rans_tables is not None else const_tables(radius)
        kexp = choose_kexp(blob_codes, tables.radius)
        level, shape, owx, model_bits = choose_const_level(blob_codes, tables, kexp)
        hblob = None
        if len(blob_codes) >= _LITERAL_MIN:
            syms = narrow_literal(blob_codes, tables.radius, kexp)
            # A cheap pass first: only when a fast deflate already beats the
            # model does the (much slower) full-strength one get to run.
            scout = zstandard.ZstdCompressor(level=1).compress(syms)
            if len(scout) * _LITERAL_SCOUT_SLACK < model_bits / 8.0:
                blob = zstandard.ZstdCompressor(level=zstd_level).compress(syms)
                if len(blob) * 8.0 < model_bits:
                    hblob, literal = blob, True
            del syms
        if hblob is None:
            hblob = rans_encode_const(
                blob_codes, level, tables, kexp=kexp, shape=shape, owx=owx
            )
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
        struct.pack("<IQ", len(blob_codes), _pack_hlen(len(hblob), kexp, shape, level, owx, literal))
        + hblob
        + struct.pack("<I", len(out))
        + out.tobytes()
    )


# The blob length occupies the low bits; the coder window, the shape id and the
# fitted (level, outlier weight) share the top 18, which the length word had
# going spare (46 bits still frames 64 TB per stage).
_HLEN_BITS = 45
_KEXP_BITS = 4
_SHAPE_BITS = 5
_LEVEL_BITS = 6
_KEXP_SHIFT = _HLEN_BITS
_SHAPE_SHIFT = _KEXP_SHIFT + _KEXP_BITS
_LEVEL_SHIFT = _SHAPE_SHIFT + _SHAPE_BITS
_OWX_SHIFT = _LEVEL_SHIFT + _LEVEL_BITS
_LIT_SHIFT = _OWX_SHIFT + OUTLIER_BITS


def _pack_hlen(
    hlen: int,
    kexp: int,
    shape: int,
    level: int = 0,
    owx: int = 0,
    literal: bool = False,
) -> int:
    if hlen >= 1 << _HLEN_BITS:
        raise ValueError("stage entropy blob is too large to frame")
    if not 0 <= int(kexp) < 1 << _KEXP_BITS:
        raise ValueError("rANS coder window does not fit the stage frame")
    if not 0 <= int(shape) < 1 << _SHAPE_BITS:
        raise ValueError("rANS shape id does not fit the stage frame")
    if not 0 <= int(level) < 1 << _LEVEL_BITS:
        raise ValueError("rANS scale level does not fit the stage frame")
    if not 0 <= int(owx) < 1 << OUTLIER_BITS:
        raise ValueError("rANS outlier weight does not fit the stage frame")
    return (
        hlen
        | (int(kexp) << _KEXP_SHIFT)
        | (int(shape) << _SHAPE_SHIFT)
        | (int(level) << _LEVEL_SHIFT)
        | (int(owx) << _OWX_SHIFT)
        | (int(bool(literal)) << _LIT_SHIFT)
    )


def unpack_stage(
    buf: bytes,
    off: int,
    *,
    rans_levels: np.ndarray | None = None,
    rans_tables=None,
    radius: int = DEFAULT_RADIUS,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_codes, framed = struct.unpack_from("<IQ", buf, off)
    hlen = framed & ((1 << _HLEN_BITS) - 1)
    kexp = (framed >> _KEXP_SHIFT) & ((1 << _KEXP_BITS) - 1)
    shape = (framed >> _SHAPE_SHIFT) & ((1 << _SHAPE_BITS) - 1)
    level = (framed >> _LEVEL_SHIFT) & ((1 << _LEVEL_BITS) - 1)
    owx = (framed >> _OWX_SHIFT) & ((1 << OUTLIER_BITS) - 1)
    literal = bool((framed >> _LIT_SHIFT) & 1)
    off += 12
    if rans_levels is None:
        from .rans import const_tables, rans_decode_const, widen_literal

        tables = rans_tables if rans_tables is not None else const_tables(radius)
        if not kexp:
            raise ValueError("rANS stage record has no coder window")
        if literal:
            codes = widen_literal(
                zstandard.ZstdDecompressor().decompress(buf[off : off + hlen]),
                n_codes,
                tables.radius,
                kexp,
            )
        else:
            codes = rans_decode_const(
                buf[off : off + hlen],
                n_codes,
                level,
                tables,
                kexp=kexp,
                shape=shape,
                owx=owx,
            )
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
