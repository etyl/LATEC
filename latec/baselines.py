"""Reference baselines for the eval command.

SZ3 is used via the official ``pysz`` binding (pip install pysz) when
available, falling back to a locally built ``sz3`` CLI (tools/sz3/bin/sz3).
Note: numpy must be imported before pysz — the pysz extension otherwise pulls
in the old system libstdc++ first, which breaks numpy's C extensions on this
glibc-2.27 machine.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np  # keep before any pysz import (libstdc++ load order)

_pysz_warned = False


def _sz3_pysz(channel: np.ndarray, eb: float) -> tuple[int, np.ndarray] | None:
    global _pysz_warned
    try:
        from pysz import sz, szAlgorithm, szConfig, szErrorBoundMode
    except ImportError as exc:
        if not _pysz_warned:  # ponytail: warn once, not once per channel
            print(
                f"[sz3] pysz unavailable ({exc}); `pip install pysz` or "
                "build tools/sz3/bin/sz3",
                file=sys.stderr,
            )
            _pysz_warned = True
        return None
    config = szConfig()
    config.errorBoundMode = szErrorBoundMode.ABS
    config.absErrorBound = eb
    # Use SZ3's tuned hybrid explicitly instead of relying on szConfig's
    # version-dependent default.  INTERP_LORENZO profiles the field and tunes
    # interpolation order/direction and its error-distribution parameters,
    # selecting the best-rate prediction path at this error bound.
    config.cmprAlgo = szAlgorithm.INTERP_LORENZO
    compressed, _ratio = sz.compress(channel, config)
    rec, _cfg = sz.decompress(compressed, channel.dtype, channel.shape)
    return len(compressed), rec


def find_sz3() -> str | None:
    """Locate the SZ3 command line tool (project-local build first)."""
    local = Path(__file__).resolve().parent.parent / "tools" / "sz3" / "bin" / "sz3"
    if local.exists():
        return str(local)
    for name in ("sz3", "SZ3"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _sz3_cli(channel: np.ndarray, eb: float) -> tuple[int, np.ndarray] | None:
    exe = find_sz3()
    if exe is None:
        return None
    h, w = channel.shape
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw, comp, dec = td / "in.f32", td / "out.sz", td / "out.f32"
        channel.tofile(raw)
        subprocess.run(
            [
                exe,
                "-f",
                "-i",
                str(raw),
                "-z",
                str(comp),
                "-o",
                str(dec),
                "-2",
                str(w),
                str(h),
                "-M",
                "ABS",
                str(eb),
            ],
            check=True,
            capture_output=True,
        )
        return comp.stat().st_size, np.fromfile(dec, np.float32).reshape(h, w)


def sz3_roundtrip(img: np.ndarray, eb: float) -> tuple[int, np.ndarray] | None:
    """Compress/decompress with SZ3 at absolute error bound ``eb``.

    Images are handled per channel as 2D float32 fields (SZ3's native input);
    integer sources are rounded back after decoding. Returns (total compressed
    bytes, reconstruction) or None when no SZ3 implementation is available.
    """
    arr = img if img.ndim == 3 else img[..., None]
    h, w, c = arr.shape
    total = 0
    rec = np.empty((h, w, c), np.float32)
    for k in range(c):
        channel = np.ascontiguousarray(arr[..., k], np.float32)
        result = _sz3_pysz(channel, eb) or _sz3_cli(channel, eb)
        if result is None:
            return None
        n, rec_k = result
        total += n
        rec[..., k] = rec_k

    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        rec = np.clip(np.rint(rec), info.min, info.max)
    rec = rec.astype(img.dtype)
    return total, rec if img.ndim == 3 else rec[..., 0]
