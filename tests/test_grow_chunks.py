"""Grow chunk schedule (``GNNCompressorCodec(grow=True)``): each chunk owns the
shared boundary hyperplane on its internal high faces (extended +1 block) and the
right/bottom neighbour inherits it as an already-decoded low face (column-split).

Correctness is the load-bearing property: the extended block + column-split must
still code every off-grid cell exactly once (bound holds) and the decoder must
reproduce the encoder's reconstruction bit for bit (raster order, shared
column-split). Uses the random tiny checkpoint on CPU, in the fast suite. The gate
composes with grow, so both are exercised."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from deepsz import GNNCompressorCodec
from deepsz.gnn_predictor import CKPT_VERSION, build_model


@pytest.fixture()
def tiny_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = build_model(d=8).eval()
    path = tmp_path / "gnn.pt"
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


def _codec(path, gate, eb=1e-3, chunk_size=8):
    return GNNCompressorCodec(
        path,
        error_bound=eb,
        levels=2,  # anchor_stride 4; chunk_size 8 -> multi-chunk grid
        chunk_size=chunk_size,
        gate=gate,
        grow=True,
        fp16=False,
        compile=False,
        device="cpu",
    )


def _maxerr(a, r):
    a = a.astype(np.float64)
    r = np.asarray(r).reshape(a.shape).astype(np.float64)
    return float(np.abs(a - r).max())


@pytest.mark.parametrize("shape", [(24, 24), (16, 16, 16)])
@pytest.mark.parametrize("gate", [False, True])
def test_grow_roundtrip_holds_bound(tiny_checkpoint, shape, gate):
    rng = np.random.default_rng(0)
    coords = np.meshgrid(*[np.linspace(0, 1, n) for n in shape], indexing="ij")
    field = np.tanh(20 * (coords[0] - 0.5)).astype(np.float32)  # sharp front at seams
    field += 0.05 * rng.standard_normal(shape).astype(np.float32)
    eb = 1e-3
    codec = _codec(tiny_checkpoint, gate=gate, eb=eb)
    rec = codec.uncompress(codec.compress(field)).numpy()
    span = float(field.max() - field.min())
    assert _maxerr(field, rec) <= eb * span + 1e-6


def test_grow_stream_marks_schedule(tiny_checkpoint):
    field = np.linspace(0, 1, 24 * 24, dtype=np.float32).reshape(24, 24)
    meta, _ = _read_meta(_codec(tiny_checkpoint, gate=False).compress(field))
    assert meta.get("grow") is True
    plain, _ = _read_meta(_plain(tiny_checkpoint).compress(field))
    assert "grow" not in plain


def test_grow_decoder_matches_encoder_recon(tiny_checkpoint):
    rng = np.random.default_rng(1)
    field = rng.standard_normal((16, 16, 16)).astype(np.float32)
    codec = _codec(tiny_checkpoint, gate=True)
    rec1 = codec.uncompress(codec.compress(field)).numpy()
    rec2 = codec.uncompress(codec.compress(field)).numpy()
    assert np.array_equal(rec1, rec2)


def test_grow_and_edge_sched_mutually_exclusive(tiny_checkpoint):
    with pytest.raises(ValueError):
        GNNCompressorCodec(
            tiny_checkpoint, error_bound=1e-3, levels=2, chunk_size=8,
            grow=True, edge_sched=True, device="cpu",
        )


def _plain(path):
    return GNNCompressorCodec(
        path, error_bound=1e-3, levels=2, chunk_size=8, gate=False,
        fp16=False, compile=False, device="cpu",
    )


def _read_meta(stream):
    from deepsz.gnn_codec import _read_stream

    return _read_stream(bytes(stream))
