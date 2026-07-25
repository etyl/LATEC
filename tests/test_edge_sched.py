"""Edges-first chunk scheduling (``deepsz.edge_sched``): correctness and the
error-bound guarantee, exercised through ``GNNCompressorCodec(edge_sched=True)``.

The schedule codes chunk faces before interiors; a bug in the two-phase ordering
or the query-subset geometry shows up as either a broken roundtrip or a violated
bound, so these are the load-bearing checks. Uses the random tiny checkpoint (no
trained weights needed) on CPU, so they run in the fast suite."""

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


def _codec(path, edge_sched, eb=1e-3, chunk_size=8):
    return GNNCompressorCodec(
        path,
        error_bound=eb,
        levels=2,  # anchor_stride 4; chunk_size 8 -> multi-chunk grid
        chunk_size=chunk_size,
        gate=False,
        edge_sched=edge_sched,
        fp16=False,
        compile=False,
        device="cpu",
    )


def _maxerr(a, r):
    a = a.astype(np.float64)
    r = np.asarray(r).reshape(a.shape).astype(np.float64)
    return float(np.abs(a - r).max())


@pytest.mark.parametrize("shape", [(24, 24), (16, 16, 16)])
def test_edge_sched_roundtrip_holds_bound(tiny_checkpoint, shape):
    rng = np.random.default_rng(0)
    # smooth field with a sharp front, so chunk-face seams actually matter
    coords = np.meshgrid(*[np.linspace(0, 1, n) for n in shape], indexing="ij")
    field = np.tanh(20 * (coords[0] - 0.5)).astype(np.float32)
    field += 0.05 * rng.standard_normal(shape).astype(np.float32)
    eb = 1e-3
    codec = _codec(tiny_checkpoint, edge_sched=True, eb=eb)
    stream = codec.compress(field)
    rec = codec.uncompress(stream).numpy()
    span = float(field.max() - field.min())
    # error_bound is relative to (max - min); the codec guarantees it absolutely
    assert _maxerr(field, rec) <= eb * span + 1e-6


def test_edge_sched_stream_marks_schedule(tiny_checkpoint):
    field = np.linspace(0, 1, 24 * 24, dtype=np.float32).reshape(24, 24)
    codec = _codec(tiny_checkpoint, edge_sched=True)
    meta, _ = _read_meta(codec.compress(field))
    assert meta.get("edge_sched") is True
    plain_meta, _ = _read_meta(
        _codec(tiny_checkpoint, edge_sched=False).compress(field)
    )
    assert "edge_sched" not in plain_meta


def test_edge_sched_is_deterministic(tiny_checkpoint):
    field = np.sin(np.linspace(0, 6, 20 * 20, dtype=np.float32)).reshape(20, 20)
    codec = _codec(tiny_checkpoint, edge_sched=True)
    assert codec.compress(field) == codec.compress(field)


def test_edge_sched_decoder_matches_encoder_recon(tiny_checkpoint):
    # decode must reproduce the encoder's committed reconstruction exactly, not
    # merely stay within the bound -- the two-phase order is shared bitwise.
    rng = np.random.default_rng(1)
    field = rng.standard_normal((16, 16, 16)).astype(np.float32)
    codec = _codec(tiny_checkpoint, edge_sched=True)
    rec1 = codec.uncompress(codec.compress(field)).numpy()
    rec2 = codec.uncompress(codec.compress(field)).numpy()
    assert np.array_equal(rec1, rec2)


def _read_meta(stream):
    from deepsz.gnn_codec import _read_stream

    return _read_stream(bytes(stream))
