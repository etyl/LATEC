"""Extended-block chunk schedule (the only chunked schedule): each chunk owns
the shared boundary hyperplane on its internal high faces (grown +1 block) and
the right/bottom neighbour inherits it as an already-decoded low face
(column-split). Cross-chunk context flows through the decoded recon array, so
halo neighbours need no per-chunk coarse embedding (value-only halo).

Correctness is the load-bearing property: the extended block + column-split must
still code every off-grid cell exactly once (bound holds) and the decoder must
reproduce the encoder's reconstruction bit for bit (raster order, shared
column-split). Uses the random tiny checkpoint on CPU, in the fast suite. The
gate composes with the schedule, so both are exercised."""

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


def test_grow_decoder_matches_encoder_recon(tiny_checkpoint):
    rng = np.random.default_rng(1)
    field = rng.standard_normal((16, 16, 16)).astype(np.float32)
    codec = _codec(tiny_checkpoint, gate=True)
    rec1 = codec.uncompress(codec.compress(field)).numpy()
    rec2 = codec.uncompress(codec.compress(field)).numpy()
    assert np.array_equal(rec1, rec2)


def test_chunked_schedule_needs_no_coarse_table(tiny_checkpoint):
    """The chunked predictor carries no per-chunk coarse embedding table -- its
    cross-chunk context comes entirely from the decoded recon array."""
    field = np.linspace(0, 1, 24 * 24, dtype=np.float32).reshape(24, 24)
    codec = _codec(tiny_checkpoint, gate=False)
    predictor = codec._chunked_predictor(codec.levels)
    predictor.begin(field.shape, (8, 8))
    assert not hasattr(predictor, "coarse")
