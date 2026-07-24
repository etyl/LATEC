"""Public GNN tensor codec API tests."""

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


def _codec(path, eb=1e-3):
    return GNNCompressorCodec(
        path,
        error_bound=eb,
        levels=2,
        chunk_size=0,
        fp16=False,
        compile=False,
    )


def test_defaults_match_eval_tensor(tiny_checkpoint):
    codec = GNNCompressorCodec(tiny_checkpoint)

    assert codec.error_bound == 0.01
    # levels default is now "auto": resolved per input shape at compress time,
    # so the fixed levels/anchor_stride are unset until then.
    assert codec.auto_levels is True
    assert codec.levels is None
    assert codec.anchor_stride is None
    assert not hasattr(codec, "max_radius")
    assert not hasattr(codec, "anchor_block")
    assert not hasattr(codec, "agg_level")
    assert codec.chunk_size is None
    assert not hasattr(codec, "chunk_batch")
    assert codec.fp16 is True
    # compile default is now "auto": a benchmark-backed decision (no measured
    # crossover -> off), so the resolved flag is False and auto_compile is set.
    assert codec.auto_compile is True
    assert codec.compile is False


@pytest.mark.parametrize("levels", [1, 2, 4, 6])
def test_anchor_stride_is_derived_from_levels(tiny_checkpoint, levels):
    codec = GNNCompressorCodec(tiny_checkpoint, levels=levels)

    assert codec.auto_levels is False
    assert codec.anchor_stride == 1 << levels


def test_levels_must_be_positive(tiny_checkpoint):
    with pytest.raises(ValueError, match="levels must be >= 1"):
        GNNCompressorCodec(tiny_checkpoint, levels=0)


def test_bad_levels_string_rejected(tiny_checkpoint):
    with pytest.raises(ValueError, match="positive int or 'auto'"):
        GNNCompressorCodec(tiny_checkpoint, levels="deep")


@pytest.mark.parametrize(
    "shape,expected",
    [
        # rank sets the depth; the tuned common ranks are 2-D 9, 3-D 7, 4-D 5.
        ((64, 64, 64, 64), 5),  # 4-D -> level 5 (edge-32 operating point)
        ((128, 128, 128, 128), 5),
        ((256, 256, 256), 7),  # 3-D -> 7 when the size guard does not bind
        ((2048, 2048), 9),  # 2-D -> 9 when the size guard does not bind
        ((4, 4, 4, 100000), 5),  # rank 4 regardless of one huge axis
        ((256, 256, 256, 256, 256), 4),  # 5-D fallback -> smaller stride
        # size guard: small / low-rank inputs cannot exceed anchor_stride=max/2
        ((512, 512), 8),  # 2-D wants 9 but the 512 axis caps it
        ((64, 64), 5),  # 2-D wants 9 but the 64 axis caps it
        ((32, 32), 4),
        ((16, 16, 16, 16), 3),  # small 4-D block: guard drops 5 -> 3
        ((4,), 1),
        ((1,), 1),  # degenerate axis still yields a valid (>=1) schedule
    ],
)
def test_auto_levels_from_shape(shape, expected):
    from deepsz.gnn_codec import _auto_levels

    lv = _auto_levels(shape)
    assert lv == expected
    assert lv >= 1
    assert (1 << lv) >= 2


def test_auto_levels_is_rank_driven_not_size_driven():
    """Above the size guard, levels depends only on rank, not extent."""
    from deepsz.gnn_codec import _auto_levels

    assert _auto_levels((64, 64, 64, 64)) == _auto_levels((256, 256, 256, 256))
    # higher rank -> shallower schedule (smaller stride keeps a chunk bounded)
    assert _auto_levels((256,) * 5) < _auto_levels((256,) * 4)
    assert _auto_levels((256,) * 4) < _auto_levels((256,) * 3)


def test_per_step_eb_ratio_is_depth_normalised():
    """The coarsest level lands on eb * coarse_factor for any schedule depth."""
    from deepsz.gnn_codec import _per_step_eb_ratio

    for coarse in (0.41, 0.65, 0.25):
        for levels in (2, 5, 7, 9):
            per_step = _per_step_eb_ratio(coarse, levels)
            # stage_ebs applies per_step ** depth, depth in [0, levels-1]
            assert per_step ** (levels - 1) == pytest.approx(coarse)
    # levels == 1 has a single finest level -> the factor is a no-op
    assert _per_step_eb_ratio(0.41, 1) == 1.0


def test_default_coarse_factor_preserves_4d_operating_point():
    """Default 0.41 reproduces the legacy per-step 0.8 at the 4-D levels=5 point."""
    from deepsz.gnn_codec import _GNN_EB_COARSE_FACTOR, _per_step_eb_ratio

    assert _per_step_eb_ratio(_GNN_EB_COARSE_FACTOR, 5) == pytest.approx(0.8, abs=2e-3)


def test_numpy_nd_tensor_roundtrip(tiny_checkpoint):
    from deepsz.gnn_codec import _read_stream

    rng = np.random.RandomState(0)
    x = rng.rand(5, 6, 4).astype(np.float32)
    codec = _codec(tiny_checkpoint, eb=0.01)

    stream = codec.compress(x)
    y = codec.uncompress(stream)

    meta = _read_stream(stream)[0]
    assert meta["shape"] == list(x.shape)
    assert meta["dtype"] == x.dtype.str
    for redundant in (
        "codec",
        "coded_shape",
        "anchor_stride",
        "anchor_block",
        "max_radius",
        "agg_level",
        "entropy_coder",
        "chunk_batch",
    ):
        assert redundant not in meta
    assert isinstance(stream, bytes)
    assert isinstance(y, torch.Tensor)
    assert tuple(y.shape) == x.shape
    assert y.dtype == torch.float32
    assert torch.max(torch.abs(y - torch.from_numpy(x))) <= 0.01


def test_torch_tensor_input_roundtrip(tiny_checkpoint):
    x = torch.linspace(-1.0, 1.0, 35, dtype=torch.float32).reshape(7, 5)
    codec = _codec(tiny_checkpoint, eb=0.005)

    y = codec.uncompress(codec.compress(x))

    assert tuple(y.shape) == tuple(x.shape)
    assert y.dtype == x.dtype
    assert torch.max(torch.abs(y - x)) <= 0.005


def test_scalar_shape_roundtrip(tiny_checkpoint):
    x = np.asarray(3.25, dtype=np.float32)
    codec = _codec(tiny_checkpoint, eb=0.001)

    y = codec.uncompress(codec.compress(x))

    assert tuple(y.shape) == ()
    assert y.dtype == torch.float32
    assert torch.abs(y - torch.tensor(3.25)) <= 0.001
