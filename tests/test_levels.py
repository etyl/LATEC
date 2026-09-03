import numpy as np
import pytest

from latec.levels import (
    LazyStageMasks,
    mask_tiles,
    n_stages,
    stage_counts,
    stage_masks,
    stage_plan,
)


# all levels == log2(stride) so the schedule densifies to stride 1 (guard)
@pytest.mark.parametrize(
    "shape,levels,stride,block",
    [
        ((64, 64), 3, 8, 1),
        ((64, 64), 3, 8, 4),
        ((512, 512), 3, 8, 4),
        ((512, 512), 4, 16, 8),
        ((64, 48), 2, 4, 1),
        ((33, 65), 4, 16, 2),
        ((16, 16, 16), 3, 8, 1),  # 3-D: schedule is dimension-agnostic
        ((24,), 3, 8, 1),  # 1-D
    ],
)
def test_partition(shape, levels, stride, block):
    masks = stage_masks(shape, levels, stride, block)
    ndim = len(shape)
    # anchor + (2*ndim - 1) sub-stages per level: one per axis (weight 1) plus
    # one fused sub-stage per weight >= 2 (all same-weight axis sets together)
    assert len(masks) == 1 + (2 * ndim - 1) * levels
    total = np.zeros(shape, int)
    for m in masks:
        assert m.shape == shape
        total += m.astype(int)
    assert (total == 1).all()  # disjoint and exhaustive


def test_weight1_stays_separate_weight2_plus_fuses():
    """3-D: the 3 weight-1 (edge-midpoint) sub-stages stay one-per-axis, but
    the 3 weight-2 (face-centre) axis sets are revealed together as one fused
    sub-stage, and the weight-3 (cube-interior) sub-stage is unchanged (it was
    already a single combination)."""
    shape, levels, stride, block = (16, 16, 16), 1, 2, 1
    plan = stage_plan(shape, levels, stride, block)
    # [anchor, axis0, axis1, axis2, fused-weight2, weight3]
    assert len(plan) == 6
    axes = [ax for _, _, ax in plan]
    assert axes == [(), (0,), (1,), (2,), (0, 1, 2), (0, 1, 2)]

    # the fused weight-2 mask equals the union of the 3 individual face masks
    s = plan[4][1]  # finest-level stride (levels=1, so max(stride >> 1, 1))
    coords = [np.arange(n) for n in shape]

    def combo(ax_pair):
        m = np.ones(shape, bool)
        for j, cj in enumerate(coords):
            sel = (
                ((cj % s) == 0) & ((cj % (2 * s)) != 0)
                if j in ax_pair
                else (cj % (2 * s)) == 0
            )
            sh = [1, 1, 1]
            sh[j] = cj.shape[0]
            m &= sel.reshape(sh)
        return m

    union = combo((0, 1)) | combo((0, 2)) | combo((1, 2))
    assert np.array_equal(plan[4][0], union)


def test_anchor_geometry():
    masks = stage_masks((32, 32), 3, 8, 2)  # levels==log2(stride)
    anchor = masks[0]
    assert anchor[0, 0] and anchor[0, 1] and anchor[1, 0] and anchor[1, 1]
    assert not anchor[0, 2] and not anchor[2, 2]
    assert anchor[8, 8] and anchor[9, 9]


def test_deterministic():
    a = stage_masks((64, 64), 3, 8, 4)
    b = stage_masks((64, 64), 3, 8, 4)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_validation():
    with pytest.raises(ValueError):
        stage_masks((64, 64), 0, 8)
    with pytest.raises(ValueError):
        stage_masks((64, 64), 3, 7)
    with pytest.raises(ValueError):
        stage_masks((64, 64), 3, 8, 9)
    with pytest.raises(ValueError):  # levels < log2(stride): can't reach stride 1
        stage_masks((64, 64), 2, 16)


# --- streaming the schedule instead of materializing it ---


SCHEDULES = [
    ((24,), 3, 8, 1),
    ((64, 64), 3, 8, 1),
    ((64, 64), 3, 8, 4),
    ((33, 65), 4, 16, 2),
    ((19, 23, 17), 3, 8, 1),
    ((16, 16, 16, 16), 4, 16, 1),
    ((17, 16, 16, 16, 16), 4, 16, 1),
    ((64, 64), 5, 16, 1),  # levels > log2(stride): the extra levels are empty
]


@pytest.mark.parametrize("shape,levels,stride,block", SCHEDULES)
def test_lazy_masks_match_the_list(shape, levels, stride, block):
    """The streamed masks are the list, in order, with a closed-form length."""
    lazy = LazyStageMasks(shape, levels, stride, block)
    eager = stage_masks(shape, levels, stride, block)
    assert len(lazy) == len(eager) == n_stages(len(shape), levels)
    for got, want in zip(lazy, eager, strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("shape,levels,stride,block", SCHEDULES)
def test_stage_counts_match_the_masks(shape, levels, stride, block):
    """Closed-form stage sizes: the interp predictor keys its geometry on the
    running total, and must not pay a pass over the field's masks to get it."""
    masks = stage_masks(shape, levels, stride, block)
    assert stage_counts(shape, levels, stride, block) == [
        int(m.sum()) for m in masks
    ]


def test_lazy_masks_reject_bad_parameters_eagerly():
    with pytest.raises(ValueError):
        LazyStageMasks((64, 64), 3, 7, 1)  # stride not a power of two
    with pytest.raises(ValueError):
        stage_counts((64, 64), 2, 16, 1)  # levels too small to reach stride 1


@pytest.mark.parametrize("shape", [(64, 64), (9, 7, 5)])
def test_mask_tiles_reproduce_the_boolean_index(shape):
    """Tiled flat indices must visit exactly ``recon[:, pos]`` order, so a
    gather/scatter through them is indistinguishable from boolean indexing."""
    rng = np.random.RandomState(0)
    pos = rng.rand(*shape) < 0.3
    tiles = list(mask_tiles(pos, shape, tile=7))
    assert max(len(t) for t in tiles) <= 7
    np.testing.assert_array_equal(np.concatenate(tiles), np.flatnonzero(pos))
    field = rng.rand(*shape)
    np.testing.assert_array_equal(
        field.reshape(-1)[np.concatenate(tiles)], field[pos]
    )


def test_mask_tiles_without_a_mask_covers_the_grid():
    tiles = list(mask_tiles(None, (4, 5), tile=3))
    np.testing.assert_array_equal(np.concatenate(tiles), np.arange(20))
