"""Finetune the GNN predictor on the training split of any The Well dataset.

--dataset is the registry name of the simulation (turbulence_gravity_cooling,
MHD_64, rayleigh_taylor_instability, active_matter, ...); --params is a
substring of the wanted file's name (the parameter combination, e.g. `At_75`
or `Ma_0.7_Ms_0.5`), defaulting to the first file of the split, and repeats to
mix several combinations into the training set. --field takes a scalar name
(density) or a vector component (magnetic_field_x, velocity_z), defaulting to
the first scalar field in the file.

The tensor rank follows the dataset: a trajectory is (T, *spatial), so 2-D
simulations train and eval as 3-D tensors and 3-D ones as 4-D. Training fields
are hypercubes of --crop cells along time and each spatial axis (clamped to the
axis length). --crop and --stride default per rank to the size inference codes
one chunk at — 64/32 for 3-D tensors, 32/16 for 4-D — so a training slab is
about the block the model will be asked to predict.

Eval is a whole trajectory of the dataset's *test* split (--test-traj, T
truncated to a power of two), pushed through the real codec, so the reported
bits/value and PSNR are the deployed numbers on data the training split never
touched. The best-scoring weights are the ones written out.

    python scripts/finetune_well.py --init data/gnn_predictor.pt --steps 500

Tuning is always on the train split and validation on the test split. The first
--params token also selects the test-split file the eval trajectory comes from
(--test-params overrides it, which convective_envelope_rsg needs: it partitions
whole trajectories across the splits, so no one token names a file in both);
further tokens only add training data (each is a separate download).

The eval tensor must be the one the benchmark reads, or a bpp won here will not
show up there. Get the flags from the benchmark itself rather than by hand:

    python check_ft_match.py    # in benchmark-scientific-data-compression

Well files run to tens of GB apiece, so --n-files caps how many are pulled per
--params token and --max-traj N pulls only the first N trajectories of --field
out of each (euler's train file is one 212 GB bag of 400). Download before the
job, on a login node:

    python scripts/finetune_well.py --download-only --max-traj 32 \
        --dataset euler_multi_quadrants_openBC --params gamma_1.4_Dry_air_20 \
        --field density

scripts/well_data.py does the same download without importing torch, which is
what to run on a login node.

ponytail: reuses train_gnn.py's stage loop, EMA, noise sampling and codec eval
instead of re-implementing them; only the data source is new.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_gnn import (  # noqa: E402
    ModelEMA,
    eval_tensor_codec,
    normalize_tensor,
    run_stages,
    sample_noise,
    training_autocast,
)

from well_data import (  # noqa: E402
    _base_path,
    _field_dataset,
    _spatial_dims,
    default_field,
    ensure_file,
)

from latec.gnn_predictor import (  # noqa: E402
    CKPT_VERSION,
    build_model,
    build_stage_geoms,
)

# (crop, anchor stride) per tensor rank: two anchor cells per axis, at the size
# inference codes one chunk at. Ranks outside the table halve both per extra
# axis, keeping the slab near the same point count.
_CROP_STRIDE = {3: (32, 16), 4: (16, 8)}


def _crop_stride(ndim: int) -> tuple[int, int]:
    if ndim in _CROP_STRIDE:
        return _CROP_STRIDE[ndim]
    crop, stride = _CROP_STRIDE[min(max(ndim, 3), 4)]
    shift = max(0, ndim - 4)
    return max(4, crop >> shift), max(2, stride >> shift)


class WellSlabs:
    """Random hypercube crops (time, *spatial) of trajectories, across files.

    The rank follows the dataset — 1 time axis plus however many spatial dims
    the file declares — so a 2-D simulation yields 3-D slabs and a 3-D one 4-D
    slabs. Files stay open and only the crop is read off disk. `--crop` is the
    side along *every* axis, time included, clamped to each axis length; 0 keeps
    each axis whole.
    """

    def __init__(self, paths, field, crop):
        self.files = [h5py.File(p, "r") for p in paths]
        self.field = field
        self.crop = crop
        self.ndim = 1 + len(_spatial_dims(self.files[0]))  # time + spatial
        self.index = []  # (file_idx, trajectory)
        for fi, file in enumerate(self.files):
            if 1 + len(_spatial_dims(file)) != self.ndim:
                raise SystemExit(f"{file.filename}: rank differs from the first file")
            dset, _ = _field_dataset(file, field)
            self.index += [(fi, traj) for traj in range(dset.shape[0])]
        if not self.index:
            raise SystemExit("no training slabs found")

    def _axes(self, dset) -> tuple[int, ...]:
        return tuple(dset.shape[1 : 1 + self.ndim])

    def __len__(self):
        return len(self.index)

    def slab(self, i, rng, log=False) -> np.ndarray:
        """One normalized [0, 1] hypercube; optional log10 first, since density
        and pressure span decades."""
        fi, traj = self.index[i]
        dset, component = _field_dataset(self.files[fi], self.field)
        axes = self._axes(dset)
        sides = [min(self.crop, n) if self.crop else n for n in axes]
        starts = [rng.integers(0, n - side + 1) for n, side in zip(axes, sides)]
        sl = tuple(slice(s, s + side) for s, side in zip(starts, sides))
        vol = np.asarray(dset[(traj,) + sl], dtype=np.float32)
        if component is not None:
            vol = vol[..., component]
        return self._prepare(vol, log)

    def trajectory(self, i, log=False, n_time=0) -> np.ndarray:
        """One whole trajectory (T, *spatial) — the tensor the benchmark hands
        the codec. T is truncated to the largest power of two, or to `n_time`
        for the datasets the benchmark caps instead (euler and convective at 64,
        gray_scott at 256): the two must read the same tensor."""
        fi, traj = self.index[i]
        dset, component = _field_dataset(self.files[fi], self.field)
        available = int(dset.shape[1])
        if n_time and n_time > available:
            raise SystemExit(f"--test-timesteps {n_time} > the {available} in the file")
        n_time = n_time or 1 << (available.bit_length() - 1)
        vol = np.asarray(dset[traj, :n_time], dtype=np.float32)
        if component is not None:
            vol = vol[..., component]
        return self._prepare(vol, log)

    @staticmethod
    def _prepare(vol, log):
        if log:
            vol = np.log10(np.maximum(vol, np.finfo(np.float32).tiny))
        return normalize_tensor(vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="data/gnn_predictor.pt", help="checkpoint to finetune")
    ap.add_argument(
        "--out",
        default=None,
        help="checkpoint name; always written to <parent>/runs/well-<timestamp>/",
    )
    ap.add_argument(
        "--dataset",
        default="turbulence_gravity_cooling",
        help="The Well registry dataset name",
    )
    ap.add_argument("--field", default=None, help="default: the file's first scalar field")
    ap.add_argument("--log", action="store_true", help="log10 the field before normalizing")
    ap.add_argument(
        "--params",
        action="append",
        default=None,
        metavar="TOKEN",
        help="file to use, by name substring, i.e. its parameter combination "
        "(default: the split's first file). Repeatable: the first holds the "
        "eval trajectory, the rest are extra training files (each a separate "
        "multi-GB download)",
    )
    ap.add_argument(
        "--test-traj", type=int, default=0, help="eval trajectory index in the test-split file"
    )
    ap.add_argument(
        "--test-params",
        default=None,
        help="test-split file, by name substring (default: the first --params). "
        "Needed where the splits hold different files, i.e. convective_envelope_rsg, "
        "which partitions whole trajectories across them",
    )
    ap.add_argument(
        "--test-timesteps",
        type=int,
        default=0,
        help="truncate the eval trajectory to this many timesteps (0 = the largest "
        "power of two). Set it to whatever the benchmark's dataset caps n_timesteps "
        "at, or the two eval on different tensors",
    )
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1, help="slabs per optimizer step")
    ap.add_argument(
        "--crop",
        type=int,
        default=None,
        help="side of the random hypercube crop, time axis included "
        "(0 = the whole trajectory; default: the pretraining crop for this rank)",
    )
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500, help="linear lr warmup steps")
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--noise-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    ap.add_argument(
        "--stride", type=int, default=None, help="default: the pretraining anchor stride"
    )
    ap.add_argument("--levels", type=int, default=None)
    ap.add_argument("--max-radius", type=int, default=64)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--eval-eb", type=float, default=0.01)
    ap.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="full-codec roundtrip of the held-out trajectory every N steps "
        "(minutes on a 64^4 tensor — keep it rare)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    ap.add_argument("--wandb-project", default="gnn-sz")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--self-check", action="store_true", help="run on a tiny synthetic HDF5 and exit")
    ap.add_argument(
        "--download-only",
        action="store_true",
        help="fetch the --params files and exit (run this on a login node)",
    )
    ap.add_argument(
        "--n-files",
        type=int,
        default=1,
        help="files to take per --params token (one file is one trajectory on "
        "convective_envelope_rsg, a few hundred elsewhere)",
    )
    ap.add_argument(
        "--max-traj",
        type=int,
        default=0,
        help="fetch only this many trajectories of --field per file, by HTTP range "
        "reads, instead of the whole (212 GB on euler) file. Downloading only",
    )
    args = ap.parse_args()

    if args.self_check:
        return self_check(args)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    base = _base_path()
    params = args.params or [""]
    paths = [
        f
        for p in params
        for f in ensure_file(
            base, args.dataset, p, args.n_files, args.max_traj, args.field, "train"
        )
    ]
    # Tuning is on train, validation on test: the eval file is the test-split
    # one matching --test-params, or the first --params token. With --max-traj,
    # only enough trajectories to reach --test-traj are pulled.
    test_path = ensure_file(
        base,
        args.dataset,
        args.test_params or params[0],
        1,
        args.max_traj and args.test_traj + 1,
        args.field,
        "test",
    )[0]
    if args.download_only:
        return print("\n".join(str(p) for p in paths + [test_path]))
    args.field = args.field or default_field(paths[0])
    train = WellSlabs(paths, args.field, args.crop)
    test = WellSlabs([test_path], args.field, 0)
    test.index = [i for i in test.index if i[1] == args.test_traj]
    if not test.index:
        raise SystemExit(f"{test_path.name} has no trajectory {args.test_traj}")
    print(
        f"{args.dataset} field={args.field} ndim={train.ndim}   "
        f"train: {len(train)} trajectories over {len(paths)} file(s)   "
        f"test: {test_path.name}[{args.test_traj}]"
    )
    run(args, train, test, device, rng)


def run(args, train, test, device, rng):
    import wandb

    # --out names the checkpoint, never the path: it always lands in a fresh
    # per-run dir, so a rerun cannot clobber an earlier run's weights.
    name = Path(args.out).name if args.out else "gnn_well.pt"
    parent = Path(args.out).resolve().parent if args.out else Path("data").resolve()
    run_dir = parent / "runs" / f"well-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    out = run_dir / name
    last_out = out.with_name(f"{out.stem}-last{out.suffix}")  # latest eval weights
    print(f"run dir: {run_dir}")
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=vars(args),
        dir=str(run_dir),
    )

    ckpt = torch.load(args.init, map_location="cpu", weights_only=True)
    if ckpt.get("version") != CKPT_VERSION:
        raise SystemExit(
            f"{args.init}: checkpoint version {ckpt.get('version')} != {CKPT_VERSION}"
        )
    d, agg_level = ckpt["d"], ckpt["agg_level"]
    args.agg_level = agg_level  # eval_tensor_codec reads it off args
    model = build_model(d, agg_level).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"finetuning {args.init} (d={d}, agg_level={agg_level}) on {device}")

    crop, stride = _crop_stride(train.ndim)
    train.crop = crop if args.crop is None else args.crop
    args.stride = stride if args.stride is None else args.stride
    print(f"slabs: crop={train.crop} stride={args.stride}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    # linear warmup, then the same late-decay tail as before
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(
            1.0, (s + 1) / max(args.warmup, 1), (args.steps - s) / max(args.steps * 0.4, 1)
        ),
    )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay else None
    eval_model = build_model(d, agg_level).to(device) if ema else model

    def eval_weights():
        if ema is None:
            return model
        ema.copy_to(eval_model)
        eval_model.eval()
        return eval_model

    levels = args.levels or args.stride.bit_length() - 1
    geom_cache = {}

    def geoms_for(shape):
        if shape not in geom_cache:
            geom_cache[shape] = build_stage_geoms(
                shape, levels, args.stride, 1, args.max_radius, torch, device, agg_level
            )[0]
        return geom_cache[shape]

    # Held-out eval: the *whole* test-split trajectory (64^4 for this dataset),
    # roundtripped through the real chunked codec. A
    # closed-loop run_stages pass is not an option at that size (its per-point
    # embedding buffer alone would be tens of GB); the codec chunks it, and its
    # bits/value is the number that matters anyway.
    eval_tensor = test.trajectory(0, args.log, args.test_timesteps)
    print(f"held-out eval tensor: {eval_tensor.shape}")

    best = float("inf")

    def evaluate():
        nonlocal best
        m = eval_weights()
        with torch.no_grad():
            metrics = eval_tensor_codec(
                m, d, args, eval_tensor, args.eval_eb, device, run_dir / "eval_tensor.pt"
            )
        if m is model:
            model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        bpv = metrics["eval_tensor/bits_per_value"]
        save(last_out, m, d, agg_level, levels, args.stride)
        if bpv < best:
            best = bpv
            save(out, m, d, agg_level, levels, args.stride)
        return metrics

    # step 0: the un-finetuned baseline, so every later eval has a reference
    wandb.log(evaluate(), step=0)

    bar = tqdm(range(1, args.steps + 1), desc="finetune")
    for step in bar:
        idx = rng.integers(0, len(train), args.batch)
        vols = [train.slab(int(i), rng, args.log) for i in idx]
        x = torch.as_tensor(np.stack([v.reshape(-1) for v in vols])).to(device)
        eb = sample_noise(x.shape[0], args, device)
        opt.zero_grad()
        with training_autocast(args.fp16, device):
            nll, npix, _, _, aux = run_stages(
                model, x, geoms_for(vols[0].shape), d, device, eb=eb, teacher_force=True
            )
        loss = nll / max(npix, 1)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if ema is not None:
            ema.update(model, step)
        log = {"train/bpp": loss.item(), "lr": sched.get_last_lr()[0]}

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate()
            log.update(metrics)
            bpv = metrics["eval_tensor/bits_per_value"]
            bar.set_postfix(
                bpp=f"{log['train/bpp']:.4f}",
                bpv=f"{bpv:.4f}",
                psnr=f"{metrics['eval_tensor/psnr_db']:.2f}",
            )
        wandb.log(log, step=step)

    print(f"best held-out bits/value={best:.5f}; saved {out} (best) and {last_out}")
    wandb.finish()


def save(path, model, d, agg_level, levels=None, stride=None):
    """Write the weights, plus the schedule geometry they were tuned at.

    Finetuning specialises a model to its rollout depth: on supernova, tuning at
    levels=4 gained 12% there and 2% at levels=5, which is what ``levels="auto"``
    resolves to for a 4-D tensor. The pretrained model is depth-flat, so nothing
    upstream records this -- but a finetuned checkpoint is only worth its
    benchmark number at the depth it saw, so the depth travels with it.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d": d,
            "agg_level": agg_level,
            "version": CKPT_VERSION,
            "train_levels": levels,
            "train_stride": stride,
        },
        path,
    )


def _synthetic(path: Path, spatial: list[str], n_traj=3, n_time=12, n=8):
    """A tiny HDF5 in the Well layout, with `len(spatial)` spatial axes."""
    rng = np.random.default_rng(0)
    shape = (n_traj, n_time) + (n,) * len(spatial)
    with h5py.File(path, "w") as f:
        f.attrs["n_trajectories"] = n_traj
        f.create_group("dimensions").attrs["spatial_dims"] = spatial
        g = f.create_group("t0_fields")
        g.attrs["field_names"] = ["density"]
        g.create_dataset("density", data=rng.random(shape, np.float32))
        g = f.create_group("t1_fields")  # MHD-style vector field
        g.attrs["field_names"] = ["magnetic_field"]
        g.create_dataset(
            "magnetic_field", data=rng.random(shape + (len(spatial),), np.float32)
        )
        f.create_group("t2_fields").attrs["field_names"] = []
    return path


def self_check(args):
    """Build synthetic 2-D and 3-D HDF5s in the Well layout and finetune on one
    for a few steps; asserts the rank follows the file and that eval reads a
    separate test-split file."""
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    path = _synthetic(tmp / "sim_At_75.h5", ["x", "y", "z"])
    test_path = _synthetic(tmp / "sim_At_75_test.h5", ["x", "y", "z"])
    flat = _synthetic(tmp / "sim_2d.h5", ["x", "y"])

    with h5py.File(path, "r") as f:
        assert _field_dataset(f, "density")[1] is None
        assert _field_dataset(f, "magnetic_field_y")[1] == 1
    assert default_field(path) == "density"
    assert WellSlabs([path], "magnetic_field_z", 4).slab(0, rng).shape == (4,) * 4
    # rank follows the file: 2 spatial dims -> 3-D slabs, and --crop is clamped
    # per axis instead of erroring out.
    flat_slabs = WellSlabs([flat], "density", 4)
    assert flat_slabs.ndim == 3 and flat_slabs.slab(0, rng).shape == (4, 4, 4)
    assert WellSlabs([flat], "density", 64).slab(0, rng).shape == (12, 8, 8)
    assert flat_slabs.trajectory(0).shape == (8, 8, 8)
    # two anchor cells per axis at every rank, halving past 4-D
    assert _crop_stride(3) == (64, 32) and _crop_stride(4) == (32, 16)
    assert _crop_stride(5) == (16, 8)

    crop = 8
    train = WellSlabs([path], "density", crop)
    test = WellSlabs([test_path], "density", 0)
    test.index = [i for i in test.index if i[1] == 0]
    assert test.files[0].filename != train.files[0].filename, "eval must read the test split"
    assert len(train) == 3 and len(test.index) == 1
    vol = train.slab(0, rng, log=True)
    assert vol.shape == (crop,) * 4 and 0.0 <= vol.min() and vol.max() <= 1.0
    # n_time (12) is not a power of two: the eval tensor must be truncated.
    assert test.trajectory(0).shape == (8, 8, 8, 8)
    # ...unless the benchmark caps it somewhere else, which --test-timesteps says
    assert test.trajectory(0, n_time=10).shape == (10, 8, 8, 8)
    try:
        test.trajectory(0, n_time=13)
    except SystemExit:
        pass
    else:
        raise AssertionError("--test-timesteps past the end must fail loudly")

    args.steps, args.batch, args.crop, args.stride, args.warmup = 2, 1, crop, 4, 1
    args.wandb_mode, args.eval_every = "disabled", 2
    args.out = str(tmp / "ckpt.pt")
    run(args, train, test, torch.device(args.device), rng)
    (run_dir,) = (tmp / "runs").iterdir()  # fresh per-run dir, nothing clobbered
    for p in (run_dir / "ckpt.pt", run_dir / "ckpt-last.pt"):
        assert torch.load(p, map_location="cpu", weights_only=True)["version"] == CKPT_VERSION
    assert not (tmp / "ckpt.pt").exists()
    print("self-check OK")


if __name__ == "__main__":
    main()
