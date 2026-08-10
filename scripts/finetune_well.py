"""Finetune the GNN predictor on a 4-D (T, X, Y, Z) The Well dataset.

--dataset picks the simulation (see DATASETS): turbulence_gravity_cooling or
MHD_64. --field takes a scalar name (density) or a vector component
(magnetic_field_x, velocity_z).

Training fields are 4-D hypercube crops of a trajectory — --crop cells along
time and each spatial axis — matching how the benchmark feeds the codec the
whole (T, X, Y, Z) trajectory. Crops come from trajectories *other* than the
held-out one.

Eval is that whole held-out trajectory — 64^4 for this dataset, T truncated to
a power of two, never sampled during training — pushed through the real codec,
so the reported bits/value and PSNR are the deployed numbers. The best-scoring
weights are the ones written out.

    python scripts/finetune_well.py --init data/gnn_predictor.pt --steps 500

Extra parameter combinations can be mixed into the training set with repeated
--train-params RHO0,Z,T0 (each one is a separate multi-GB download).

ponytail: reuses train_gnn.py's stage loop, EMA, noise sampling and codec eval
instead of re-implementing them; only the data source is new.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

from latec.gnn_predictor import (  # noqa: E402
    CKPT_VERSION,
    build_model,
    build_stage_geoms,
)

SPLIT = "train"
# dataset -> (file-stem template, default held-out params)
DATASETS = {
    "turbulence_gravity_cooling": (
        "turbulence_gravity_cooling_rho0_{}_Z_{}_T0_{}",
        ("4.45", "1", "100"),
    ),
    "MHD_64": ("MHD_Ma_{}_Ms_{}", ("0.7", "0.5")),
}


def _base_path() -> Path:
    for env_var in ("THE_WELL_DATA_DIR", "WELL_DATA_DIR"):
        if os.environ.get(env_var):
            return Path(os.environ[env_var]).expanduser()
    return Path.home() / ".cache" / "the_well"


def ensure_file(base_path: Path, dataset: str, params: tuple[str, ...]) -> Path:
    """Path to the HDF5 file for one parameter combination, downloading it from
    The Well registry (via curl) if it is not already on disk."""
    template, _ = DATASETS[dataset]
    stem = template.format(*params)
    split_dir = base_path / "datasets" / dataset / "data" / SPLIT
    for suffix in (".h5", ".hdf5"):
        path = split_dir / f"{stem}{suffix}"
        if path.is_file() and path.stat().st_size > 0:
            return path

    import yaml
    from the_well.utils.download import WELL_REGISTRY

    with open(WELL_REGISTRY) as f:
        registry = yaml.safe_load(f)
    urls = [u for u in registry[dataset][SPLIT] if Path(u).stem == stem]
    if not urls:
        raise SystemExit(f"unknown {dataset} parameters: {params}")
    path = split_dir / Path(urls[0]).name
    subprocess.run(
        ["curl", "--create-dirs", "--continue-at", "-", "-o", str(path), urls[0]],
        check=True,
    )
    return path


def _field_dataset(file: h5py.File, field: str):
    """(h5 dataset, component index or None) for a Well field name.

    Scalars are stored under their own name; a vector component is written
    `<vector>_<x|y|z>`, so fall back to splitting on the last underscore.
    """
    groups = {}
    for group_name in ("t0_fields", "t1_fields", "t2_fields"):
        group = file[group_name]
        for n in group.attrs["field_names"]:
            groups[n.decode() if isinstance(n, bytes) else str(n)] = group
    if field in groups:
        return groups[field][field], None
    base, _, suffix = field.rpartition("_")
    if base in groups:
        spatial = [
            d.decode() if isinstance(d, bytes) else str(d)
            for d in file["dimensions"].attrs["spatial_dims"]
        ]
        if suffix in spatial:
            return groups[base][base], spatial.index(suffix)
    raise SystemExit(f"field {field!r} not found in {file.filename}: have {sorted(groups)}")


class WellSlabs:
    """Random 4-D hypercube crops (time, x, y, z) of trajectories, across files.

    Files stay open and only the crop is read off disk. `--crop` is the side
    along *all four* axes, time included, so a slab is a hypercube; 0 keeps each
    axis whole.
    """

    def __init__(self, paths, field, crop, exclude=()):
        self.files = [h5py.File(p, "r") for p in paths]
        self.field = field
        self.crop = crop
        self.index = []  # (file_idx, trajectory)
        for fi, file in enumerate(self.files):
            dset, _ = _field_dataset(file, field)
            if crop and any(n < crop for n in dset.shape[1:5]):
                raise SystemExit(f"--crop {crop} exceeds axes {dset.shape[1:5]}")
            self.index += [
                (fi, traj)
                for traj in range(dset.shape[0])
                if (Path(file.filename), traj) not in exclude
            ]
        if not self.index:
            raise SystemExit("no training slabs found")

    def __len__(self):
        return len(self.index)

    def slab(self, i, rng, log=False) -> np.ndarray:
        """One normalized [0, 1] hypercube; optional log10 first, since density
        and pressure span decades."""
        fi, traj = self.index[i]
        dset, component = _field_dataset(self.files[fi], self.field)
        sides = [self.crop or n for n in dset.shape[1:5]]
        starts = [rng.integers(0, n - side + 1) for n, side in zip(dset.shape[1:5], sides)]
        sl = tuple(slice(s, s + side) for s, side in zip(starts, sides))
        vol = np.asarray(dset[(traj,) + sl], dtype=np.float32)
        if component is not None:
            vol = vol[..., component]
        return self._prepare(vol, log)

    def trajectory(self, i, log=False) -> np.ndarray:
        """One whole trajectory (T, X, Y, Z), T truncated to the largest power
        of two — the tensor the benchmark hands the codec."""
        fi, traj = self.index[i]
        dset, component = _field_dataset(self.files[fi], self.field)
        n_time = 1 << (int(dset.shape[1]).bit_length() - 1)
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
    ap.add_argument("--dataset", choices=tuple(DATASETS), default="turbulence_gravity_cooling")
    ap.add_argument("--field", default="density")
    ap.add_argument("--log", action="store_true", help="log10 the field before normalizing")
    ap.add_argument(
        "--params",
        default=None,
        metavar="P1,P2,...",
        help="parameters of the held-out file (default: the dataset's own)",
    )
    ap.add_argument(
        "--train-params",
        action="append",
        default=None,
        metavar="P1,P2,...",
        help="extra parameter combinations to train on (repeatable); the "
        "held-out file's other trajectories are always included",
    )
    ap.add_argument("--test-traj", type=int, default=0, help="held-out trajectory index")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=1, help="slabs per optimizer step")
    ap.add_argument(
        "--crop",
        type=int,
        default=32,
        help="side of the random 4-D hypercube crop, time axis included "
        "(0 = the whole trajectory)",
    )
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500, help="linear lr warmup steps")
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--noise-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    ap.add_argument("--stride", type=int, default=16)
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
    args = ap.parse_args()

    if args.self_check:
        return self_check(args)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    base = _base_path()
    params = tuple(args.params.split(",")) if args.params else DATASETS[args.dataset][1]
    test_path = ensure_file(base, args.dataset, params)
    paths = [test_path] + [
        ensure_file(base, args.dataset, tuple(p.split(",")))
        for p in (args.train_params or [])
    ]
    train = WellSlabs(
        paths, args.field, args.crop, exclude={(test_path, args.test_traj)}
    )
    test = WellSlabs([test_path], args.field, 0)
    test.index = [i for i in test.index if i[1] == args.test_traj]
    print(f"train trajectories: {len(train)}   held-out: {len(test.index)}")
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

    # Held-out eval: the *whole* test trajectory (64^4 for this dataset), never
    # sampled during training, roundtripped through the real chunked codec. A
    # closed-loop run_stages pass is not an option at that size (its per-point
    # embedding buffer alone would be tens of GB); the codec chunks it, and its
    # bits/value is the number that matters anyway.
    eval_tensor = test.trajectory(0, args.log)
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
        save(last_out, m, d, agg_level)
        if bpv < best:
            best = bpv
            save(out, m, d, agg_level)
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


def save(path, model, d, agg_level):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d": d,
            "agg_level": agg_level,
            "version": CKPT_VERSION,
        },
        path,
    )


def self_check(args):
    """Build a 3-trajectory synthetic HDF5 in the Well layout and finetune on
    it for a few steps; asserts the held-out trajectory is never trained on."""
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    path = tmp / (DATASETS["turbulence_gravity_cooling"][0].format("4.45", "1", "100") + ".h5")
    n_traj, n_time, n = 3, 12, 8
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.attrs["n_trajectories"] = n_traj
        f.create_group("dimensions").attrs["spatial_dims"] = ["x", "y", "z"]
        g = f.create_group("t0_fields")
        g.attrs["field_names"] = ["density"]
        g.create_dataset("density", data=rng.random((n_traj, n_time, n, n, n), np.float32))
        g = f.create_group("t1_fields")  # MHD-style vector field
        g.attrs["field_names"] = ["magnetic_field"]
        g.create_dataset(
            "magnetic_field", data=rng.random((n_traj, n_time, n, n, n, 3), np.float32)
        )
        f.create_group("t2_fields").attrs["field_names"] = []

    with h5py.File(path, "r") as f:
        assert _field_dataset(f, "density")[1] is None
        assert _field_dataset(f, "magnetic_field_y")[1] == 1
    assert WellSlabs([path], "magnetic_field_z", 4).slab(0, rng).shape == (4,) * 4

    crop = 8
    train = WellSlabs([path], "density", crop, exclude={(path, 0)})
    test = WellSlabs([path], "density", 0)
    test.index = [i for i in test.index if i[1] == 0]
    assert all(traj != 0 for _, traj in train.index), "held-out trajectory leaked"
    assert len(train) == n_traj - 1 and len(test.index) == 1
    vol = train.slab(0, rng, log=True)
    assert vol.shape == (crop,) * 4 and 0.0 <= vol.min() and vol.max() <= 1.0
    # n_time is not a power of two here: the eval tensor must be truncated.
    assert test.trajectory(0).shape == (8, n, n, n)

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
