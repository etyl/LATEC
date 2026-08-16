"""The Well data layer: registry download and HDF5 layout, with no torch import.

Run it to fetch training data before a job (compute nodes have no internet):

    python scripts/well_data.py --dataset convective_envelope_rsg --n-files 3

--params is a substring of the wanted files' names (the parameter combination,
"" takes all of them) and repeats; --n-files caps how many matching files are
pulled. On datasets that store one trajectory per file (convective_envelope_rsg)
that is the trajectory count; elsewhere a file holds a few hundred.

Some files are one big bag of trajectories -- euler_multi_quadrants_openBC's
train file is 212 GB, 400 trajectories -- so --max-traj N reads just the first
N of --field out of the remote file over HTTP range requests and writes them
locally in the same layout:

    python scripts/well_data.py --dataset euler_multi_quadrants_openBC \\
        --params gamma_1.4_Dry_air_20 --field density --max-traj 32

--split defaults to train, which is the one to finetune on; valid/test are
smaller (euler 26.5 GB against 212) but are the benchmark's held-out numbers.

Files land under $THE_WELL_DATA_DIR (or ~/.cache/the_well), in the layout
the_well's own downloader uses, and are skipped if already there.
"""

from __future__ import annotations

import http.client
import io
import os
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import h5py
import numpy as np



def _base_path() -> Path:
    for env_var in ("THE_WELL_DATA_DIR", "WELL_DATA_DIR"):
        if os.environ.get(env_var):
            return Path(os.environ[env_var]).expanduser()
    return Path.home() / ".cache" / "the_well"


def _registry() -> dict:
    import yaml
    from the_well.utils.download import WELL_REGISTRY

    with open(WELL_REGISTRY) as f:
        return yaml.safe_load(f)


class _HttpFile(io.RawIOBase):
    """Read-only file over HTTP range requests, so h5py can pull just the bytes
    it asks for out of a remote Well file (they run to hundreds of GB).

    One connection is kept open for all of them: a site proxy that has to tunnel
    a fresh CONNECT per read starts answering 403 partway through (Jean Zay
    does), and a whole-file fetch is tens of thousands of reads.
    """

    def __init__(self, url: str):
        self.url = url
        self._pos = 0
        self._conn = None
        r = self._request(method="HEAD")
        self.size = int(r.headers["Content-Length"])
        r.read()

    def _connect(self):
        # ponytail: https only -- every registry URL is https.
        u = urllib.parse.urlsplit(self.url)
        proxy = urllib.request.getproxies().get("https")
        if proxy:
            p = urllib.parse.urlsplit(proxy if "://" in proxy else f"http://{proxy}")
            conn = http.client.HTTPSConnection(p.hostname, p.port or 80, timeout=120)
            conn.set_tunnel(u.hostname, u.port or 443)
        else:
            conn = http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=120)
        self._conn = conn
        self._path = urllib.parse.urlunsplit(("", "", u.path, u.query, ""))

    def _request(self, method="GET", headers=None):
        """One request on the shared connection, reconnecting and retrying when
        the proxy or the server drops it."""
        for attempt in range(6):
            try:
                if self._conn is None:
                    self._connect()
                self._conn.request(method, self._path, headers=headers or {})
                return self._conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
                if attempt == 5:
                    raise SystemExit(
                        f"cannot reach {self.url}: {e}\nif this host has no internet, "
                        "fetch the data where it does and rsync it into "
                        "$THE_WELL_DATA_DIR; if it needs a proxy, set https_proxy"
                    )
                time.sleep(2**attempt)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        self._pos = (off, self._pos + off, self.size + off)[whence]
        return self._pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self._pos)
        if n <= 0:
            return 0
        r = self._request(headers={"Range": f"bytes={self._pos}-{self._pos + n - 1}"})
        data = r.read()
        buf[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self):
        if self._conn is not None:
            self._conn.close()
        super().close()


def fetch_subset(url: str, path: Path, n_traj: int, field: str | None) -> None:
    """Write a Well-layout HDF5 holding only the first `n_traj` trajectories of
    one field, reading them straight off the remote file by HTTP range."""
    part = path.with_suffix(".part")
    src_file = io.BufferedReader(_HttpFile(url), buffer_size=1 << 22)
    # "a": a half-written .part is resumed from its "done" mark, so a dropped
    # connection an hour in costs the last timestep, not the whole fetch.
    with h5py.File(src_file, "r") as src, h5py.File(part, "a") as dst:
        field = field or default_field(src)
        dset, _ = _field_dataset(src, field)
        name = dset.name.rsplit("/", 1)[-1]
        n = min(n_traj, dset.shape[0])
        group_name = dset.parent.name.strip("/")
        if group_name in dst:
            out = dst[group_name][name]
        else:
            dst.create_group("dimensions").attrs["spatial_dims"] = _spatial_dims(src)
            group = dst.create_group(group_name)
            group.attrs["field_names"] = [name]
            out = group.create_dataset(name, shape=(n,) + dset.shape[1:], dtype=dset.dtype)
            out.attrs["done"] = 0
        steps = dset.shape[1]
        done = int(out.attrs["done"])
        total_gb = out.dtype.itemsize * np.prod(out.shape) / 1e9
        for i in range(n):
            # ponytail: a timestep at a time -- a whole 3-D trajectory is several
            # GB in RAM (256x128x256x100 f32 = 3.3 GB) and gets the process
            # OOM-killed on a login node.
            for t in range(steps):
                if i * steps + t < done:
                    continue
                out[i, t] = dset[i, t]
                done = i * steps + t + 1
                out.attrs["done"] = done
                print(
                    f"\r{path.name}: {done}/{n * steps} timesteps "
                    f"({total_gb * done / (n * steps):.2f}/{total_gb:.2f} GB)",
                    end="",
                    flush=True,
                )
            dst.flush()
        del out.attrs["done"]
    print()
    part.rename(path)


def _existing_subset(split_dir: Path, stem: str, field: str | None, max_traj: int) -> Path | None:
    """The largest already-fetched `<stem>.<field>-<N>traj.hdf5` with N >= max_traj,
    or None. With no --field, any field's subset matches: the caller's default is
    that file's first scalar field, which cannot be known before it is on disk."""
    best = None
    for path in split_dir.glob(f"{stem}.{field or '*'}-*traj.hdf5"):
        try:
            n = int(path.name.rsplit("-", 1)[1].removesuffix("traj.hdf5"))
        except ValueError:
            continue
        if n >= max_traj and (best is None or n > best[0]):
            best = (n, path)
    return best[1] if best else None


def _have_locally(split_dir: Path, url: str, field: str | None, max_traj: int) -> bool:
    """Is this registry file usable without downloading — whole, or as a subset?"""
    return (split_dir / Path(url).name).exists() or _existing_subset(
        split_dir, Path(url).stem, field, max_traj
    ) is not None


def ensure_file(
    base_path: Path,
    dataset: str,
    token: str = "",
    n_files: int = 1,
    max_traj: int = 0,
    field: str | None = None,
    split: str = "train",
) -> list[Path]:
    """Paths to `dataset`'s training files whose name contains `token` (the
    parameter combination; "" matches all), downloading the first `n_files` of
    them through the_well's downloader if they are not already on disk. With
    `max_traj`, only that many trajectories of `field` are fetched per file."""
    # ponytail: substring match, not exact stem -- Well filenames carry extra
    # name parts and each dataset spells its parameters differently.
    import yaml
    from the_well.utils.download import well_download

    registry = _registry()
    if dataset not in registry:
        raise SystemExit(f"unknown dataset {dataset!r}: have {sorted(registry)}")
    split_dir = base_path / "datasets" / dataset / "data" / split
    matches = [u for u in registry[dataset][split] if token in Path(u).name]
    if not matches:
        raise SystemExit(
            f"no {dataset} {split} file matching {token!r}: have "
            f"{sorted(Path(u).name for u in registry[dataset][split])}"
        )
    # Already-downloaded matches first (stable, so registry order breaks ties):
    # with a loose token the registry's first file is an arbitrary pick, and
    # picking one that is not on disk means a fresh multi-GB download -- or, on a
    # compute node with no internet, a hard failure -- next to a usable file.
    urls = sorted(matches, key=lambda u: not _have_locally(split_dir, u, field, max_traj))[:n_files]

    paths = []
    for url in urls:
        whole = split_dir / Path(url).name
        # An already-fetched subset of at least max_traj trajectories stands in
        # for the fetch, and for the whole file too (max_traj=0): downloading
        # hundreds of GB next to a subset that already covers the run is never
        # what the caller meant.
        subset = _existing_subset(split_dir, Path(url).stem, field, max_traj)
        if subset is not None and not whole.exists():
            print(f"using subset {subset.name}")
            paths.append(subset)
        elif max_traj:
            # field in the name: a subset holds that one field, so another field
            # must not silently reuse the file. The marker goes after a dot so a
            # dot-terminated --params token still matches the name.
            path = split_dir / f"{Path(url).stem}.{field or 'first'}-{max_traj}traj.hdf5"
            path.parent.mkdir(parents=True, exist_ok=True)
            fetch_subset(url, path, max_traj, field)
            paths.append(path)
        else:
            paths.append(whole)
    if max_traj:
        return paths

    missing = [u for u, p in zip(urls, paths) if not p.exists()]
    if missing:
        # well_download takes a registry, not a URL list, so hand it a trimmed
        # copy -- that reuses its resumable curl and its stats.yaml fetch.
        trimmed = {dataset: {"stats": registry[dataset]["stats"], split: missing}}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            yaml.dump(trimmed, f)
            f.flush()
            well_download(str(base_path), dataset, split, registry_path=f.name)
    for p in paths:
        if not p.exists():
            have = sorted(q.name for q in split_dir.glob("*.hdf5")) if split_dir.is_dir() else []
            raise SystemExit(
                f"{p} was not downloaded (no internet on this node?). "
                f"{split_dir} holds: {have or 'nothing'}. Fetch it on a login node, "
                f"or pass --params matching a file you have"
            )
        try:  # curl resumes, so a killed download leaves a plausible-looking file
            h5py.File(p, "r").close()
        except OSError as e:
            raise SystemExit(f"{p} is incomplete, delete it and retry ({e})")
    return paths


def _decode(names) -> list[str]:
    return [n.decode() if isinstance(n, bytes) else str(n) for n in names]


def _spatial_dims(file: h5py.File) -> list[str]:
    return _decode(file["dimensions"].attrs["spatial_dims"])


def _field_names(file: h5py.File) -> dict:
    """field name -> the h5 group holding it."""
    groups = {}
    for group_name in ("t0_fields", "t1_fields", "t2_fields"):
        if group_name not in file:
            continue
        group = file[group_name]
        for n in _decode(group.attrs["field_names"]):
            groups[n] = group
    return groups


def default_field(file) -> str:
    """First scalar (t0) field of an open file or a path — the sane default for
    any dataset."""
    if not isinstance(file, h5py.File):
        with h5py.File(file, "r") as f:
            return default_field(f)
    names = _decode(file["t0_fields"].attrs["field_names"]) if "t0_fields" in file else []
    if not names:
        raise SystemExit(f"{file.filename} has no scalar field; pass --field")
    return names[0]


def _field_dataset(file: h5py.File, field: str):
    """(h5 dataset, component index or None) for a Well field name.

    Scalars are stored under their own name; a vector component is written
    `<vector>_<x|y|z>`, so fall back to splitting on the last underscore.
    """
    groups = _field_names(file)
    if field in groups:
        return groups[field][field], None
    base, _, suffix = field.rpartition("_")
    if base in groups:
        spatial = _spatial_dims(file)
        if suffix in spatial:
            return groups[base][base], spatial.index(suffix)
    raise SystemExit(f"field {field!r} not found in {file.filename}: have {sorted(groups)}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, help="The Well registry dataset name")
    ap.add_argument("--params", action="append", default=None, metavar="TOKEN")
    ap.add_argument("--n-files", type=int, default=1, help="files per --params token")
    ap.add_argument("--split", default="train", choices=("train", "valid", "test"))
    ap.add_argument("--field", default=None, help="default: the file's first scalar field")
    ap.add_argument(
        "--max-traj",
        type=int,
        default=0,
        help="fetch only this many trajectories of --field per file (0 = whole file)",
    )
    args = ap.parse_args()
    for token in args.params or [""]:
        for path in ensure_file(
            _base_path(), args.dataset, token, args.n_files, args.max_traj, args.field,
            args.split,
        ):
            print(f"{path}  ({path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
