"""Plot a grid of 2-D synthetic training fields, drawn with the exact same
generator and default settings used by scripts/train_gnn.py (see
latec/synthetic_data.py:sample_synthetic_batch and synthetic_dist.yaml).

Usage:
    python scripts/plot_synthetic_2d.py                # 20 samples, random seed
    python scripts/plot_synthetic_2d.py --n 12 --seed 0
    python scripts/plot_synthetic_2d.py --crop 256 --out samples.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from latec.synthetic_data import DEFAULT_SYNTHETIC_DIST as _SYN
from latec.synthetic_data import sample_synthetic_batch


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--crop", type=int, default=_SYN.crop)
    p.add_argument("--out", default=str(ROOT / "synthetic_2d_samples.png"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    x = sample_synthetic_batch(
        args.n,
        (args.crop, args.crop),
        _SYN.correlation_2d,
        rng,
        turbulence_frac=_SYN.turbulence_frac,
        max_discontinuities=_SYN.max_discontinuities,
        discontinuity_frac=_SYN.discontinuity_frac,
    ).reshape(args.n, args.crop, args.crop).numpy()

    ncols = 5
    nrows = -(-args.n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        if i < args.n:
            ax.imshow(x[i], cmap="viridis", vmin=0, vmax=1)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
