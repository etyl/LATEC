#!/bin/bash
#SBATCH --job-name=gnn-bench-subset
#SBATCH --qos=qos_gpu-dev
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_p13
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --account=lzs@v100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

# Benchmark the GNN codec on a fixed 64^4 (default) centred subset of a large
# tensor, reporting PSNR / bpp / time / max RAM / mean GPU utilisation so an
# optimisation can be judged before-vs-after. Every run prints its report to
# the SLURM .out, so re-run this on two commits and diff the two reports.
#
#   sbatch bench_subset.sh                       # defaults below
#   LABEL=baseline sbatch bench_subset.sh        # tag the report header
#   EDGE=32 EB=1e-3 sbatch bench_subset.sh       # override any knob via env

module purge
module load pytorch-gpu

export PYTHONUNBUFFERED=1          # flush progress to the SLURM .out live
unset LATEC_M_TILE

# GNN checkpoint (same one eval_tensor.sh uses; override with CKPT=...).
CKPT=${CKPT:-./checkpoints/v7-d64-1agg.pt}

# Large source tensor; a centred EDGE^ndim hypercube is cropped out of it.
DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy}

EDGE=${EDGE:-128}                   # subset edge (capped per axis, floored to stride)
EB=${EB:-0.0001}
LEVELS=${LEVELS:-5}
TUNE=${TUNE:-fast}
LABEL=${LABEL:-}

python scripts/bench_gnn_subset.py "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --subset-edge "$EDGE" \
    --eb "$EB" \
    --levels "$LEVELS" \
    --tune "$TUNE" \
    --label "$LABEL" \
    --fp16 \
    "$@"
