#!/bin/bash
#SBATCH --job-name=gnn-eval-tensor
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

module purge
module load pytorch-gpu

export PYTHONUNBUFFERED=1   # flush progress to the SLURM .out live
unset DEEPSZ_M_TILE

# D128 checkpoint
# CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260710-115201-7bbb4e/gnn_predictor.pt}
# D32 checkpoint
# CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260714-095239-e16d09/gnn_predictor.pt}
# CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260714-182442-d79742/gnn_predictor.pt}
CKPT=${CKPT:-/lustre/fswork/projects/rech/lzs/uhq13gg/MAT-SZ/data/runs/20260716-190203-07b676/gnn_predictor.pt}

DATA=${DATA:-/lustre/fswork/projects/rech/lzs/uhq13gg/benchmark-scientific-data-compression/rti_75_density.npy}
# DATA=${DATA:-./data/rti_rough.npy}

EB=${EB:-0.0001}
TUNE=${TUNE:-fast}               # fast (1 encode) | size/rd (4 encodes)
# SZ3 independently uses its tuned INTERP_LORENZO hybrid at every error bound.
TUNE_SIZE_SLACK=${TUNE_SIZE_SLACK:-1.05}

python scripts/eval_tensor.py "$DATA" \
    --gnn-checkpoint "$CKPT" \
    --predictor gnn \
    --eb "$EB" \
    --levels 5 \
    --tune "$TUNE" \
    --tune-size-slack "$TUNE_SIZE_SLACK" \
    --normalize \
    "$@"
