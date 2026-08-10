#!/bin/bash
#SBATCH --job-name=gnn-well
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --time=2:00:00
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --account=lzs@h100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out

module purge
module load arch/h100
module load pytorch-gpu

unset LATEC_M_TILE

# Compute nodes have no internet: pre-download the .hdf5 on a login node (just
# run this script's python line there once, it curls then exits on the first
# eval) or copy it in. 25 GB per parameter combo, 80 trajectories of 50x64^3.
export THE_WELL_DATA_DIR=$WORK/benchmark-scientific-data-compression/data/the_well

# Simulation to finetune on. MHD_64 instead:
#   sbatch finetune_well_h100.sh --dataset MHD_64 --field magnetic_field_x
# (its default --params is 0.7,0.5; --params 2,7 etc. for another combo).
DATASET=${DATASET:-MHD_64}

# Finetune on 4-D hypercube crops (16 cells on time and each spatial axis) from
# the non-held-out trajectories of the default-params file. Held-out eval is
# trajectory 0 whole -- T truncated to a power of two -- run through the real
# chunked codec, so it takes minutes: keep --eval-every large.
# stride 16 gives 2 anchors per 16-axis. Add --log for fields spanning decades.
python scripts/finetune_well.py \
    --init checkpoints/v7-d64-1agg.pt \
    --out data/gnn_well.pt \
    --dataset "$DATASET" \
    --field density \
    --test-traj 0 \
    --steps 3000 \
    --batch 16 \
    --crop 16 \
    --stride 16 \
    --warmup 500 \
    --lr 0.0001 \
    --noise-range 0.000001 0.01 \
    --eval-eb 0.001 \
    --eval-every 200 \
    --device cuda \
    --wandb-mode offline \
    --run-name "gnn-$DATASET" \
    "$@"

# per-run dir -> data/runs/well-<timestamp>/; best + <out>-last checkpoints
