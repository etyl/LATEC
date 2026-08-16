#!/bin/bash
#SBATCH --job-name=gnn-well
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=6:00:00
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

# Compute nodes have no internet: pre-download on a login node with
#   THE_WELL_DATA_DIR=... python scripts/finetune_well.py --download-only \
#       --dataset "$DATASET" --params "$PARAMS" --field "$FIELD" \
#       --max-traj $MAX_TRAJ
# That pulls both splits it needs: train to tune on, test to validate on.
# MAX_TRAJ=0 curls whole files (euler train: 212 GB, 400 trajectories, all
# fields); set it >0 to range-read only that many trajectories of $FIELD
# instead. Same flags here, so the job finds them on disk.
export THE_WELL_DATA_DIR=$WORK/benchmark-scientific-data-compression/data/the_well

# Any The Well dataset name works, e.g.
#   DATASET=MHD_64 PARAMS=MHD_Ma_0.7_Ms_0.5 sbatch finetune_well_h100.sh
#   DATASET=active_matter FIELD=concentration sbatch finetune_well_h100.sh
# PARAMS is a substring of the wanted file's name; FIELD empty = the file's
# first scalar field. Repeat --params on the command line to train on several
# files at once.
#
# Leave PARAMS empty and the split's *first* file is used -- and ensure_file
# prefers files already on disk, so which one that is depends on the node. The
# benchmark picks by its own glob, so an empty PARAMS is how a run ends up
# validating on a different tensor than the benchmark reports. Get the flags
# from the benchmark instead of guessing:
#   python check_ft_match.py   # in benchmark-scientific-data-compression
# TEST_PARAMS: test-split file, when it is not the PARAMS one (convective_
# envelope_rsg splits by trajectory, so it always differs there).
# TEST_TIMESTEPS: the benchmark's n_timesteps cap, when it has one (euler and
# convective 64, gray_scott 256); 0 = the largest power of two.
DATASET=${DATASET:-supernova_explosion_128}
PARAMS=${PARAMS:-supernova_explosion_Msun_0.1_dim128_file_00}
FIELD=${FIELD:-density}
TEST_PARAMS=${TEST_PARAMS:-}
TEST_TIMESTEPS=${TEST_TIMESTEPS:-0}

# Finetune on hypercube crops (clamped to shorter axes) of the train split. The
# rank follows the dataset: 2-D sims train as 3-D tensors, 3-D sims as 4-D, and
# --crop/--stride default per rank to inference's block size (3-D 64/32, 4-D
# 32/16). A 3-D slab is 256k points: raise --batch until the GPU is busy.
# Validation is --test-traj of the *test* split, whole -- T truncated per
# TEST_TIMESTEPS -- run through the real chunked codec, so it takes minutes:
# keep --eval-every large.
# Add --log for fields spanning decades.
python scripts/finetune_well.py \
    --init checkpoints/v7-d64-1agg.pt \
    --out data/gnn_well.pt \
    --dataset "$DATASET" \
    --params "$PARAMS" \
    ${TEST_PARAMS:+--test-params "$TEST_PARAMS"} \
    ${FIELD:+--field "$FIELD"} \
    --test-traj 0 \
    --test-timesteps "$TEST_TIMESTEPS" \
    --steps 10000 \
    --batch 16 \
    --warmup 500 \
    --lr 0.00005 \
    --noise-range 0.000001 0.01 \
    --eval-eb 0.001 \
    --eval-every 500 \
    --device cuda \
    --wandb-mode offline \
    --run-name "gnn-$DATASET" \
    "$@"

# per-run dir -> data/runs/well-<timestamp>/; best + <out>-last checkpoints
