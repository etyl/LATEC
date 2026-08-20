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

# Pick a dataset by its key in well_datasets.yml, which holds the flags that
# make this read the same tensor the benchmark does:
#   WELL=mhd sbatch finetune_well_h100.sh
# A dataset not in there yet: add an entry, taking the flags from the benchmark
# rather than guessing (leave PARAMS empty and the split's *first* file is used,
# which is how a run ends up validating on another tensor than the benchmark
# reports). In benchmark-scientific-data-compression:
#   python check_ft_match.py
WELL=${WELL:-euler}
VARS=$(python -c 'import sys, yaml
d = yaml.safe_load(open(sys.argv[1]))[sys.argv[2]]
print("\n".join(f"{k.upper()}=\"{v}\"" for k, v in d.items()))' \
    "${SLURM_SUBMIT_DIR:-$(dirname "$0")}/well_datasets.yml" "$WELL") || exit 1
eval "$VARS"

# Finetune on hypercube crops (clamped to shorter axes) of the train split. The
# rank follows the dataset: 2-D sims train as 3-D tensors, 3-D sims as 4-D, and
# --crop/--stride default per rank (3-D 32/16, 4-D 16/8). A 3-D slab is 32k
# points and a 4-D one 65k: raise --batch until the GPU is busy. Crop scales as
# crop**rank, so doubling it on a 4-D dataset is 16x the memory and wants
# --batch 1.
# Validation is --test-traj of the *test* split, whole -- T truncated per
# TEST_TIMESTEPS -- run through the real chunked codec, so it takes minutes:
# keep --eval-every large. It codes at --eval-levels, default "auto", i.e. the
# depth deployment resolves to (5 for a 4-D tensor) rather than the depth the
# slabs train at -- that is what the best checkpoint is selected on.
# The schedule depth is sampled per optimizer step, from the deepest schedule
# --crop can hold (levels = log2(crop)) down _LEVEL_SPAN levels: 3-D crops at 32
# -> levels 3..5 and 4-D at 16 -> 2..4, each one short of the depth it deploys
# at (6 and 5), which is what the smaller crop costs. Doubling --crop buys that
# level back at 2**rank the memory per slab -- 3-D --crop 64 gives levels 4..6,
# --crop 128 gives 5..7 (a 3-D checkpoint at aggregation level 1), 4-D --crop 32
# --batch 1 gives 3..5. Pass --stride or
# --levels to pin one depth instead, for a deterministic, comparable profile.
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
