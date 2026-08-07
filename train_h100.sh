#!/bin/bash
#SBATCH --job-name=gnn-sz
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --time=8:00:00
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

# The profile is launch-bound (thousands of ~10 us kernels per step). Ask
# torch.compile to use CUDA graphs where shapes permit, reducing CPU dispatch
# overhead without changing the batch or model.
export LATEC_COMPILE_MODE=reduce-overhead

# Training must run embed untiled (max-M kernels, fewest launches). sbatch
# inherits the submitting shell's env, so drop any stray LATEC_M_TILE.
unset LATEC_M_TILE

# 16*128^2 2-D + 4*16^4 4-D synthetic = 524,288 total points (50% 4-D). All
# training data is synthetic now (no natural images): the 2-D branch draws
# --crop x --crop fields, the 4-D branch --synthetic-shape fields, from the same
# generator. synthetic-stride 8 (< field 16) makes each 4-D field span 2^4
# chunks, so the chunked/coarse path (CoarseProj) trains on n-D. Each step splits
# the 4-D batch in half: half full-level dense, half chunked coarse. Fields are
# generated on the GPU by train_gnn.py.
# --synthetic-correlation / --synthetic-2d-correlation are *bands* (MIN MAX):
# each field draws a base smoothness in [MIN/2, MAX], a random per-axis spread
# (-> isotropic..anisotropic), and a random value-marginal warp (unimodal..
# bimodal). --synthetic-turbulence-frac gives a share of power-law turbulent
# (fractal) spectra, and --synthetic-discontinuities overlays sharp jump fronts
# (shocks / interfaces), so the prior spans diverse scientific fields (smooth,
# turbulent, and discontinuous) rather than one texture a wide model overfits.
# --noise-range reaches 1e-6: the old floor of 1e-4 meant the model never saw a
# context cleaner than 1e-4 and had no gradient incentive to predict below it,
# leaving a ~1e-5 bulk-error floor that costs ~3 bits/pt at eb=1e-6 (interp,
# being analytic, has no such floor). Pairs with the widened rANS scale grid
# (rans.SCALE_HI_MULT 64->4096) + head clamp (-4..12): sub-1e-4 ebs need scale
# levels the old grid couldn't express. Tensor eval at 1e-5 watches this regime.
python scripts/train_gnn.py \
    --out data/gnn_predictor.pt \
    --steps 50000 \
    --save-every 5000 \
    --batch 16 \
    --crop 128 \
    --synthetic-frac 0.5 \
    --synthetic-shape 16 16 16 16 \
    --synthetic-correlation 1.0 8 \
    --synthetic-2d-correlation 2.0 32 \
    --synthetic-turbulence-frac 0.5 \
    --synthetic-discontinuities 3 \
    --synthetic-batch 4 \
    --synthetic-stride 8 \
    --agg-level 1 \
    --d 64 \
    --lr 0.0005 \
    --noise-range 0.000001 0.01 \
    --eval-shape 256 256 \
    --eval-eb 0.001 \
    --eval-every 500 \
    --img-every 10000 \
    --eval-tensor ./data/rti_normal.npy \
    --eval-tensor-normalize \
    --eval-tensor-eb 0.001 \
    --eval-tensor-every 500 \
    --device cuda \
    --wandb-mode offline \
    --run-name gnn-agg1  \
    --compile \
    "$@"

# per-run dir -> data/runs/<date>-<hash>/ (checkpoint + config.json)
