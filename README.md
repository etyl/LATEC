# LATEC

LATEC is an error-bounded lossy compressor for images and scientific tensors.
It uses progressive closed-loop prediction with either SZ3-style interpolation
or a trained, dimension-agnostic graph neural network (GNN). Quantized residuals
are entropy-coded and wrapped in a zstd frame.

The encoder reconstructs every stage before predicting the next one. The
decoder therefore sees the same context, and every decoded value satisfies the
requested absolute error bound.

## Installation

LATEC requires Python 3.10 or newer.

```bash
pip install -e ".[image,test]"       # interpolation, CLI, and development tools
pip install -e ".[gnn,image,test]"   # add PyTorch-backed GNN compression
```

## Python API

The tensor-oriented GNN API accepts NumPy arrays or PyTorch tensors of any rank:

```python
from latec import GNNCodec

codec = GNNCodec("data/gnn_predictor.pt", error_bound=0.01)
stream = codec.compress(array_or_tensor)
reconstructed = codec.uncompress(stream)  # torch.Tensor
```

The GNN codec derives its anchor stride as `2 ** levels`; inference callers
only specify `levels`, so the dyadic schedule always reaches unit stride.

The stream records the tensor shape, dtype, codec parameters, numerical mode,
and checkpoint hash. Decoding checks the checkpoint hash by default. Large
tensors are automatically divided into the largest dependency-safe chunks that
fit the codec's point budget and processed one at a time; set `chunk_size=0` to
force the whole-tensor path.

The lower-level API supports both predictors and returns codec diagnostics:

```python
from latec import compress, decompress
from latec.predictor import InterpPredictor

predictor = InterpPredictor("cubic", levels=4, anchor_stride=16)
stream, stats = compress(values, 0.01, predictor)
reconstructed = decompress(stream)
```

## Command Line

```bash
latec compress input.png output.msz --eb 2 --predictor interp
latec decompress output.msz reconstructed.png
latec eval input.png --eb 2 --predictor interp
```

Use `--predictor gnn --gnn-checkpoint PATH` for checkpoint-backed prediction.
`--rel` interprets the bound relative to the input range, while `--tune size`
tests several coarse-stage error schedules and retains the smallest stream.

## GNN Training and Evaluation

The trainer runs entirely on synthetic data — no natural images or external
datasets required. It mixes 2-D fields with anisotropic n-D fields drawn from
one generator that spans smooth Gaussian random fields, power-law turbulence,
and sharp random-hyperplane discontinuities (shocks / interfaces):

```bash
python scripts/train_gnn.py \
    --synthetic-frac 0.5 \
    --synthetic-shape 16 16 16 16 \
    --synthetic-correlation 1 8 \
    --synthetic-2d-correlation 2 32 \
    --synthetic-turbulence-frac 0.5 \
    --synthetic-discontinuities 3 \
    --synthetic-stride 8 \
    --agg-level 2
```

Run `python scripts/train_gnn.py --help` for all training options and
`python scripts/eval_predictors.py --help` for rate-distortion evaluation.
Root-level `train*.sh`, `eval*.sh`, and `profile*.sh` files are Jean Zay Slurm
launchers. Evaluation and profiling launchers require an explicit `CKPT` value:

```bash
CKPT=/path/to/gnn_predictor.pt sbatch eval_tensor.sh
```

Large Jean Zay jobs must be submitted directly by the user.

## Codec Design

For each progressive stage, the predictor estimates only newly introduced grid
points. Residuals are quantized with a linear error-bounded quantizer; values
that cannot be represented safely are stored as exact outliers. Interpolation
streams use canonical Huffman coding. GNN streams additionally predict a local
Laplacian scale and use scale-conditioned rANS coding.

Both supported predictors operate on the complete field without padding or
prediction seams. The dedicated GNN tensor codec adds chunking for bounded GPU
memory while preserving the dependency order between chunks.

## Repository Layout

- `latec/` contains codecs, predictors, scheduling, quantization, and entropy coding.
- `tests/` contains deterministic unit and round-trip coverage.
- `scripts/` contains training, evaluation, plotting, and profiling utilities.
- Root shell launchers contain cluster resource configurations and experiment defaults.

## Development

```bash
python -m ruff check .
pytest -q
pytest tests/test_interp_predictor.py -q
```

Tests cover stream validation, deterministic reconstruction, checkpoint
compatibility, dtype/rank preservation, and absolute error-bound enforcement.
