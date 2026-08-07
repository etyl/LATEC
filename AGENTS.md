# Repository Guidelines

## Project Structure & Module Organization

Core library code lives in `latec/`. Compression orchestration is in `codec.py` and `gnn_codec.py`; predictors, quantization, entropy coding, level scheduling, and bitstream handling have dedicated modules. The command-line entry point is `latec/cli.py`. Keep reusable functionality in this package rather than in experiment scripts.

`tests/` contains pytest tests organized by component, such as `test_quantizer.py` and `test_gnn_codec.py`. `scripts/` holds training, evaluation, profiling, and plotting programs. Root-level `train*.sh`, `eval*.sh`, and `profile*.sh` files are Slurm-oriented launchers with cluster-specific defaults.

## Development Compatibility Policy

This codebase is under active development. Backward compatibility with previously generated compressed binaries or bitstreams is not required; prefer keeping the current format simple over retaining legacy decoding branches, migrations, or obsolete header fields unless explicitly requested.

Trained model and checkpoint versioning remains important. Explicitly version checkpoint formats, bump the version whenever the model architecture, stored schema, or interpretation of checkpoint state changes, and keep checkpoint writers and loaders synchronized. Reject incompatible checkpoint versions with a clear error, and update the relevant tests when the checkpoint format changes.

## Jean Zay Job Safety

Agents must not submit large or long-running jobs on Jean Zay. They may inspect launchers and prepare or validate `sbatch` commands, but the user must execute those commands directly. Editing, testing, or reviewing a launcher never grants permission to enqueue it. If the current machine has a GPU, agents may run small local tests.

## Build, Test, and Development Commands

- `pytest` runs the standard test suite from the repository root.
- `pytest tests/test_quantizer.py -q` runs one focused module while iterating.
- `pytest -m slow` runs checkpoint-backed end-to-end tests and may take minutes.
- `python -m ruff check .` runs the configured import, syntax, and undefined-name checks.
- `latec eval IMAGE --eb 2 --predictor interp` performs a quick local round trip without a trained checkpoint.
- `python scripts/train_gnn.py --help` and `python scripts/eval_predictors.py --help` document experiment-specific options. Review cluster paths before submitting the shell launchers.

## Coding Style & Naming Conventions

Follow the existing PEP 8-like style: `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for constants. Add type hints to public or non-obvious interfaces and short docstrings where behavior, shapes, or codec invariants need explanation. Keep encoder and decoder changes synchronized, especially scheduling, header fields, numeric precision, and reconstruction arithmetic. Run Ruff before submitting changes; keep imports grouped and lines readable.

## Testing Guidelines

Name files `test_<component>.py` and tests `test_<behavior>()`. Add regression tests for error-bound guarantees, deterministic decoding, malformed streams, and rank/dtype edge cases when relevant. Prefer small seeded arrays, private test doubles, and interpolation for fast tests; mark checkpoint- or hardware-dependent coverage as `slow`.
