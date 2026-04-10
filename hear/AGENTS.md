# Repository Guidelines

## Project Structure & Module Organization
`openpi/` contains the main library: `models/`, `models_pytorch/`, `policies/`, `training/`, `serving/`, and shared utilities. Tests live beside the code as `*_test.py`. `packages/openpi-client/src/openpi_client/` is the separately packaged client library. Use `scripts/` for operational entrypoints such as `train.py`, `train_pytorch.py`, `serve_policy.py`, and `compute_norm_stats.py`. `examples/` holds robot- and benchmark-specific workflows, while `docs/` contains setup notes. Treat `assets/`, `mimi/`, and `checkpoints/` as artifact locations rather than general code. Follow the current `openpi/` tree even if older docs still mention `src/openpi`.

## Build, Test, and Development Commands
Use `uv` with Python 3.11 for repo work:

- `GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .` installs the workspace and editable package.
- `uv run ruff check .` runs lint checks.
- `uv run ruff format .` applies formatting.
- `uv run pre-commit run --all-files` runs the configured hooks, including lockfile checks when dependencies change.
- `uv run pytest openpi packages/openpi-client/src scripts` runs the unit tests against the current tree.
- `uv run scripts/compute_norm_stats.py --config-name pi05_libero` prepares dataset normalization stats before training.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite` starts JAX training.

## Coding Style & Naming Conventions
Use 4-space indentation, keep lines within 120 characters, and prefer clear type-annotated Python. Ruff enforces formatting and import order; keep imports simple and consistently sorted. Use `snake_case` for modules and functions, `CamelCase` for classes, and descriptive lowercase config IDs such as `pi05_libero`.

## Testing Guidelines
Pytest is the standard test runner. Add or update tests for every behavioral change in `openpi/`, `scripts/`, or `packages/openpi-client/`. Keep tests adjacent to the code and name them `*_test.py`, for example `openpi/policies/policy_test.py`. During iteration, run focused commands such as `uv run pytest openpi/policies/policy_test.py -k droid` before the full suite.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects, sometimes with a prefix such as `fix:` or `docs:`. Keep commits narrowly scoped. PRs should include a clear summary, linked issue or discussion when relevant, the exact validation commands you ran, and any GPU, dataset, or checkpoint assumptions. Include screenshots only for UI-facing changes such as updates to `index.html`.

## Configuration & Artifacts
Do not commit downloaded checkpoints, cached weights, or large generated assets. Use `OPENPI_DATA_HOME` when you need to relocate model downloads, and keep external checkpoint paths in config or documentation rather than hardcoding local machine paths.
