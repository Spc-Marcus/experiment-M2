# T1 — Experiment Pipeline

> **Scope: experimentation only — no analysis, no visualisation.**
>
> This folder contains a reproducible pipeline that runs exact solvers and
> heuristics, collects raw metrics, and writes a CSV + per-run JSON logs.
> Analysis and plots are out of scope here (see `T1_test_plan.md` for the full
> analysis plan).

---

## Directory layout

```
T1/
├── run_experiment.py      # Main CLI runner
├── config.arg             # Working config (quick-test defaults)
├── config.arg.example     # Fully documented example config
├── README.md              # This file
└── results/               # Created at runtime
    ├── results_<ts>.csv   # One CSV per invocation (all runs)
    └── logs/
        └── *.json         # One JSON log per run
```

---

## Quick start

```bash
# From the repository root

# 1. Validate the pipeline with a minimal 5×5 matrix
python T1/run_experiment.py --quick-check

# 2. Preview planned runs without executing (dry run)
python T1/run_experiment.py --dry-run

# 3. Full run (reads T1/config.arg)
python T1/run_experiment.py

# 4. Specify a different config file
python T1/run_experiment.py path/to/my_config.arg

# 5. Override log level
python T1/run_experiment.py --log-level DEBUG
```

---

## Config format (`key=value`)

The config file uses plain `key=value` syntax.  Lines starting with `#` are
comments.  Lists are **comma-separated** values on a single line.

### Complete key reference

| Key | Default | Description |
|-----|---------|-------------|
| `instances_dir` | `Mat` | Directory of real CSV instances |
| `instances` | *(unset)* | Explicit list of CSV filenames inside `instances_dir`; if absent all `*.csv` files are used |
| `synthetic` | `false` | `true` → generate matrices; `false` → load from `instances_dir` |
| `synthetic_specs` | `L:50,C:50,density:0.35` | Dimensions and base density for synthetic matrices |
| `seeds` | `42` | Seeds for matrix generation (synthetic) and heuristic reproducibility |
| `gammas` | `0.9,0.95,0.99,1.0` | Target minimum sub-matrix densities; `error_rate = 1 − γ` |
| `solvers` | *(unset)* | Exact solver class names (see below) |
| `heuristics` | *(unset)* | Heuristic function names (see below) |
| `timeout_exact` | `600` | Time limit (s) for each exact solver run |
| `timeout_heuristic` | `150` | Time limit (s) for each heuristic run |
| `output_dir` | `T1/results` | Output directory (relative to repo root, or absolute) |
| `parallel_jobs` | `1` | `1` = sequential; `N` = N-thread pool (exact→heuristic order preserved within a group) |
| `dry_run` | `false` | Print planned runs, no execution |
| `quick_check` | `false` | Minimal 5×5 validation run |

### How to write `solvers`

The pipeline scans `model/final/` for all subclasses of `BiclusterModelBase`.

| Format | Example | Behaviour |
|--------|---------|-----------|
| `ClassName` | `MaxOneModel` | Searches all modules under `model/final` |
| `module:ClassName` | `max_one_final:MaxOneModel` | Targets a specific file |
| `model.final.module:ClassName` | `model.final.max_one_final:MaxOneModel` | Fully qualified |

Multiple solvers are comma-separated:

```
solvers=MaxOneModel,MaxSurfaceModel
```

### How to write `heuristics`

The pipeline scans `model/heuristics/` for all callable functions.

| Format | Example | Behaviour |
|--------|---------|-----------|
| `func_name` | `heuristicA` | Searches all modules under `model/heuristics` |
| `module:func_name` | `heuristicA:heuristicA` | Targets a specific file |
| `model.heuristics.module:func` | `model.heuristics.heuristicA:heuristicA` | Fully qualified |

---

## CSV output format

One CSV file per invocation is written to `T1/results/results_<timestamp>.csv`.

**Header:**

```
instance_id,m,n,base_dens,gamma,solver,seed,heuristic,time,status,objective,area,density,gap
```

| Column | Description |
|--------|-------------|
| `instance_id` | Filename stem (real) or `synthetic_L{L}_C{C}_d{density}_s{seed}` |
| `m`, `n` | Matrix dimensions |
| `base_dens` | Global density of the input matrix |
| `gamma` | Target minimum density (config value) |
| `solver` | Name of the solver class used |
| `seed` | Seed used (synthetic) or `NA` (real instance exact run) |
| `heuristic` | Heuristic function name, or `NA` for exact runs |
| `time` | Wall-clock time in seconds |
| `status` | `optimal` \| `time_limit` \| `error` \| Gurobi status string |
| `objective` | Number of 1s in the selected sub-matrix (computed from row/col indices) |
| `area` | `#rows_selected × #cols_selected` |
| `density` | `objective / area` (0 if area = 0) |
| `gap` | `100 × (best_known − objective) / best_known` (%) or `NA` |

`best_known` is the maximum `objective` observed across all exact solvers for
the same `(instance, gamma)` group.

---

## JSON log format

One file per run in `T1/results/logs/<run_id>.json`.  Content includes:

- `run_id`, `instance_id`, `solver`, `heuristic`, `gamma`, `error_rate`, `seed`
- `timeout`, `elapsed`
- `env`: `git_hash`, `python_version`, `platform`, `pip_freeze`
- `assumptions`: list of auto-assumption strings applied during this session
- `import_path`: Python module path of the solver/heuristic called
- `introspected_params` (heuristic only): list of parameter names detected
- `raw_status`: integer code returned by the solver
- `selected_rows`, `selected_cols`: index lists
- `model_ObjVal` (exact only): ObjVal reported by the model
- `traceback`: full traceback string on error, else `null`
- `csv_row`: the exact dict written to the CSV

---

## Reproducibility

- **Synthetic matrices** are generated with `utils/create_matrix_V2.create_matrix(L, C, density, seed)`.
  Only the `seed` is stored (not the matrix), so it can be regenerated exactly by calling the same function.
- **Heuristics** receive the configured `seed` and the runner also sets `random.seed(seed)` and
  `numpy.random.seed(seed)` before each heuristic call.

---

## Default assumptions (applied when config is incomplete)

These assumptions are logged at runtime and stored in every JSON log.

| Situation | Default behaviour |
|-----------|------------------|
| `instances_dir` absent | Use `Mat` |
| `solvers` empty and `synthetic=true` | Auto-select the first class found in `model/final` |
| `seeds` empty | Use `[42]` |
| `gammas` empty | Use `[0.95]` |
| `synthetic_specs` absent | `L=50, C=50, density=0.35` |
| Real CSV separator unknown | Auto-detect from first line; log detection decision |
| Real CSV has a header row | Auto-detect non-numeric first row; skip it; log decision |
| Heuristic signature differs | Introspect with `inspect.signature`; match parameters by name; skip unknown required params with `status=error` |
| Solver's `setParam` fails | Log warning and continue (solver handles its own timeout internally if possible) |
| `gurobipy` unavailable / no licence | Run is skipped; `status=error` with full message; pipeline continues |

---

## Reproducing `quick_check`

```bash
python T1/run_experiment.py --quick-check
```

This runs one exact solver + (if configured) one heuristic on a 5×5 synthetic
matrix (`density=0.35, seed=42, gamma=0.9`) and verifies that:

1. The runner exits without an unhandled exception.
2. A CSV with at least one data row is produced.
3. At least one JSON log file is produced.

If `quick_check` fails, a diagnostic file is written to
`T1/results/quick_check_diagnostic.txt` and the process exits with code 1.

---

## Dependencies

Standard library only (`csv`, `concurrent.futures`, `inspect`, `json`, …) plus
`numpy` (already used by `model/heuristics`).  No additional packages required
beyond what is already present in the repository environment.

A `requirements.txt` is not added because no new external dependencies are
introduced.
