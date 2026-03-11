"""
T1/pipeline/runner.py — Orchestrate the full experiment pipeline.

Responsibilities
----------------
- Discover solvers and heuristics.
- For each (instance/matrix, gamma) group:
    1. Run exact solvers → collect best_known.
    2. Run heuristics → compute gap against best_known.
- Write one CSV row + one JSON log per run (via pipeline.io).
- Support parallel execution via a ThreadPoolExecutor.
- Provide run_quick_check() for pipeline validation.

Public API
----------
execute_pipeline(cfg, csv_path, log_dir, env_info) -> None
run_quick_check(cfg, csv_path, log_dir, env_info)  -> None  (raises on failure)
"""

import logging
import os
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Root path bootstrap ────────────────────────────────────────────────────────
_T1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.dirname(_T1_DIR)
for _p in (_ROOT, _T1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.discovery import discover_heuristics, discover_solvers, resolve_all  # noqa: E402
from pipeline.executor import run_exact_solver, run_heuristic  # noqa: E402
from pipeline.io import append_csv_row, init_csv  # noqa: E402
from pipeline.planner import discover_instances  # noqa: E402


# ── Seed generation ────────────────────────────────────────────────────────────

def _new_seed() -> int:
    """Return a fresh random seed in [0, 2**31 - 1] suitable for any RNG."""
    return random.randint(0, 2**31 - 1)


# ── Group execution ────────────────────────────────────────────────────────────

def _execute_group(
    matrix: np.ndarray,
    instance_id: str,
    gamma: float,
    run_seed: int,
    base_dens: float,
    solver_classes: Dict[str, Any],
    heuristic_fns: Dict[str, Any],
    cfg: Dict[str, Any],
    csv_path: str,
    log_dir: str,
    env_info: Dict[str, Any],
    assumptions: List[str],
) -> None:
    """
    Execute all exact solvers then all heuristics for one (instance, gamma, seed)
    group.

    Each call is completely independent:
      - ``error_rate = 1 − gamma`` is computed locally.
      - ``best_known`` starts at None and is updated only from this group's exact
        solvers, so every gamma value is solved in full isolation.
      - ``run_seed`` drives heuristic reproducibility and, for synthetic matrices,
        also identifies which matrix was used.

    Exact solvers run before heuristics so that ``best_known`` is available for
    the gap metric.
    """
    # Each gamma is independent: own error_rate, own best_known
    error_rate = 1.0 - gamma
    best_known: Optional[float] = None

    # ── Exact solvers ──────────────────────────────────────────────────────
    for solver_name, solver_cls in solver_classes.items():
        row = run_exact_solver(
            matrix=matrix,
            solver_name=solver_name,
            solver_class=solver_cls,
            error_rate=error_rate,
            timeout=cfg["timeout_exact"],
            instance_id=instance_id,
            gamma=gamma,
            seed=run_seed,
            base_dens=base_dens,
            log_dir=log_dir,
            env_info=env_info,
            assumptions=assumptions,
        )
        append_csv_row(csv_path, row)

        obj = row.get("objective")
        if isinstance(obj, int) and row.get("status") != "error":
            if best_known is None or obj > best_known:
                best_known = float(obj)

    # ── Heuristics ─────────────────────────────────────────────────────────
    # The run_seed is used for every heuristic in this group so results are
    # tied to a single recorded seed.
    for heuristic_name, heuristic_fn in heuristic_fns.items():
        for solver_name, solver_cls in solver_classes.items():
            row = run_heuristic(
                matrix=matrix,
                heuristic_name=heuristic_name,
                heuristic_fn=heuristic_fn,
                solver_class=solver_cls,
                solver_name=solver_name,
                error_rate=error_rate,
                timeout=cfg["timeout_heuristic"],
                seed=run_seed,
                instance_id=instance_id,
                gamma=gamma,
                base_dens=base_dens,
                best_known=best_known,
                log_dir=log_dir,
                env_info=env_info,
                assumptions=assumptions,
            )
            append_csv_row(csv_path, row)


# ── Parallelism helper ─────────────────────────────────────────────────────────

def _parallel_execute(items: List[Any], fn: Callable, n_jobs: int) -> None:
    """
    Apply *fn* to every item in *items*, sequentially or via a thread pool.

    Exceptions raised inside workers are logged but do not abort the
    remaining work.

    Parameters
    ----------
    items  : list of arguments to pass to fn
    fn     : callable accepting a single argument
    n_jobs : 1 → sequential;  >1 → ThreadPoolExecutor with that many threads
    """
    if n_jobs <= 1:
        for item in items:
            fn(item)
        return

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {pool.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                logging.error(
                    "Worker error for %s:\n%s",
                    futures[future],
                    traceback.format_exc(),
                )


# ── Full pipeline ──────────────────────────────────────────────────────────────

def execute_pipeline(
    cfg: Dict[str, Any],
    csv_path: str,
    log_dir: str,
    env_info: Dict[str, Any],
) -> None:
    """
    Discover solvers/heuristics, generate/load matrices, and run all experiments.

    One CSV row and one JSON log are written per run.  The function never
    raises — errors in individual runs are caught inside the executor.

    Parameters
    ----------
    cfg      : resolved config dict from ``pipeline.config.build``
    csv_path : path to the results CSV (created/overwritten here)
    log_dir  : directory for JSON run logs
    env_info : environment metadata from ``utils.env_info.collect``
    """
    all_solvers = discover_solvers(_ROOT)
    all_heuristics = discover_heuristics(_ROOT)

    logging.info("Solvers available:    %s", [k for k in all_solvers if ":" not in k])
    logging.info("Heuristics available: %s", [k for k in all_heuristics if ":" not in k])

    solver_classes, heuristic_fns, assumptions = resolve_all(
        cfg, all_solvers, all_heuristics
    )

    if not solver_classes:
        logging.warning("No solver classes resolved — exact runs will be skipped.")

    init_csv(csv_path)

    if cfg["synthetic"]:
        from utils.create_matrix_V2 import create_matrix  # noqa: E402

        # Generate one seed per repetition dynamically.
        # Each seed is recorded in the CSV/log for full reproducibility.
        rep_seeds = [_new_seed() for _ in range(cfg["repetitions"])]

        # Groups: each repetition × each gamma is independent
        groups: List[Tuple] = [
            (seed, gamma)
            for seed in rep_seeds
            for gamma in cfg["gammas"]
        ]

        def _run_synthetic(seed_gamma: Tuple) -> None:
            seed, gamma = seed_gamma
            matrix = np.array(
                create_matrix(cfg["L"], cfg["C"], cfg["density"], seed), dtype=int
            )
            iid = f"synthetic_L{cfg['L']}_C{cfg['C']}_d{cfg['density']}_s{seed}"
            logging.info(
                "Synthetic seed=%d gamma=%.3f: %dx%d dens=%.4f",
                seed, gamma, *matrix.shape, float(matrix.mean()),
            )
            _execute_group(
                matrix=matrix,
                instance_id=iid,
                gamma=gamma,
                run_seed=seed,
                base_dens=float(matrix.mean()),
                solver_classes=solver_classes,
                heuristic_fns=heuristic_fns,
                cfg=cfg,
                csv_path=csv_path,
                log_dir=log_dir,
                env_info=env_info,
                assumptions=assumptions,
            )

        _parallel_execute(groups, _run_synthetic, cfg["parallel_jobs"])

    else:
        from utils.matrix_io import load_csv_matrix  # noqa: E402

        instances = discover_instances(cfg, _ROOT)
        if not instances:
            logging.warning(
                "No instances found. Check 'instances_dir' / 'instances' in config."
            )
            return

        def _run_real(inst_gamma_seed: Tuple) -> None:
            inst_path, gamma, run_seed = inst_gamma_seed
            iid = os.path.splitext(os.path.basename(inst_path))[0]
            try:
                matrix = load_csv_matrix(inst_path)
            except Exception as exc:
                logging.error("Failed to load %s: %s", inst_path, exc)
                return
            if matrix.size == 0:
                logging.warning("Empty matrix for %s — skipped.", inst_path)
                return
            logging.info(
                "Instance %s gamma=%.3f seed=%d: %dx%d dens=%.4f",
                iid, gamma, run_seed, *matrix.shape, float(matrix.mean()),
            )
            _execute_group(
                matrix=matrix,
                instance_id=iid,
                gamma=gamma,
                run_seed=run_seed,
                base_dens=float(matrix.mean()),
                solver_classes=solver_classes,
                heuristic_fns=heuristic_fns,
                cfg=cfg,
                csv_path=csv_path,
                log_dir=log_dir,
                env_info=env_info,
                assumptions=assumptions,
            )

        # Generate one seed per repetition; pair with every (instance, gamma)
        rep_seeds = [_new_seed() for _ in range(cfg["repetitions"])]
        groups = [
            (p, g, s)
            for p in instances
            for g in cfg["gammas"]
            for s in rep_seeds
        ]
        _parallel_execute(groups, _run_real, cfg["parallel_jobs"])


# ── Quick check ────────────────────────────────────────────────────────────────

def run_quick_check(
    cfg: Dict[str, Any],
    csv_path: str,
    log_dir: str,
    env_info: Dict[str, Any],
) -> None:
    """
    Minimal pipeline validation: 5×5 synthetic matrix, density=0.35,
    seed=42, gamma=0.9.

    Validates that:
      1. The runner executes without an unhandled exception.
      2. A CSV with at least one data row is produced.
      3. At least one JSON log file is produced.

    Raises
    ------
    RuntimeError
        If no solver is available, or if the expected outputs are missing.
    """
    from utils.create_matrix_V2 import create_matrix  # noqa: E402

    logging.info("quick_check: 5×5 matrix, density=0.35, seed=42, gamma=0.9")

    all_solvers = discover_solvers(_ROOT)
    all_heuristics = discover_heuristics(_ROOT)

    qcfg: Dict[str, Any] = {
        "L": 5, "C": 5, "density": 0.35,
        "repetitions": 1,
        "gammas": [0.9],
        "synthetic": True,
        "timeout_exact": 60,
        "timeout_heuristic": 30,
        "parallel_jobs": 1,
        "solvers": list(cfg.get("solvers", [])),
        "heuristics": list(cfg.get("heuristics", [])),
        "_assumptions": list(cfg.get("_assumptions", [])),
    }

    solver_classes, heuristic_fns, assumptions = resolve_all(
        qcfg, all_solvers, all_heuristics
    )

    if not solver_classes:
        raise RuntimeError("quick_check FAILED: no solver class available.")

    init_csv(csv_path)

    matrix = np.array(create_matrix(5, 5, 0.35, 42), dtype=int)
    _execute_group(
        matrix=matrix,
        instance_id="quick_check_5x5",
        gamma=0.9,
        run_seed=42,
        base_dens=float(matrix.mean()),
        solver_classes=solver_classes,
        heuristic_fns=heuristic_fns,
        cfg=qcfg,
        csv_path=csv_path,
        log_dir=log_dir,
        env_info=env_info,
        assumptions=assumptions,
    )

    # ── Validate outputs ───────────────────────────────────────────────────
    if not os.path.exists(csv_path):
        raise RuntimeError("quick_check FAILED: CSV not produced.")

    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise RuntimeError(
            f"quick_check FAILED: CSV has {len(lines)} line(s); "
            "expected header + ≥1 data row."
        )

    log_files = (
        [fn for fn in os.listdir(log_dir) if fn.endswith(".json")]
        if os.path.isdir(log_dir)
        else []
    )
    if not log_files:
        raise RuntimeError("quick_check FAILED: No JSON logs produced.")

    logging.info(
        "quick_check PASSED: %d CSV row(s), %d JSON log(s).",
        len(lines) - 1,
        len(log_files),
    )
    print(
        f"quick_check PASSED: {len(lines) - 1} CSV row(s), "
        f"{len(log_files)} JSON log(s).  CSV: {csv_path}"
    )
