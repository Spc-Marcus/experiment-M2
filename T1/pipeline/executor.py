"""
T1/pipeline/executor.py — Execute individual solver and heuristic runs.

Each public function:
  1. Runs one solver / heuristic with the given parameters.
  2. Catches *all* exceptions so the pipeline never crashes on a bad run.
  3. Writes a detailed JSON log via ``pipeline.io.write_json_log``.
  4. Returns a CSV row dict ready for ``pipeline.io.append_csv_row``.

Gurobi status codes are mapped to readable strings; unknown codes are
returned as ``"unknown_<code>"``.

Public API
----------
normalize_status(raw) -> str
run_exact_solver(...)  -> dict
run_heuristic(...)     -> dict
"""

import inspect
import logging
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

# ── Root path bootstrap ────────────────────────────────────────────────────────
_T1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.dirname(_T1_DIR)
for _p in (_ROOT, _T1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.io import write_json_log  # noqa: E402
from pipeline.metrics import compute_gap, compute_metrics, matrix_to_model_inputs  # noqa: E402

# ── Gurobi status mapping ──────────────────────────────────────────────────────
_GUROBI_STATUS: Dict[int, str] = {
    1: "loaded",
    2: "optimal",
    3: "infeasible",
    4: "inf_or_unbd",
    5: "unbounded",
    6: "cutoff",
    7: "iteration_limit",
    8: "node_limit",
    9: "time_limit",
    10: "solution_limit",
    11: "interrupted",
    12: "numeric",
    13: "suboptimal",
    14: "inprogress",
    15: "user_obj_limit",
}


def normalize_status(raw: Any) -> str:
    """Map a raw solver status code to a human-readable string."""
    if isinstance(raw, bool):
        return "optimal" if raw else "error"
    if isinstance(raw, int):
        if raw == 2:
            return "optimal"
        if raw == 9:
            return "time_limit"
        return _GUROBI_STATUS.get(raw, f"unknown_{raw}")
    return str(raw)


# ── Internal helpers ───────────────────────────────────────────────────────────

def _base_result(
    instance_id: str,
    m: int,
    n: int,
    base_dens: float,
    gamma: float,
    solver: str,
    seed: Any,
    heuristic: str = "NA",
) -> Dict[str, Any]:
    """Return an error-state CSV row template."""
    return {
        "instance_id": instance_id,
        "m": m,
        "n": n,
        "base_dens": round(base_dens, 6),
        "gamma": gamma,
        "solver": solver,
        "seed": seed,
        "heuristic": heuristic,
        "time": "NA",
        "status": "error",
        "objective": "NA",
        "area": "NA",
        "density": "NA",
        "gap": "NA",
    }


# ── Exact solver ───────────────────────────────────────────────────────────────

def run_exact_solver(
    matrix: np.ndarray,
    solver_name: str,
    solver_class: Any,
    error_rate: float,
    timeout: int,
    instance_id: str,
    gamma: float,
    seed: Any,
    base_dens: float,
    log_dir: str,
    env_info: Dict[str, Any],
    assumptions: List[str],
) -> Dict[str, Any]:
    """
    Execute one exact solver run.

    Never raises — all exceptions are caught, written to the JSON log with
    ``status='error'``, and the CSV row is returned for recording.

    Parameters
    ----------
    matrix       : binary input matrix
    solver_name  : display name for the solver
    solver_class : BiclusterModelBase subclass to instantiate
    error_rate   : 1 − γ, passed to the model constructor
    timeout      : seconds passed via ``model.setParam('TimeLimit', ...)``
    instance_id  : label used in CSV and log filename
    gamma        : target density (stored for reference)
    seed         : synthetic seed (or ``'NA'`` for real instances)
    base_dens    : global density of the input matrix
    log_dir      : directory to write the JSON log
    env_info     : collected environment metadata
    assumptions  : list of auto-assumption strings for this session

    Returns
    -------
    dict  — CSV row compatible with ``pipeline.io.CSV_HEADER``
    """
    m, n = matrix.shape
    run_id = f"{instance_id}__{solver_name}__g{gamma}__s{seed}__{int(time.time() * 1000)}"
    result = _base_result(instance_id, m, n, base_dens, gamma, solver_name, seed)

    log: Dict[str, Any] = {
        "run_id": run_id,
        "type": "exact",
        "instance_id": instance_id,
        "solver": solver_name,
        "gamma": gamma,
        "error_rate": error_rate,
        "seed": seed,
        "timeout": timeout,
        "env": env_info,
        "assumptions": assumptions,
        "traceback": None,
        "raw_status": None,
        "import_path": getattr(solver_class, "__module__", "unknown"),
    }

    t0 = time.time()
    try:
        rows_data, cols_data, edges = matrix_to_model_inputs(matrix)
        model = solver_class(rows_data, cols_data, edges, error_rate)

        try:
            model.setParam("TimeLimit", timeout)
        except Exception as exc:
            logging.warning("setParam('TimeLimit') failed for %s: %s", solver_name, exc)
            log["setparam_timelimit_warning"] = str(exc)

        try:
            model.setParam("OutputFlag", 0)
        except Exception:
            pass

        model.optimize()
        elapsed = time.time() - t0

        raw_status = model.status
        log["raw_status"] = raw_status

        selected_rows: List[int] = []
        selected_cols: List[int] = []
        try:
            selected_rows = model.get_selected_rows()
            selected_cols = model.get_selected_cols()
        except Exception as exc:
            logging.warning("get_selected_rows/cols failed for %s: %s", solver_name, exc)
            log["get_solution_warning"] = str(exc)

        model_obj: Optional[float] = None
        try:
            model_obj = model.ObjVal
        except Exception:
            pass

        objective, area, density = compute_metrics(matrix, selected_rows, selected_cols)

        result.update({
            "time": round(elapsed, 4),
            "status": normalize_status(raw_status),
            "objective": objective,
            "area": area,
            "density": round(density, 6),
        })
        log.update({
            "elapsed": elapsed,
            "selected_rows": selected_rows,
            "selected_cols": selected_cols,
            "model_ObjVal": model_obj,
        })

    except Exception:
        result["time"] = round(time.time() - t0, 4)
        log["traceback"] = traceback.format_exc()
        logging.error(
            "Unhandled error in exact solver %s:\n%s", solver_name, log["traceback"]
        )

    log["csv_row"] = result
    write_json_log(log_dir, run_id, log)
    return result


# ── Heuristic ──────────────────────────────────────────────────────────────────

def run_heuristic(
    matrix: np.ndarray,
    heuristic_name: str,
    heuristic_fn: Any,
    solver_class: Any,
    solver_name: str,
    error_rate: float,
    timeout: int,
    seed: int,
    instance_id: str,
    gamma: float,
    base_dens: float,
    best_known: Optional[float],
    log_dir: str,
    env_info: Dict[str, Any],
    assumptions: List[str],
) -> Dict[str, Any]:
    """
    Execute one heuristic run with introspection-based parameter matching.

    The function signature of *heuristic_fn* is inspected via
    ``inspect.signature`` and only parameters present in our known set
    (``input_matrix``, ``model_class``, ``error_rate``, ``time_limit``,
    ``seed``) are forwarded.  Unknown *required* positional parameters
    cause a ``ValueError`` → the run is recorded with ``status='error'``.

    ``random.seed`` and ``numpy.random.seed`` are both set to *seed* before
    the call to guarantee reproducibility regardless of the heuristic's
    internal RNG.

    Never raises — all exceptions are caught, written to the JSON log.

    Returns
    -------
    dict  — CSV row compatible with ``pipeline.io.CSV_HEADER``
    """
    m, n = matrix.shape
    run_id = (
        f"{instance_id}__{heuristic_name}__g{gamma}__s{seed}__{int(time.time() * 1000)}"
    )
    result = _base_result(
        instance_id, m, n, base_dens, gamma, solver_name, seed, heuristic=heuristic_name
    )

    log: Dict[str, Any] = {
        "run_id": run_id,
        "type": "heuristic",
        "instance_id": instance_id,
        "heuristic": heuristic_name,
        "solver_used": solver_name,
        "gamma": gamma,
        "error_rate": error_rate,
        "seed": seed,
        "timeout": timeout,
        "best_known": best_known,
        "env": env_info,
        "assumptions": assumptions,
        "traceback": None,
        "import_path": getattr(heuristic_fn, "__module__", "unknown"),
    }

    # Fix random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    _param_map: Dict[str, Any] = {
        "input_matrix": matrix,
        "model_class": solver_class,
        "error_rate": error_rate,
        "time_limit": timeout,
        "seed": seed,
    }

    t0 = time.time()
    try:
        sig = inspect.signature(heuristic_fn)
        log["introspected_params"] = list(sig.parameters.keys())

        pos_args: List[Any] = []
        kw_args: Dict[str, Any] = {}

        for pname, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if pname in _param_map:
                    pos_args.append(_param_map[pname])
                elif param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Heuristic '{heuristic_name}' requires unknown positional "
                        f"parameter '{pname}' — cannot call."
                    )
            elif (
                param.kind == inspect.Parameter.KEYWORD_ONLY
                and pname in _param_map
            ):
                kw_args[pname] = _param_map[pname]

        heuristic_result = heuristic_fn(*pos_args, **kw_args)
        elapsed = time.time() - t0

        # Unpack (rows, cols[, status]) result
        if isinstance(heuristic_result, (tuple, list)) and len(heuristic_result) >= 2:
            selected_rows = list(heuristic_result[0] or [])
            selected_cols = list(heuristic_result[1] or [])
            raw_status = heuristic_result[2] if len(heuristic_result) > 2 else True
        else:
            selected_rows, selected_cols, raw_status = [], [], False

        objective, area, density = compute_metrics(matrix, selected_rows, selected_cols)
        gap = compute_gap(best_known, objective)

        result.update({
            "time": round(elapsed, 4),
            "status": normalize_status(raw_status),
            "objective": objective,
            "area": area,
            "density": round(density, 6),
            "gap": gap,
        })
        log.update({
            "elapsed": elapsed,
            "selected_rows": selected_rows,
            "selected_cols": selected_cols,
            "raw_status": raw_status,
        })

    except Exception:
        result["time"] = round(time.time() - t0, 4)
        log["traceback"] = traceback.format_exc()
        logging.error(
            "Unhandled error in heuristic %s:\n%s", heuristic_name, log["traceback"]
        )

    log["csv_row"] = result
    write_json_log(log_dir, run_id, log)
    return result
