#!/usr/bin/env python3
"""
T1/run_experiment.py – Reproducible experiment pipeline for biclustering.

Usage
-----
    python T1/run_experiment.py [config_file]
    python T1/run_experiment.py --dry-run [config_file]
    python T1/run_experiment.py --quick-check [config_file]

If config_file is not provided, defaults to T1/config.arg.
"""

import argparse
import csv
import inspect
import importlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Root path setup ────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── CSV Header ─────────────────────────────────────────────────────────────────
CSV_HEADER = [
    "instance_id", "m", "n", "base_dens", "gamma", "solver",
    "seed", "heuristic", "time", "status", "objective", "area", "density", "gap",
]

# ── Gurobi status codes ────────────────────────────────────────────────────────
_GUROBI_STATUS = {
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


def normalize_status(raw_status: Any) -> str:
    """Map raw solver status to a readable string."""
    if isinstance(raw_status, bool):
        return "optimal" if raw_status else "error"
    if isinstance(raw_status, int):
        if raw_status == 2:
            return "optimal"
        if raw_status == 9:
            return "time_limit"
        return _GUROBI_STATUS.get(raw_status, f"unknown_{raw_status}")
    return str(raw_status)


# ── Environment helpers ────────────────────────────────────────────────────────

def get_git_hash() -> str:
    """Return short git hash or 'no-git'."""
    try:
        result = subprocess.run(
            ["git", "-C", _ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "no-git"


def get_env_info() -> Dict:
    """Collect git hash, python version, and pip freeze."""
    env: Dict[str, Any] = {
        "git_hash": get_git_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
        env["pip_freeze"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        env["pip_freeze"] = "unavailable"
    return env


# ── Config parsing ─────────────────────────────────────────────────────────────

def parse_config(path: str) -> Dict[str, str]:
    """Parse key=value config file. Returns raw string dict."""
    config: Dict[str, str] = {}
    if not os.path.exists(path):
        logging.warning("Config file not found: %s — using defaults.", path)
        return config
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    return config


def _parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _parse_list(value: Any, sep: str = ",") -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(sep) if v.strip()]


def _parse_synthetic_specs(value: str) -> Dict[str, str]:
    """Parse 'L:200,C:200,density:0.1' into a dict."""
    specs: Dict[str, str] = {}
    for part in _parse_list(value):
        if ":" in part:
            k, _, v = part.partition(":")
            specs[k.strip()] = v.strip()
    return specs


def build_config(raw: Dict[str, str]) -> Dict[str, Any]:
    """Build fully-resolved config from raw key=value dict with defaults."""
    assumptions: List[str] = []
    cfg: Dict[str, Any] = {}

    # instances_dir
    cfg["instances_dir"] = raw.get("instances_dir", "Mat")
    if "instances_dir" not in raw:
        assumptions.append("instances_dir not set → defaulted to 'Mat'")

    # instances (optional, override instances_dir sweep)
    cfg["instances"] = _parse_list(raw.get("instances", ""))

    # synthetic flag
    cfg["synthetic"] = _parse_bool(raw.get("synthetic", "false"))

    # synthetic_specs
    specs_raw = raw.get("synthetic_specs", "L:50,C:50,density:0.35")
    specs = _parse_synthetic_specs(specs_raw)
    cfg["L"] = int(specs.get("L", 50))
    cfg["C"] = int(specs.get("C", 50))
    cfg["density"] = float(specs.get("density", 0.35))
    if "synthetic_specs" not in raw:
        assumptions.append(
            f"synthetic_specs not set → defaulted to L={cfg['L']},C={cfg['C']},density={cfg['density']}"
        )

    # seeds
    seeds_raw = raw.get("seeds", "42")
    cfg["seeds"] = [int(s) for s in _parse_list(seeds_raw)]
    if not cfg["seeds"]:
        cfg["seeds"] = [42]
        assumptions.append("seeds not set → defaulted to [42]")

    # gammas
    gammas_raw = raw.get("gammas", "0.9,0.95,0.99,1.0")
    cfg["gammas"] = [float(g) for g in _parse_list(gammas_raw)]
    if not cfg["gammas"]:
        cfg["gammas"] = [0.95]
        assumptions.append("gammas not set → defaulted to [0.95]")

    # solvers and heuristics
    cfg["solvers"] = _parse_list(raw.get("solvers", ""))
    cfg["heuristics"] = _parse_list(raw.get("heuristics", ""))

    # timeouts
    cfg["timeout_exact"] = int(raw.get("timeout_exact", 600))
    cfg["timeout_heuristic"] = int(raw.get("timeout_heuristic", 150))

    # output_dir
    cfg["output_dir"] = raw.get("output_dir", "T1/results")

    # parallel_jobs
    cfg["parallel_jobs"] = int(raw.get("parallel_jobs", 1))

    # flags
    cfg["dry_run"] = _parse_bool(raw.get("dry_run", "false"))
    cfg["quick_check"] = _parse_bool(raw.get("quick_check", "false"))

    cfg["_assumptions"] = assumptions
    return cfg


# ── Solver / heuristic discovery ───────────────────────────────────────────────

def discover_solvers() -> Dict[str, Any]:
    """
    Scan model/final/ for BiclusterModelBase subclasses.

    Returns a dict keyed by:
      - ClassName
      - module_filename:ClassName
      - model.final.module_filename:ClassName
    """
    try:
        from model.base import BiclusterModelBase
    except ImportError as exc:
        logging.error("Cannot import model.base.BiclusterModelBase: %s", exc)
        return {}

    final_dir = os.path.join(_ROOT, "model", "final")
    solvers: Dict[str, Any] = {}

    if not os.path.isdir(final_dir):
        logging.warning("model/final directory not found at %s", final_dir)
        return solvers

    for fname in sorted(os.listdir(final_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        module_stem = fname[:-3]
        module_name = f"model.final.{module_stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            logging.warning("Cannot import %s: %s", module_name, exc)
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, BiclusterModelBase)
                and obj is not BiclusterModelBase
                and obj.__module__ == module_name
            ):
                solvers[name] = obj
                solvers[f"{module_stem}:{name}"] = obj
                solvers[f"{module_name}:{name}"] = obj

    return solvers


def discover_heuristics() -> Dict[str, Any]:
    """
    Scan model/heuristics/ for callable functions.

    Returns a dict keyed by:
      - func_name
      - module_filename:func_name
      - model.heuristics.module_filename:func_name
    """
    heuristics_dir = os.path.join(_ROOT, "model", "heuristics")
    heuristics: Dict[str, Any] = {}

    if not os.path.isdir(heuristics_dir):
        logging.warning("model/heuristics directory not found at %s", heuristics_dir)
        return heuristics

    for fname in sorted(os.listdir(heuristics_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        module_stem = fname[:-3]
        module_name = f"model.heuristics.{module_stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            logging.warning("Cannot import %s: %s", module_name, exc)
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if obj.__module__ == module_name:
                heuristics[name] = obj
                heuristics[f"{module_stem}:{name}"] = obj
                heuristics[f"{module_name}:{name}"] = obj

    return heuristics


def resolve_solver(name: str, all_solvers: Dict) -> Optional[Any]:
    """Resolve a solver class by name (exact, module:Class, or partial)."""
    if name in all_solvers:
        return all_solvers[name]
    for key, cls in all_solvers.items():
        if key.endswith(f":{name}") or key == name:
            return cls
    return None


def resolve_heuristic(name: str, all_heuristics: Dict) -> Optional[Any]:
    """Resolve a heuristic function by name (exact, module:func, or partial)."""
    if name in all_heuristics:
        return all_heuristics[name]
    for key, fn in all_heuristics.items():
        if key.endswith(f":{name}") or key == name:
            return fn
    return None


# ── Matrix utilities ───────────────────────────────────────────────────────────

def matrix_to_model_inputs(
    matrix: np.ndarray,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Convert numpy binary matrix to (rows_data, cols_data, edges)."""
    m, n = matrix.shape
    rows_data = [(i, int(matrix[i, :].sum())) for i in range(m)]
    cols_data = [(j, int(matrix[:, j].sum())) for j in range(n)]
    edges = [(i, j) for i in range(m) for j in range(n) if matrix[i, j] == 1]
    return rows_data, cols_data, edges


def load_csv_matrix(path: str) -> np.ndarray:
    """
    Load binary matrix from CSV.

    Auto-detects separator and header presence.
    Default assumption: comma separator, no header.
    """
    detected_sep = ","
    has_header = False

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()

    for sep in [",", ";", "\t", " "]:
        if sep in first_line:
            detected_sep = sep
            break

    # If the first line contains non-numeric tokens, treat it as a header
    try:
        [float(v.strip()) for v in first_line.split(detected_sep) if v.strip()]
    except ValueError:
        has_header = True

    if detected_sep != "," or has_header:
        logging.info(
            "CSV auto-detection for %s: sep=%r, header=%s", path, detected_sep, has_header
        )

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=detected_sep)
        if has_header:
            next(reader)
        for row in reader:
            if row:
                try:
                    rows.append([int(float(v.strip())) for v in row if v.strip()])
                except ValueError:
                    continue

    return np.array(rows, dtype=int) if rows else np.zeros((0, 0), dtype=int)


def compute_metrics(
    matrix: np.ndarray,
    row_indices: List[int],
    col_indices: List[int],
) -> Tuple[int, int, float]:
    """Return (objective, area, density) for selected submatrix."""
    if not row_indices or not col_indices:
        return 0, 0, 0.0
    sub = matrix[np.ix_(sorted(row_indices), sorted(col_indices))]
    objective = int(sub.sum())
    area = len(row_indices) * len(col_indices)
    density = objective / area if area > 0 else 0.0
    return objective, area, density


# ── JSON log writing ───────────────────────────────────────────────────────────

def write_json_log(log_dir: str, run_id: str, data: Dict) -> str:
    """Write a JSON log file for a single run. Returns the file path."""
    os.makedirs(log_dir, exist_ok=True)
    # Sanitize run_id for filesystem
    safe_id = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in run_id)
    path = os.path.join(log_dir, f"{safe_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


# ── CSV row appender (thread-safe) ─────────────────────────────────────────────
_csv_lock = threading.Lock()


def append_csv_row(csv_path: str, row: Dict) -> None:
    """Append one row to the results CSV (thread-safe)."""
    with _csv_lock:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writerow(row)


# ── Exact solver run ───────────────────────────────────────────────────────────

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
    env_info: Dict,
    assumptions: List[str],
) -> Dict:
    """Execute one exact solver run. Returns a CSV row dict."""
    m, n = matrix.shape
    run_id = (
        f"{instance_id}__{solver_name}__g{gamma}__s{seed}__{int(time.time() * 1000)}"
    )

    result: Dict[str, Any] = {
        "instance_id": instance_id,
        "m": m,
        "n": n,
        "base_dens": round(base_dens, 6),
        "gamma": gamma,
        "solver": solver_name,
        "seed": seed,
        "heuristic": "NA",
        "time": "NA",
        "status": "error",
        "objective": "NA",
        "area": "NA",
        "density": "NA",
        "gap": "NA",
    }

    log_data: Dict[str, Any] = {
        "run_id": run_id,
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
            log_data["setparam_timelimit_warning"] = str(exc)

        try:
            model.setParam("OutputFlag", 0)
        except Exception:
            pass

        model.optimize()
        elapsed = time.time() - t0

        raw_status = model.status
        log_data["raw_status"] = raw_status
        status_str = normalize_status(raw_status)

        selected_rows: List[int] = []
        selected_cols: List[int] = []
        try:
            selected_rows = model.get_selected_rows()
            selected_cols = model.get_selected_cols()
        except Exception as exc:
            logging.warning("get_selected_rows/cols failed for %s: %s", solver_name, exc)
            log_data["get_solution_warning"] = str(exc)

        obj_val_model = None
        try:
            obj_val_model = model.ObjVal
        except Exception:
            pass

        objective, area, density = compute_metrics(matrix, selected_rows, selected_cols)

        result.update({
            "time": round(elapsed, 4),
            "status": status_str,
            "objective": objective,
            "area": area,
            "density": round(density, 6),
        })

        log_data.update({
            "elapsed": elapsed,
            "selected_rows": selected_rows,
            "selected_cols": selected_cols,
            "model_ObjVal": obj_val_model,
        })

    except Exception:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        result["time"] = round(elapsed, 4)
        result["status"] = "error"
        log_data["traceback"] = tb
        logging.error("Error in exact solver %s: %s", solver_name, tb)

    log_data["csv_row"] = result
    write_json_log(log_dir, run_id, log_data)
    return result


# ── Heuristic run ──────────────────────────────────────────────────────────────

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
    env_info: Dict,
    assumptions: List[str],
) -> Dict:
    """Execute one heuristic run. Returns a CSV row dict."""
    m, n = matrix.shape
    run_id = (
        f"{instance_id}__{heuristic_name}__g{gamma}__s{seed}__{int(time.time() * 1000)}"
    )

    result: Dict[str, Any] = {
        "instance_id": instance_id,
        "m": m,
        "n": n,
        "base_dens": round(base_dens, 6),
        "gamma": gamma,
        "solver": solver_name,
        "seed": seed,
        "heuristic": heuristic_name,
        "time": "NA",
        "status": "error",
        "objective": "NA",
        "area": "NA",
        "density": "NA",
        "gap": "NA",
    }

    log_data: Dict[str, Any] = {
        "run_id": run_id,
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

    # Fix seeds for reproducibility before calling the heuristic
    random.seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    try:
        sig = inspect.signature(heuristic_fn)

        _param_map = {
            "input_matrix": matrix,
            "model_class": solver_class,
            "error_rate": error_rate,
            "time_limit": timeout,
            "seed": seed,
        }

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
                        f"Heuristic {heuristic_name} requires unknown positional param '{pname}'"
                    )
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                if pname in _param_map:
                    kw_args[pname] = _param_map[pname]

        log_data["introspected_params"] = list(sig.parameters.keys())

        heuristic_result = heuristic_fn(*pos_args, **kw_args)
        elapsed = time.time() - t0

        # Parse result tuple
        if isinstance(heuristic_result, (tuple, list)) and len(heuristic_result) >= 2:
            selected_rows = list(heuristic_result[0]) if heuristic_result[0] else []
            selected_cols = list(heuristic_result[1]) if heuristic_result[1] else []
            raw_status = heuristic_result[2] if len(heuristic_result) > 2 else True
        else:
            selected_rows = []
            selected_cols = []
            raw_status = False

        status_str = normalize_status(raw_status)
        objective, area, density = compute_metrics(matrix, selected_rows, selected_cols)

        gap: Any = "NA"
        if best_known is not None and best_known > 0 and isinstance(objective, int):
            gap = round(100.0 * (best_known - objective) / best_known, 4)

        result.update({
            "time": round(elapsed, 4),
            "status": status_str,
            "objective": objective,
            "area": area,
            "density": round(density, 6),
            "gap": gap,
        })

        log_data.update({
            "elapsed": elapsed,
            "selected_rows": selected_rows,
            "selected_cols": selected_cols,
            "raw_status": raw_status,
        })

    except Exception:
        elapsed = time.time() - t0
        tb = traceback.format_exc()
        result["time"] = round(elapsed, 4)
        result["status"] = "error"
        log_data["traceback"] = tb
        logging.error("Error in heuristic %s: %s", heuristic_name, tb)

    log_data["csv_row"] = result
    write_json_log(log_dir, run_id, log_data)
    return result


# ── Instance discovery ─────────────────────────────────────────────────────────

def discover_instances(cfg: Dict) -> List[str]:
    """Return a sorted list of CSV file paths for real instances."""
    if cfg["instances"]:
        instances_dir = cfg["instances_dir"]
        if not os.path.isabs(instances_dir):
            instances_dir = os.path.join(_ROOT, instances_dir)
        return [
            os.path.join(instances_dir, f) if not os.path.isabs(f) else f
            for f in cfg["instances"]
        ]

    instances_dir = cfg["instances_dir"]
    if not os.path.isabs(instances_dir):
        instances_dir = os.path.join(_ROOT, instances_dir)

    if not os.path.isdir(instances_dir):
        logging.warning("instances_dir not found: %s", instances_dir)
        return []

    return sorted(
        os.path.join(instances_dir, f)
        for f in os.listdir(instances_dir)
        if f.endswith(".csv")
    )


# ── Run planning ───────────────────────────────────────────────────────────────

def _resolve_solvers_and_heuristics(
    cfg: Dict,
    all_solvers: Dict,
    all_heuristics: Dict,
) -> Tuple[Dict, Dict, List[str]]:
    """
    Resolve solver classes and heuristic functions from config names.
    Auto-selects first available solver if config is empty and synthetic=true.
    Returns (solver_classes, heuristic_fns, updated_assumptions).
    """
    assumptions = list(cfg.get("_assumptions", []))

    solver_names = list(cfg["solvers"])
    if not solver_names and cfg["synthetic"] and all_solvers:
        # ASSUMPTION: Pick the first class-name key (no ':') as the default solver
        for k in all_solvers:
            if ":" not in k:
                solver_names = [k]
                msg = f"No solvers configured; auto-selected first available: {k}"
                assumptions.append(msg)
                logging.warning("ASSUMPTION: %s", msg)
                break

    solver_classes: Dict[str, Any] = {}
    for s in solver_names:
        cls = resolve_solver(s, all_solvers)
        if cls is not None:
            solver_classes[s] = cls
        else:
            logging.warning("Solver '%s' not found in model/final — skipping.", s)

    heuristic_fns: Dict[str, Any] = {}
    for h in cfg["heuristics"]:
        fn = resolve_heuristic(h, all_heuristics)
        if fn is not None:
            heuristic_fns[h] = fn
        else:
            logging.warning("Heuristic '%s' not found in model/heuristics — skipping.", h)

    return solver_classes, heuristic_fns, assumptions


def plan_runs(cfg: Dict, all_solvers: Dict, all_heuristics: Dict) -> List[Dict]:
    """Build a flat list of planned run descriptions (for dry_run display)."""
    solver_classes, heuristic_fns, _ = _resolve_solvers_and_heuristics(
        cfg, all_solvers, all_heuristics
    )

    runs: List[Dict] = []

    if cfg["synthetic"]:
        for seed in cfg["seeds"]:
            iid = f"synthetic_L{cfg['L']}_C{cfg['C']}_d{cfg['density']}_s{seed}"
            for gamma in cfg["gammas"]:
                for sn in solver_classes:
                    runs.append({"type": "exact", "instance_id": iid, "seed": seed,
                                 "gamma": gamma, "solver_name": sn})
                for hn in heuristic_fns:
                    for sn in solver_classes:
                        runs.append({"type": "heuristic", "instance_id": iid, "seed": seed,
                                     "gamma": gamma, "heuristic_name": hn, "solver_name": sn})
    else:
        instances = discover_instances(cfg)
        for inst_path in instances:
            iid = os.path.splitext(os.path.basename(inst_path))[0]
            for gamma in cfg["gammas"]:
                for sn in solver_classes:
                    runs.append({"type": "exact", "instance_id": iid, "seed": "NA",
                                 "gamma": gamma, "solver_name": sn})
                for hn in heuristic_fns:
                    for seed in cfg["seeds"]:
                        for sn in solver_classes:
                            runs.append({"type": "heuristic", "instance_id": iid, "seed": seed,
                                         "gamma": gamma, "heuristic_name": hn, "solver_name": sn})

    return runs


# ── Pipeline execution ─────────────────────────────────────────────────────────

def _init_csv(csv_path: str) -> None:
    """Write CSV header."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()


def _execute_group(
    matrix: np.ndarray,
    instance_id: str,
    gamma: float,
    seed_for_synthetic: Any,
    base_dens: float,
    solver_classes: Dict,
    heuristic_fns: Dict,
    cfg: Dict,
    csv_path: str,
    log_dir: str,
    env_info: Dict,
    assumptions: List[str],
) -> None:
    """
    Execute all solvers then all heuristics for one (instance, gamma) group.
    Updates best_known from exact solvers before running heuristics.
    """
    error_rate = 1.0 - gamma
    best_known: Optional[float] = None

    # --- Exact solvers ---
    for solver_name, solver_cls in solver_classes.items():
        row = run_exact_solver(
            matrix=matrix,
            solver_name=solver_name,
            solver_class=solver_cls,
            error_rate=error_rate,
            timeout=cfg["timeout_exact"],
            instance_id=instance_id,
            gamma=gamma,
            seed=seed_for_synthetic,
            base_dens=base_dens,
            log_dir=log_dir,
            env_info=env_info,
            assumptions=assumptions,
        )
        append_csv_row(csv_path, row)

        obj = row.get("objective")
        if isinstance(obj, int) and row["status"] not in ("error",):
            if best_known is None or obj > best_known:
                best_known = float(obj)

    # --- Heuristics ---
    # For synthetic instances the seed is shared (one per matrix).
    # For real instances we iterate over cfg["seeds"].
    seeds_for_heuristics = (
        [seed_for_synthetic] if cfg["synthetic"] else cfg["seeds"]
    )

    for heuristic_name, heuristic_fn in heuristic_fns.items():
        for seed in seeds_for_heuristics:
            for solver_name, solver_cls in solver_classes.items():
                row = run_heuristic(
                    matrix=matrix,
                    heuristic_name=heuristic_name,
                    heuristic_fn=heuristic_fn,
                    solver_class=solver_cls,
                    solver_name=solver_name,
                    error_rate=error_rate,
                    timeout=cfg["timeout_heuristic"],
                    seed=int(seed),
                    instance_id=instance_id,
                    gamma=gamma,
                    base_dens=base_dens,
                    best_known=best_known,
                    log_dir=log_dir,
                    env_info=env_info,
                    assumptions=assumptions,
                )
                append_csv_row(csv_path, row)


def execute_pipeline(
    cfg: Dict,
    csv_path: str,
    log_dir: str,
    env_info: Dict,
) -> None:
    """Main pipeline: resolves solvers/heuristics and iterates over all groups."""
    all_solvers = discover_solvers()
    all_heuristics = discover_heuristics()

    logging.info("Discovered solvers: %s", [k for k in all_solvers if ":" not in k])
    logging.info("Discovered heuristics: %s", [k for k in all_heuristics if ":" not in k])

    solver_classes, heuristic_fns, assumptions = _resolve_solvers_and_heuristics(
        cfg, all_solvers, all_heuristics
    )

    if not solver_classes:
        logging.warning("No solver classes resolved — exact runs will be skipped.")
    if not heuristic_fns:
        logging.info("No heuristics configured — heuristic runs will be skipped.")

    _init_csv(csv_path)

    if cfg["synthetic"]:
        from utils.create_matrix_V2 import create_matrix

        groups = [
            (seed, gamma)
            for seed in cfg["seeds"]
            for gamma in cfg["gammas"]
        ]

        def _run_synthetic_group(seed_gamma: Tuple) -> None:
            seed, gamma = seed_gamma
            matrix_list = create_matrix(cfg["L"], cfg["C"], cfg["density"], seed)
            matrix = np.array(matrix_list, dtype=int)
            m, n = matrix.shape
            base_dens = float(matrix.mean())
            iid = f"synthetic_L{cfg['L']}_C{cfg['C']}_d{cfg['density']}_s{seed}"
            logging.info(
                "Synthetic matrix seed=%d gamma=%.3f: %dx%d dens=%.4f",
                seed, gamma, m, n, base_dens,
            )
            _execute_group(
                matrix=matrix,
                instance_id=iid,
                gamma=gamma,
                seed_for_synthetic=seed,
                base_dens=base_dens,
                solver_classes=solver_classes,
                heuristic_fns=heuristic_fns,
                cfg=cfg,
                csv_path=csv_path,
                log_dir=log_dir,
                env_info=env_info,
                assumptions=assumptions,
            )

        _parallel_execute(groups, _run_synthetic_group, cfg["parallel_jobs"])

    else:
        instances = discover_instances(cfg)
        if not instances:
            logging.warning("No instances found. Check instances_dir or instances config.")
            return

        def _run_real_group(inst_gamma: Tuple) -> None:
            inst_path, gamma = inst_gamma
            iid = os.path.splitext(os.path.basename(inst_path))[0]
            try:
                matrix = load_csv_matrix(inst_path)
            except Exception as exc:
                logging.error("Failed to load instance %s: %s", inst_path, exc)
                return
            if matrix.size == 0:
                logging.warning("Empty matrix for %s — skipping.", inst_path)
                return
            m, n = matrix.shape
            base_dens = float(matrix.mean())
            logging.info("Instance %s gamma=%.3f: %dx%d dens=%.4f", iid, gamma, m, n, base_dens)
            _execute_group(
                matrix=matrix,
                instance_id=iid,
                gamma=gamma,
                seed_for_synthetic=None,
                base_dens=base_dens,
                solver_classes=solver_classes,
                heuristic_fns=heuristic_fns,
                cfg=cfg,
                csv_path=csv_path,
                log_dir=log_dir,
                env_info=env_info,
                assumptions=assumptions,
            )

        groups = [(p, g) for p in instances for g in cfg["gammas"]]
        _parallel_execute(groups, _run_real_group, cfg["parallel_jobs"])


def _parallel_execute(items: List, fn: Any, n_jobs: int) -> None:
    """Execute fn(item) for each item, sequentially or in thread pool."""
    if n_jobs <= 1:
        for item in items:
            fn(item)
        return

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                logging.error(
                    "Unhandled error in parallel worker for %s:\n%s",
                    futures[future],
                    traceback.format_exc(),
                )


# ── Quick check ────────────────────────────────────────────────────────────────

def run_quick_check(cfg: Dict, csv_path: str, log_dir: str, env_info: Dict) -> None:
    """
    Run a minimal pipeline validation:
      - synthetic 5×5 matrix, density=0.35, seed=42, gamma=0.9
      - Uses first available solver and first available heuristic (if any)
    Raises RuntimeError on failure.
    """
    from utils.create_matrix_V2 import create_matrix

    logging.info("quick_check: minimal synthetic 5×5, density=0.35, seed=42, gamma=0.9")

    all_solvers = discover_solvers()
    all_heuristics = discover_heuristics()

    qcfg: Dict[str, Any] = {
        "L": 5, "C": 5, "density": 0.35,
        "seeds": [42],
        "gammas": [0.9],
        "synthetic": True,
        "timeout_exact": 60,
        "timeout_heuristic": 30,
        "parallel_jobs": 1,
        "_assumptions": list(cfg.get("_assumptions", [])),
    }

    # Resolve solvers
    solver_names = list(cfg.get("solvers", []))
    if not solver_names:
        for k in all_solvers:
            if ":" not in k:
                solver_names = [k]
                break

    if not solver_names:
        raise RuntimeError("quick_check FAILED: no solver available.")

    qcfg["solvers"] = solver_names
    qcfg["heuristics"] = list(cfg.get("heuristics", []))

    solver_classes, heuristic_fns, assumptions = _resolve_solvers_and_heuristics(
        qcfg, all_solvers, all_heuristics
    )

    if not solver_classes:
        raise RuntimeError("quick_check FAILED: could not resolve any solver class.")

    _init_csv(csv_path)

    matrix_list = create_matrix(5, 5, 0.35, 42)
    matrix = np.array(matrix_list, dtype=int)
    base_dens = float(matrix.mean())

    _execute_group(
        matrix=matrix,
        instance_id="quick_check_5x5",
        gamma=0.9,
        seed_for_synthetic=42,
        base_dens=base_dens,
        solver_classes=solver_classes,
        heuristic_fns=heuristic_fns,
        cfg=qcfg,
        csv_path=csv_path,
        log_dir=log_dir,
        env_info=env_info,
        assumptions=assumptions,
    )

    # Validate outputs
    if not os.path.exists(csv_path):
        raise RuntimeError("quick_check FAILED: CSV not produced.")

    with open(csv_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise RuntimeError(f"quick_check FAILED: CSV has only {len(lines)} line(s) (expected header + data).")

    log_files = (
        [f for f in os.listdir(log_dir) if f.endswith(".json")]
        if os.path.isdir(log_dir)
        else []
    )
    if not log_files:
        raise RuntimeError("quick_check FAILED: No JSON logs produced.")

    logging.info(
        "quick_check PASSED: %d CSV data row(s), %d JSON log(s).",
        len(lines) - 1,
        len(log_files),
    )
    print(
        f"quick_check PASSED: {len(lines) - 1} CSV row(s), {len(log_files)} JSON log(s). "
        f"CSV: {csv_path}"
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="T1 reproducible experiment pipeline for biclustering."
    )
    parser.add_argument(
        "config", nargs="?", default=None,
        help="Path to config.arg file (default: T1/config.arg)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    parser.add_argument("--quick-check", action="store_true", help="Run minimal validation pipeline.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Locate config file
    config_path = args.config
    if config_path is None:
        default_cfg = os.path.join(_THIS_DIR, "config.arg")
        if os.path.exists(default_cfg):
            config_path = default_cfg
        else:
            logging.warning("No config file found; using built-in defaults.")

    raw_config = parse_config(config_path) if config_path else {}
    cfg = build_config(raw_config)

    # CLI flags override config
    if args.dry_run:
        cfg["dry_run"] = True
    if args.quick_check:
        cfg["quick_check"] = True

    if cfg["_assumptions"]:
        logging.info("Active config assumptions: %s", cfg["_assumptions"])

    # Resolve output paths
    output_dir = cfg["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_ROOT, output_dir)
    timestamp = int(time.time())
    csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env_info = get_env_info()
    logging.info(
        "Environment — git: %s, python: %s",
        env_info["git_hash"],
        env_info["python_version"],
    )

    # --- quick_check mode ---
    if cfg["quick_check"]:
        try:
            run_quick_check(cfg, csv_path, log_dir, env_info)
        except Exception as exc:
            diag_path = os.path.join(output_dir, "quick_check_diagnostic.txt")
            with open(diag_path, "w", encoding="utf-8") as f:
                f.write(f"quick_check FAILED\n\n{traceback.format_exc()}")
            logging.error("quick_check FAILED: %s\nDiagnostic: %s", exc, diag_path)
            sys.exit(1)
        return

    # --- dry_run mode ---
    if cfg["dry_run"]:
        all_solvers = discover_solvers()
        all_heuristics = discover_heuristics()
        runs = plan_runs(cfg, all_solvers, all_heuristics)
        print(f"\nDRY RUN — {len(runs)} planned run(s):")
        for i, r in enumerate(runs):
            if r["type"] == "exact":
                print(
                    f"  [{i+1:>4}] EXACT   instance={r['instance_id']}  "
                    f"gamma={r['gamma']}  solver={r['solver_name']}  seed={r['seed']}"
                )
            else:
                print(
                    f"  [{i+1:>4}] HEUR    instance={r['instance_id']}  "
                    f"gamma={r['gamma']}  solver={r['solver_name']}  "
                    f"heuristic={r['heuristic_name']}  seed={r['seed']}"
                )
        return

    # --- full pipeline ---
    execute_pipeline(cfg, csv_path, log_dir, env_info)
    logging.info("Pipeline complete. Results CSV: %s", csv_path)
    print(f"Done. Results: {csv_path}")


if __name__ == "__main__":
    main()
