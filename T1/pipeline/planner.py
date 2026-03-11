"""
T1/pipeline/planner.py — Build the flat run plan (dry-run support).

The plan is a list of lightweight dicts describing every run that *would*
be executed, without actually running anything.  Used both by dry_run mode
(printed to stdout) and by the runner for iteration.

Public API
----------
discover_instances(cfg, root) -> list[str]   — CSV file paths
plan_runs(cfg, all_solvers, all_heuristics, root) -> list[dict]
print_plan(runs)                             — pretty-print to stdout
"""

import logging
import os
import sys
from typing import Any, Dict, List

# ── Root path bootstrap ────────────────────────────────────────────────────────
_T1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.dirname(_T1_DIR)
for _p in (_ROOT, _T1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.discovery import resolve_all  # noqa: E402


def discover_instances(cfg: Dict[str, Any], root: str) -> List[str]:
    """
    Return a sorted list of CSV file paths for real (non-synthetic) instances.

    Resolution order
    ----------------
    1. If ``cfg["instances"]`` is set, those filenames are used (relative to
       ``instances_dir`` unless already absolute).
    2. Otherwise, all ``*.csv`` files inside ``instances_dir`` are returned.

    Parameters
    ----------
    cfg  : resolved config dict from ``pipeline.config.build``
    root : repository root directory
    """
    instances_dir: str = cfg["instances_dir"]
    if not os.path.isabs(instances_dir):
        instances_dir = os.path.join(root, instances_dir)

    if cfg["instances"]:
        return [
            os.path.join(instances_dir, f) if not os.path.isabs(f) else f
            for f in cfg["instances"]
        ]

    if not os.path.isdir(instances_dir):
        logging.warning("instances_dir not found: %s", instances_dir)
        return []

    return sorted(
        os.path.join(instances_dir, fname)
        for fname in os.listdir(instances_dir)
        if fname.endswith(".csv")
    )


def plan_runs(
    cfg: Dict[str, Any],
    all_solvers: Dict[str, Any],
    all_heuristics: Dict[str, Any],
    root: str,
) -> List[Dict[str, Any]]:
    """
    Build the complete flat list of run-description dicts.

    Each dict contains at minimum:
      ``type``, ``instance_id``, ``gamma``, ``solver_name``, ``seed``

    Heuristic runs additionally contain ``heuristic_name``.

    This function does **not** execute anything.

    Parameters
    ----------
    cfg            : resolved config dict
    all_solvers    : registry from ``discovery.discover_solvers``
    all_heuristics : registry from ``discovery.discover_heuristics``
    root           : repository root directory

    Returns
    -------
    list[dict]
    """
    solver_classes, heuristic_fns, _ = resolve_all(cfg, all_solvers, all_heuristics)

    # Determine which solvers to use for heuristic plan entries
    _hs = cfg.get("heuristic_solver", "ALL")
    if _hs.upper() == "ALL":
        heuristic_solver_classes = solver_classes
    else:
        heuristic_solver_classes = {n: c for n, c in solver_classes.items() if n == _hs}
        if not heuristic_solver_classes:
            heuristic_solver_classes = solver_classes

    runs: List[Dict[str, Any]] = []

    if cfg["synthetic"]:
        # In dry-run we show repetitions as numbered slots; actual seeds are
        # assigned dynamically at execution time.
        for rep in range(cfg["repetitions"]):
            iid = f"synthetic_L{cfg['L']}_C{cfg['C']}_d{cfg['density']}_rep{rep + 1}"
            for gamma in cfg["gammas"]:
                for sn in solver_classes:
                    runs.append({
                        "type": "exact",
                        "instance_id": iid,
                        "seed": "<dynamic>",
                        "gamma": gamma,
                        "solver_name": sn,
                    })
                for hn in heuristic_fns:
                    for sn in heuristic_solver_classes:
                        runs.append({
                            "type": "heuristic",
                            "instance_id": iid,
                            "seed": "<dynamic>",
                            "gamma": gamma,
                            "heuristic_name": hn,
                            "solver_name": sn,
                        })
    else:
        for inst_path in discover_instances(cfg, root):
            iid = os.path.splitext(os.path.basename(inst_path))[0]
            for gamma in cfg["gammas"]:
                for sn in solver_classes:
                    runs.append({
                        "type": "exact",
                        "instance_id": iid,
                        "seed": "<dynamic>",
                        "gamma": gamma,
                        "solver_name": sn,
                    })
                for hn in heuristic_fns:
                    for rep in range(cfg["repetitions"]):
                        for sn in heuristic_solver_classes:
                            runs.append({
                                "type": "heuristic",
                                "instance_id": iid,
                                "seed": "<dynamic>",
                                "gamma": gamma,
                                "heuristic_name": hn,
                                "solver_name": sn,
                            })

    return runs


def print_plan(runs: List[Dict[str, Any]]) -> None:
    """Pretty-print the dry-run plan to stdout."""
    print(f"\nDRY RUN — {len(runs)} planned run(s):\n")
    for i, r in enumerate(runs, 1):
        if r["type"] == "exact":
            print(
                f"  [{i:>4}] EXACT   "
                f"instance={r['instance_id']}  "
                f"gamma={r['gamma']}  "
                f"solver={r['solver_name']}  "
                f"seed={r['seed']}"
            )
        else:
            print(
                f"  [{i:>4}] HEUR    "
                f"instance={r['instance_id']}  "
                f"gamma={r['gamma']}  "
                f"solver={r['solver_name']}  "
                f"heuristic={r['heuristic_name']}  "
                f"seed={r['seed']}"
            )
