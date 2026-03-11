"""
T1/pipeline/config.py — Configuration parsing and validation.

Reads a ``key=value`` config file and returns a fully-resolved dict with
typed values and sensible defaults.  Every default applied is recorded in
``cfg["_assumptions"]`` so it can be logged and stored in JSON run logs.

Public API
----------
parse_file(path) -> Dict[str, str]   – raw string key/value pairs
build(raw)       -> Dict[str, Any]   – typed, validated config
"""

import os
from typing import Any, Dict, List


# ── Private helpers ────────────────────────────────────────────────────────────

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
    """Parse ``'L:200,C:200,density:0.1'`` into ``{'L':'200', ...}``."""
    specs: Dict[str, str] = {}
    for part in _parse_list(value):
        if ":" in part:
            k, _, v = part.partition(":")
            specs[k.strip()] = v.strip()
    return specs


# ── Public API ─────────────────────────────────────────────────────────────────

def parse_file(path: str) -> Dict[str, str]:
    """
    Parse a ``key=value`` config file into a raw string dict.

    Rules
    -----
    - Lines starting with ``#`` are ignored.
    - Lines without ``=`` are ignored.
    - Values are left as strings; use :func:`build` for typed values.

    Returns an empty dict (with a warning) when *path* does not exist.
    """
    config: Dict[str, str] = {}
    if not os.path.exists(path):
        import logging
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


def build(raw: Dict[str, str]) -> Dict[str, Any]:
    """
    Build a fully-resolved, typed config dict from raw key=value strings.

    Defaults are applied when keys are absent and every applied default is
    appended to ``cfg["_assumptions"]`` for logging and audit.

    Parameters
    ----------
    raw : dict[str, str]
        Output of :func:`parse_file` (or an empty dict for all-defaults).

    Returns
    -------
    dict[str, Any]  with the following keys:

    instances_dir, instances, synthetic, L, C, density,
    repetitions, gammas, solvers, heuristics, heuristic_solver,
    timeout_exact, timeout_heuristic, output_dir,
    parallel_jobs, dry_run, quick_check, _assumptions
    """
    assumptions: List[str] = []
    cfg: Dict[str, Any] = {}

    # ── Data source ────────────────────────────────────────────────────────
    if "instances_dir" not in raw:
        assumptions.append("instances_dir not set → defaulted to 'Mat'")
    cfg["instances_dir"] = raw.get("instances_dir", "Mat")

    cfg["instances"] = _parse_list(raw.get("instances", ""))

    # ── Synthetic flag ─────────────────────────────────────────────────────
    cfg["synthetic"] = _parse_bool(raw.get("synthetic", "false"))

    # ── Synthetic matrix specs ─────────────────────────────────────────────
    specs_raw = raw.get("synthetic_specs")
    if specs_raw is None:
        assumptions.append("synthetic_specs not set → defaulted to L=50,C=50,density=0.35")
        specs_raw = "L:50,C:50,density:0.35"
    specs = _parse_synthetic_specs(specs_raw)
    cfg["L"] = int(specs.get("L", 50))
    cfg["C"] = int(specs.get("C", 50))
    cfg["density"] = float(specs.get("density", 0.35))

    # ── Repetitions ────────────────────────────────────────────────────────
    # Number of independent runs per (instance, gamma) pair.  Each run gets a
    # fresh randomly-generated seed recorded in the CSV/log for reproducibility.
    if "repetitions" not in raw:
        assumptions.append("repetitions not set → defaulted to 5")
    cfg["repetitions"] = max(1, int(raw.get("repetitions", 5)))

    # ── Gammas ─────────────────────────────────────────────────────────────
    gammas_raw = raw.get("gammas")
    if gammas_raw is None:
        assumptions.append("gammas not set → defaulted to [0.9, 0.95, 0.99, 1.0]")
        gammas_raw = "0.9,0.95,0.99,1.0"
    cfg["gammas"] = [float(g) for g in _parse_list(gammas_raw)]
    if not cfg["gammas"]:
        cfg["gammas"] = [0.95]
        assumptions.append("gammas list empty → defaulted to [0.95]")

    # ── Solvers / heuristics ───────────────────────────────────────────────
    cfg["solvers"] = _parse_list(raw.get("solvers", ""))
    cfg["heuristics"] = _parse_list(raw.get("heuristics", ""))

    # ── Heuristic solver ───────────────────────────────────────────────────
    # Which solver class to inject as ``model_class`` inside heuristics.
    # 'ALL' (case-insensitive) → every configured solver is used for each
    # heuristic run, producing one row per (heuristic, solver) pair.
    # A specific class name → only that solver is used; it must appear in
    # ``solvers``.  Falls back to ALL with a warning if not found.
    cfg["heuristic_solver"] = raw.get("heuristic_solver", "ALL").strip()

    # ── Timeouts ───────────────────────────────────────────────────────────
    cfg["timeout_exact"] = int(raw.get("timeout_exact", 600))
    cfg["timeout_heuristic"] = int(raw.get("timeout_heuristic", 150))

    # ── Output / parallelism / flags ───────────────────────────────────────
    cfg["output_dir"] = raw.get("output_dir", "T1/results")
    cfg["parallel_jobs"] = int(raw.get("parallel_jobs", 1))
    cfg["dry_run"] = _parse_bool(raw.get("dry_run", "false"))
    cfg["quick_check"] = _parse_bool(raw.get("quick_check", "false"))

    cfg["_assumptions"] = assumptions
    return cfg
