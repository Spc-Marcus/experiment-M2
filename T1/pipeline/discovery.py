"""
T1/pipeline/discovery.py — Discover and resolve solvers and heuristics.

Scans ``model/final/`` for BiclusterModelBase subclasses and
``model/heuristics/`` for callable functions, returning registries keyed
by multiple name formats for flexible look-up.

Public API
----------
discover_solvers(root)    -> Dict[str, type]
discover_heuristics(root) -> Dict[str, callable]
resolve_solver(name, registry)    -> class or None
resolve_heuristic(name, registry) -> callable or None
resolve_all(cfg, all_solvers, all_heuristics) -> (solver_classes, heuristic_fns, assumptions)
"""

import importlib
import inspect
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── Root path bootstrap (works when module is imported standalone) ─────────────
_T1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.dirname(_T1_DIR)
for _p in (_ROOT, _T1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Discovery ──────────────────────────────────────────────────────────────────

def discover_solvers(root: str) -> Dict[str, Any]:
    """
    Scan ``<root>/model/final/`` for BiclusterModelBase subclasses.

    Each class is registered under three keys:
      - ``ClassName``
      - ``module_stem:ClassName``
      - ``model.final.module_stem:ClassName``

    Parameters
    ----------
    root : str
        Repository root directory.

    Returns
    -------
    dict  (empty when ``model/final`` is missing or unimportable)
    """
    try:
        from model.base import BiclusterModelBase
    except ImportError as exc:
        logging.error("Cannot import BiclusterModelBase: %s", exc)
        return {}

    final_dir = os.path.join(root, "model", "final")
    registry: Dict[str, Any] = {}

    if not os.path.isdir(final_dir):
        logging.warning("model/final not found at %s", final_dir)
        return registry

    for fname in sorted(os.listdir(final_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        stem = fname[:-3]
        module_name = f"model.final.{stem}"
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
                registry[name] = obj
                registry[f"{stem}:{name}"] = obj
                registry[f"{module_name}:{name}"] = obj

    return registry


def discover_heuristics(root: str) -> Dict[str, Any]:
    """
    Scan ``<root>/model/heuristics/`` for callable functions.

    Each function is registered under three keys:
      - ``func_name``
      - ``module_stem:func_name``
      - ``model.heuristics.module_stem:func_name``

    Parameters
    ----------
    root : str
        Repository root directory.

    Returns
    -------
    dict  (empty when ``model/heuristics`` is missing or unimportable)
    """
    heuristics_dir = os.path.join(root, "model", "heuristics")
    registry: Dict[str, Any] = {}

    if not os.path.isdir(heuristics_dir):
        logging.warning("model/heuristics not found at %s", heuristics_dir)
        return registry

    for fname in sorted(os.listdir(heuristics_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        stem = fname[:-3]
        module_name = f"model.heuristics.{stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            logging.warning("Cannot import %s: %s", module_name, exc)
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if obj.__module__ == module_name:
                registry[name] = obj
                registry[f"{stem}:{name}"] = obj
                registry[f"{module_name}:{name}"] = obj

    return registry


# ── Resolution helpers ─────────────────────────────────────────────────────────

def resolve_solver(name: str, registry: Dict[str, Any]) -> Optional[Any]:
    """
    Look up a solver class by name, with partial / suffix matching.

    Tries exact key first, then any key ending in ``:<name>``.
    """
    if name in registry:
        return registry[name]
    for key, cls in registry.items():
        if key.endswith(f":{name}"):
            return cls
    return None


def resolve_heuristic(name: str, registry: Dict[str, Any]) -> Optional[Any]:
    """
    Look up a heuristic function by name, with partial / suffix matching.

    Tries exact key first, then any key ending in ``:<name>``.
    """
    if name in registry:
        return registry[name]
    for key, fn in registry.items():
        if key.endswith(f":{name}"):
            return fn
    return None


def resolve_all(
    cfg: Dict[str, Any],
    all_solvers: Dict[str, Any],
    all_heuristics: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Resolve configured solver and heuristic names to their callables.

    Fallback
    --------
    When ``solvers`` is empty and ``synthetic=true``, the first class-only
    key (no ``:``) in *all_solvers* is auto-selected and the decision is
    appended to *assumptions* and logged as a WARNING.

    Returns
    -------
    solver_classes : dict[name -> class]
    heuristic_fns  : dict[name -> callable]
    assumptions    : updated list of assumption strings
    """
    assumptions: List[str] = list(cfg.get("_assumptions", []))
    solver_names: List[str] = list(cfg.get("solvers", []))

    # Auto-fallback: pick first class-only key
    if not solver_names and cfg.get("synthetic"):
        for k in all_solvers:
            if ":" not in k:
                solver_names = [k]
                msg = f"No solvers configured; auto-selected first available: {k}"
                assumptions.append(msg)
                logging.warning("ASSUMPTION: %s", msg)
                break

    solver_classes: Dict[str, Any] = {}
    for name in solver_names:
        cls = resolve_solver(name, all_solvers)
        if cls is not None:
            solver_classes[name] = cls
        else:
            logging.warning("Solver '%s' not found in model/final — skipped.", name)

    heuristic_fns: Dict[str, Any] = {}
    for name in cfg.get("heuristics", []):
        fn = resolve_heuristic(name, all_heuristics)
        if fn is not None:
            heuristic_fns[name] = fn
        else:
            logging.warning("Heuristic '%s' not found in model/heuristics — skipped.", name)

    return solver_classes, heuristic_fns, assumptions
