"""
T1/pipeline/discovery.py — Découverte et résolution des solveurs et heuristiques.

Parcourt ``model/final/`` pour les sous-classes de BiclusterModelBase et
``model/heuristics/`` pour les fonctions appelables, en retournant des
registres indexés par plusieurs formats de noms pour une recherche flexible.

API publique
------------
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

# ── Bootstrap du chemin racine (fonctionne en import autonome) ─────────────────
_T1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.dirname(_T1_DIR)
for _p in (_ROOT, _T1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Découverte ─────────────────────────────────────────────────────────────────

def discover_solvers(root: str) -> Dict[str, Any]:
    """
    Parcourt ``<root>/model/final/`` pour les sous-classes de BiclusterModelBase.

    Chaque classe est enregistrée sous trois clés :
      - ``NomClasse``
      - ``nom_module:NomClasse``
      - ``model.final.nom_module:NomClasse``

    Paramètres
    ----------
    root : str
        Répertoire racine du dépôt.

    Retourne
    --------
    dict  (vide si ``model/final`` est absent ou non importable)
    """
    try:
        from model.base import BiclusterModelBase
    except ImportError as exc:
        logging.error("Impossible d'importer BiclusterModelBase : %s", exc)
        return {}

    final_dir = os.path.join(root, "model", "final")
    registry: Dict[str, Any] = {}

    if not os.path.isdir(final_dir):
        logging.warning("model/final introuvable : %s", final_dir)
        return registry

    for fname in sorted(os.listdir(final_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        stem = fname[:-3]
        module_name = f"model.final.{stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            logging.warning("Impossible d'importer %s : %s", module_name, exc)
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
    Parcourt ``<root>/model/heuristics/`` pour les fonctions appelables.

    Chaque fonction est enregistrée sous trois clés :
      - ``nom_fonction``
      - ``nom_module:nom_fonction``
      - ``model.heuristics.nom_module:nom_fonction``

    Paramètres
    ----------
    root : str
        Répertoire racine du dépôt.

    Retourne
    --------
    dict  (vide si ``model/heuristics`` est absent ou non importable)
    """
    heuristics_dir = os.path.join(root, "model", "heuristics")
    registry: Dict[str, Any] = {}

    if not os.path.isdir(heuristics_dir):
        logging.warning("model/heuristics introuvable : %s", heuristics_dir)
        return registry

    for fname in sorted(os.listdir(heuristics_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        stem = fname[:-3]
        module_name = f"model.heuristics.{stem}"
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            logging.warning("Impossible d'importer %s : %s", module_name, exc)
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if obj.__module__ == module_name:
                registry[name] = obj
                registry[f"{stem}:{name}"] = obj
                registry[f"{module_name}:{name}"] = obj

    return registry


# ── Fonctions de résolution ────────────────────────────────────────────────────

def resolve_solver(name: str, registry: Dict[str, Any]) -> Optional[Any]:
    """
    Recherche une classe de solveur par nom, avec correspondance partielle / suffixe.

    Essaie d'abord la clé exacte, puis toute clé se terminant par ``:<name>``.
    """
    if name in registry:
        return registry[name]
    for key, cls in registry.items():
        if key.endswith(f":{name}"):
            return cls
    return None


def resolve_heuristic(name: str, registry: Dict[str, Any]) -> Optional[Any]:
    """
    Recherche une fonction heuristique par nom, avec correspondance partielle / suffixe.

    Essaie d'abord la clé exacte, puis toute clé se terminant par ``:<name>``.
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
    Résout les noms de solveurs et d'heuristiques configurés en leurs callables.

    Repli automatique
    -----------------
    Lorsque ``solvers`` est vide et ``synthetic=true``, la première clé
    classe-seulement (sans ``:``) de *all_solvers* est auto-sélectionnée
    et la décision est ajoutée à *assumptions* et journalisée en WARNING.

    Retourne
    --------
    solver_classes : dict[nom -> classe]
    heuristic_fns  : dict[nom -> callable]
    assumptions    : liste mise à jour des hypothèses
    """
    assumptions: List[str] = list(cfg.get("_assumptions", []))
    solver_names: List[str] = list(cfg.get("solvers", []))

    # Repli automatique : prend la première clé classe-seulement
    if not solver_names and cfg.get("synthetic"):
        for k in all_solvers:
            if ":" not in k:
                solver_names = [k]
                msg = f"Aucun solveur configuré ; premier disponible auto-sélectionné : {k}"
                assumptions.append(msg)
                logging.warning("HYPOTHÈSE : %s", msg)
                break

    solver_classes: Dict[str, Any] = {}
    for name in solver_names:
        cls = resolve_solver(name, all_solvers)
        if cls is not None:
            solver_classes[name] = cls
        else:
            logging.warning("Solveur '%s' introuvable dans model/final — ignoré.", name)

    heuristic_fns: Dict[str, Any] = {}
    for name in cfg.get("heuristics", []):
        fn = resolve_heuristic(name, all_heuristics)
        if fn is not None:
            heuristic_fns[name] = fn
        else:
            logging.warning("Heuristique '%s' introuvable dans model/heuristics — ignorée.", name)

    return solver_classes, heuristic_fns, assumptions
