"""
T1/pipeline/planner.py — Construction du plan d'exécution plat (support dry-run).

Le plan est une liste de dicts légers décrivant chaque exécution qui *serait*
lancée, sans rien exécuter réellement. Utilisé à la fois par le mode dry_run
(affiché sur stdout) et par le runner pour l'itération.

API publique
------------
discover_instances(cfg, root) -> list[str]   — chemins vers les fichiers CSV
plan_runs(cfg, all_solvers, all_heuristics, root) -> list[dict]
print_plan(runs)                             — affichage formaté sur stdout
"""

import logging
import os
import sys
from typing import Any, Dict, List

# ── Bootstrap du chemin racine ────────────────────────────────────────────────
_T1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.dirname(_T1_DIR)
for _p in (_ROOT, _T1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.discovery import resolve_all  # noqa: E402


def discover_instances(cfg: Dict[str, Any], root: str) -> List[str]:
    """
    Retourne une liste triée de chemins vers les fichiers CSV pour les instances
    réelles (non synthétiques).

    Ordre de résolution
    -------------------
    1. Si ``cfg["instances"]`` est défini, ces noms de fichiers sont utilisés
       (relatifs à ``instances_dir`` sauf s'ils sont déjà absolus).
    2. Sinon, tous les fichiers ``*.csv`` dans ``instances_dir`` sont retournés.

    Paramètres
    ----------
    cfg  : dict de configuration résolu par ``pipeline.config.build``
    root : répertoire racine du dépôt
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
        logging.warning("instances_dir introuvable : %s", instances_dir)
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
    Construit la liste complète et plate des dicts de description d'exécution.

    Chaque dict contient au minimum :
      ``type``, ``instance_id``, ``gamma``, ``solver_name``, ``seed``

    Les exécutions heuristiques contiennent également ``heuristic_name``.

    Cette fonction n'exécute **rien**.

    Paramètres
    ----------
    cfg            : dict de configuration résolu
    all_solvers    : registre de ``discovery.discover_solvers``
    all_heuristics : registre de ``discovery.discover_heuristics``
    root           : répertoire racine du dépôt

    Retourne
    --------
    list[dict]
    """
    solver_classes, heuristic_fns, _ = resolve_all(cfg, all_solvers, all_heuristics)

    # Détermine quels solveurs utiliser pour les entrées du plan heuristique
    _hs = cfg.get("heuristic_solver", "ALL")
    if _hs.upper() == "ALL":
        heuristic_solver_classes = solver_classes
    else:
        heuristic_solver_classes = {n: c for n, c in solver_classes.items() if n == _hs}
        if not heuristic_solver_classes:
            heuristic_solver_classes = solver_classes

    runs: List[Dict[str, Any]] = []

    if cfg["synthetic"]:
        # En dry-run, les répétitions sont affichées comme des créneaux numérotés ;
        # les graines réelles sont assignées dynamiquement à l'exécution.
        for rep in range(cfg["repetitions"]):
            iid = f"synthetic_L{cfg['L']}_C{cfg['C']}_d{cfg['density']}_rep{rep + 1}"
            for gamma in cfg["gammas"]:
                for sn in solver_classes:
                    runs.append({
                        "type": "exact",
                        "instance_id": iid,
                        "seed": "<dynamique>",
                        "gamma": gamma,
                        "solver_name": sn,
                    })
                for hn in heuristic_fns:
                    for sn in heuristic_solver_classes:
                        runs.append({
                            "type": "heuristic",
                            "instance_id": iid,
                            "seed": "<dynamique>",
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
                        "seed": "<dynamique>",
                        "gamma": gamma,
                        "solver_name": sn,
                    })
                for hn in heuristic_fns:
                    for rep in range(cfg["repetitions"]):
                        for sn in heuristic_solver_classes:
                            runs.append({
                                "type": "heuristic",
                                "instance_id": iid,
                                "seed": "<dynamique>",
                                "gamma": gamma,
                                "heuristic_name": hn,
                                "solver_name": sn,
                            })

    return runs


def print_plan(runs: List[Dict[str, Any]]) -> None:
    """Affiche le plan dry-run de façon formatée sur stdout."""
    print(f"\nDRY RUN — {len(runs)} exécution(s) planifiée(s) :\n")
    for i, r in enumerate(runs, 1):
        if r["type"] == "exact":
            print(
                f"  [{i:>4}] EXACT   "
                f"instance={r['instance_id']}  "
                f"gamma={r['gamma']}  "
                f"solveur={r['solver_name']}  "
                f"graine={r['seed']}"
            )
        else:
            print(
                f"  [{i:>4}] HEUR    "
                f"instance={r['instance_id']}  "
                f"gamma={r['gamma']}  "
                f"solveur={r['solver_name']}  "
                f"heuristique={r['heuristic_name']}  "
                f"graine={r['seed']}"
            )
