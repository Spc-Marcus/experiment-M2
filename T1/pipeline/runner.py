"""
T1/pipeline/runner.py — Orchestration du pipeline d'expérimentation complet.

Responsabilités
---------------
- Découverte des solveurs et heuristiques.
- Pour chaque groupe (instance/matrice, gamma) :
    1. Exécution des solveurs exacts → collecte du best_known.
    2. Exécution des heuristiques → calcul de l'écart par rapport au best_known.
- Écriture d'une ligne CSV + d'un log JSON par exécution (via pipeline.io).
- Support de l'exécution parallèle via un ThreadPoolExecutor.
- Fourniture de run_quick_check() pour la validation du pipeline.

API publique
------------
execute_pipeline(cfg, csv_path, log_dir, env_info) -> None
run_quick_check(cfg, csv_path, log_dir, env_info)  -> None  (lève en cas d'échec)
"""

import logging
import os
import random
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Bootstrap du chemin racine ────────────────────────────────────────────────
_T1_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.dirname(_T1_DIR)
for _p in (_ROOT, _T1_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline.discovery import discover_heuristics, discover_solvers, resolve_all  # noqa: E402
from pipeline.executor import run_exact_solver, run_heuristic  # noqa: E402
from pipeline.io import append_csv_row, init_csv  # noqa: E402
from pipeline.planner import discover_instances  # noqa: E402


# ── Génération de graine ──────────────────────────────────────────────────────

def _new_seed() -> int:
    """Retourne une nouvelle graine aléatoire dans [0, 2**31 - 1] adaptée à tout RNG."""
    return random.randint(0, 2**31 - 1)


# ── Exécution d'un groupe ─────────────────────────────────────────────────────

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
    Exécute tous les solveurs exacts puis toutes les heuristiques pour un groupe
    (instance, gamma, graine).

    Chaque appel est complètement indépendant :
      - ``error_rate = 1 − gamma`` est calculé localement.
      - ``best_known`` commence à None et n'est mis à jour qu'à partir des
        solveurs exacts de ce groupe, donc chaque valeur de gamma est résolue
        en isolation complète.
      - ``run_seed`` pilote la reproductibilité des heuristiques et, pour les
        matrices synthétiques, identifie également quelle matrice a été utilisée.

    Les solveurs exacts s'exécutent avant les heuristiques afin que ``best_known``
    soit disponible pour la métrique d'écart.
    """
    # Chaque gamma est indépendant : error_rate propre, best_known propre
    error_rate = 1.0 - gamma
    best_known: Optional[float] = None

    # ── Solveurs exacts ────────────────────────────────────────────────────
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

    # ── Heuristiques ──────────────────────────────────────────────────────
    # Détermine quels solveurs injecter comme model_class dans les heuristiques.
    # 'ALL' → tous les solveurs configurés ; un nom spécifique → ce solveur seul.
    _hs = cfg.get("heuristic_solver", "ALL")
    if _hs.upper() == "ALL":
        heuristic_solver_classes = solver_classes
    else:
        heuristic_solver_classes = {n: c for n, c in solver_classes.items() if n == _hs}
        if not heuristic_solver_classes:
            logging.warning(
                "heuristic_solver '%s' introuvable dans les solveurs résolus — "
                "repli sur ALL.",
                _hs,
            )
            heuristic_solver_classes = solver_classes

    # La run_seed est utilisée pour chaque heuristique de ce groupe afin que
    # les résultats soient liés à une seule graine enregistrée.
    for heuristic_name, heuristic_fn in heuristic_fns.items():
        for solver_name, solver_cls in heuristic_solver_classes.items():
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


# ── Aide au parallélisme ──────────────────────────────────────────────────────

def _parallel_execute(items: List[Any], fn: Callable, n_jobs: int) -> None:
    """
    Applique *fn* à chaque élément de *items*, séquentiellement ou via un pool de threads.

    Les exceptions levées dans les workers sont journalisées mais n'interrompent
    pas le reste du travail.

    Paramètres
    ----------
    items  : liste d'arguments à passer à fn
    fn     : callable acceptant un seul argument
    n_jobs : 1 → séquentiel ;  >1 → ThreadPoolExecutor avec ce nombre de threads
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
                    "Erreur worker pour %s :\n%s",
                    futures[future],
                    traceback.format_exc(),
                )


# ── Pipeline complet ──────────────────────────────────────────────────────────

def execute_pipeline(
    cfg: Dict[str, Any],
    csv_path: str,
    log_dir: str,
    env_info: Dict[str, Any],
) -> None:
    """
    Découvre les solveurs/heuristiques, génère/charge les matrices et lance
    toutes les expériences.

    Une ligne CSV et un log JSON sont écrits par exécution. La fonction ne
    lève jamais d'exception — les erreurs individuelles sont capturées dans
    l'exécuteur.

    Paramètres
    ----------
    cfg      : dict de configuration résolu par ``pipeline.config.build``
    csv_path : chemin vers le CSV de résultats (créé/écrasé ici)
    log_dir  : répertoire pour les logs JSON d'exécution
    env_info : métadonnées d'environnement de ``utils.env_info.collect``
    """
    all_solvers = discover_solvers(_ROOT)
    all_heuristics = discover_heuristics(_ROOT)

    logging.info("Solveurs disponibles :    %s", [k for k in all_solvers if ":" not in k])
    logging.info("Heuristiques disponibles : %s", [k for k in all_heuristics if ":" not in k])

    solver_classes, heuristic_fns, assumptions = resolve_all(
        cfg, all_solvers, all_heuristics
    )

    if not solver_classes:
        logging.warning("Aucune classe de solveur résolue — les exécutions exactes seront ignorées.")

    init_csv(csv_path)

    if cfg["synthetic"]:
        from utils.create_matrix_V2 import create_matrix  # noqa: E402

        # Génère une graine par répétition dynamiquement.
        # Chaque graine est enregistrée dans le CSV/log pour la reproductibilité complète.
        rep_seeds = [_new_seed() for _ in range(cfg["repetitions"])]

        # Groupes : chaque répétition × chaque gamma est indépendant
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
                "Synthétique graine=%d gamma=%.3f : %dx%d dens=%.4f",
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
                "Aucune instance trouvée. Vérifiez 'instances_dir' / 'instances' dans la configuration."
            )
            return

        def _run_real(inst_gamma_seed: Tuple) -> None:
            inst_path, gamma, run_seed = inst_gamma_seed
            iid = os.path.splitext(os.path.basename(inst_path))[0]
            try:
                matrix = load_csv_matrix(inst_path)
            except Exception as exc:
                logging.error("Échec du chargement de %s : %s", inst_path, exc)
                return
            if matrix.size == 0:
                logging.warning("Matrice vide pour %s — ignorée.", inst_path)
                return
            logging.info(
                "Instance %s gamma=%.3f graine=%d : %dx%d dens=%.4f",
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

        # Génère une graine par répétition ; associe chacune à chaque (instance, gamma)
        rep_seeds = [_new_seed() for _ in range(cfg["repetitions"])]
        groups = [
            (p, g, s)
            for p in instances
            for g in cfg["gammas"]
            for s in rep_seeds
        ]
        _parallel_execute(groups, _run_real, cfg["parallel_jobs"])


# ── Vérification rapide ───────────────────────────────────────────────────────

def run_quick_check(
    cfg: Dict[str, Any],
    csv_path: str,
    log_dir: str,
    env_info: Dict[str, Any],
) -> None:
    """
    Validation minimale du pipeline : matrice synthétique 5×5, density=0.35,
    graine=42, gamma=0.9.

    Vérifie que :
      1. Le runner s'exécute sans exception non gérée.
      2. Un CSV avec au moins une ligne de données est produit.
      3. Au moins un fichier log JSON est produit.

    Lève
    ----
    RuntimeError
        Si aucun solveur n'est disponible, ou si les sorties attendues sont absentes.
    """
    from utils.create_matrix_V2 import create_matrix  # noqa: E402

    logging.info("quick_check : matrice 5×5, density=0.35, graine=42, gamma=0.9")

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
        "heuristic_solver": cfg.get("heuristic_solver", "ALL"),
        "_assumptions": list(cfg.get("_assumptions", [])),
    }

    solver_classes, heuristic_fns, assumptions = resolve_all(
        qcfg, all_solvers, all_heuristics
    )

    if not solver_classes:
        raise RuntimeError("quick_check ÉCHOUÉ : aucune classe de solveur disponible.")

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

    # ── Validation des sorties ─────────────────────────────────────────────
    if not os.path.exists(csv_path):
        raise RuntimeError("quick_check ÉCHOUÉ : CSV non produit.")

    with open(csv_path) as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise RuntimeError(
            f"quick_check ÉCHOUÉ : le CSV contient {len(lines)} ligne(s) ; "
            "en-tête + ≥1 ligne de données attendus."
        )

    log_files = (
        [fn for fn in os.listdir(log_dir) if fn.endswith(".json")]
        if os.path.isdir(log_dir)
        else []
    )
    if not log_files:
        raise RuntimeError("quick_check ÉCHOUÉ : aucun log JSON produit.")

    logging.info(
        "quick_check RÉUSSI : %d ligne(s) CSV, %d log(s) JSON.",
        len(lines) - 1,
        len(log_files),
    )
    print(
        f"quick_check RÉUSSI : {len(lines) - 1} ligne(s) CSV, "
        f"{len(log_files)} log(s) JSON.  CSV : {csv_path}"
    )
