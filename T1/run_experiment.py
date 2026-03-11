#!/usr/bin/env python3
"""
T1/run_experiment.py — Point d'entrée CLI du pipeline d'expérimentation T1.

Ce fichier est intentionnellement minimal : il gère uniquement l'analyse des
arguments et relie les composants entre eux. Toute la logique se trouve dans
``T1/pipeline/``.

Utilisation
-----------
    python T1/run_experiment.py [fichier_config]
    python T1/run_experiment.py --dry-run [fichier_config]
    python T1/run_experiment.py --quick-check [fichier_config]
    python T1/run_experiment.py --log-level DEBUG [fichier_config]

Si *fichier_config* est omis, ``T1/config.arg`` est utilisé s'il est présent.
"""

import argparse
import logging
import os
import sys
import time
import traceback

# ── Configuration du chemin (doit précéder les imports du pipeline) ───────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import config as cfg_module          # noqa: E402
from pipeline.discovery import discover_heuristics, discover_solvers  # noqa: E402
from pipeline.planner import plan_runs, print_plan  # noqa: E402
from pipeline.runner import execute_pipeline, run_quick_check  # noqa: E402
from utils.env_info import collect as collect_env  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline d'expérimentation reproductible T1 pour le biclustering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python T1/run_experiment.py --quick-check
  python T1/run_experiment.py --dry-run
  python T1/run_experiment.py T1/config.arg
  python T1/run_experiment.py --log-level DEBUG T1/config.arg
""",
    )
    p.add_argument(
        "config", nargs="?", default=None,
        help="Chemin vers config.arg (défaut : T1/config.arg)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Affiche les exécutions planifiées sans les lancer.",
    )
    p.add_argument(
        "--quick-check", action="store_true",
        help="Lance le pipeline de validation minimal sur une matrice 5x5.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        help="Niveau de journalisation : DEBUG, INFO, WARNING, ERROR (défaut : INFO)",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Localisation et lecture de la configuration ────────────────────────
    config_path = args.config
    if config_path is None:
        default_cfg = os.path.join(_THIS_DIR, "config.arg")
        if os.path.exists(default_cfg):
            config_path = default_cfg
        else:
            logging.warning("Aucun config.arg trouvé ; toutes les valeurs par défaut intégrées seront utilisées.")

    raw = cfg_module.parse_file(config_path) if config_path else {}
    cfg = cfg_module.build(raw)

    # Les indicateurs CLI remplacent les valeurs du fichier de configuration
    if args.dry_run:
        cfg["dry_run"] = True
    if args.quick_check:
        cfg["quick_check"] = True

    if cfg["_assumptions"]:
        logging.info("Hypothèses actives : %s", cfg["_assumptions"])

    # ── Chemins de sortie ─────────────────────────────────────────────────
    output_dir: str = cfg["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_ROOT, output_dir)
    timestamp = int(time.time())
    csv_path = os.path.join(output_dir, f"results_{timestamp}.csv")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env_info = collect_env(_ROOT)
    logging.info("git=%s  python=%s", env_info["git_hash"], env_info["python_version"])

    # ── Mode quick_check ──────────────────────────────────────────────────
    if cfg["quick_check"]:
        try:
            run_quick_check(cfg, csv_path, log_dir, env_info)
        except Exception as exc:
            diag = os.path.join(output_dir, "quick_check_diagnostic.txt")
            with open(diag, "w", encoding="utf-8") as fh:
                fh.write(f"quick_check ÉCHOUÉ\n\n{traceback.format_exc()}")
            logging.error("quick_check ÉCHOUÉ : %s  Diagnostic : %s", exc, diag)
            sys.exit(1)
        return

    # ── Mode dry_run ──────────────────────────────────────────────────────
    if cfg["dry_run"]:
        all_solvers = discover_solvers(_ROOT)
        all_heuristics = discover_heuristics(_ROOT)
        runs = plan_runs(cfg, all_solvers, all_heuristics, _ROOT)
        print_plan(runs)
        return

    # ── Pipeline complet ──────────────────────────────────────────────────
    execute_pipeline(cfg, csv_path, log_dir, env_info)
    logging.info("Terminé. CSV de résultats : %s", csv_path)
    print(f"Terminé. Résultats : {csv_path}")


if __name__ == "__main__":
    main()
