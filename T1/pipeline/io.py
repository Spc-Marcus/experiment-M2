"""
T1/pipeline/io.py — Écriture CSV et JSON thread-safe pour les résultats d'expériences.

L'en-tête CSV est défini ici comme unique source de vérité ; tout autre
module ayant besoin des noms de colonnes importe ``CSV_HEADER`` depuis ici.

API publique
------------
CSV_HEADER        : list[str]  — ordre canonique des colonnes
init_csv(path)    — crée / écrase le CSV de résultats avec l'en-tête
append_csv_row(path, row) — ajoute une ligne de résultat (thread-safe)
write_json_log(log_dir, run_id, data) -> str  — écrit un journal JSON par exécution
"""

import csv
import json
import os
import threading
from typing import Any, Dict

# ── En-tête CSV canonique ──────────────────────────────────────────────────────
CSV_HEADER = [
    "instance_id", "m", "n", "base_dens", "gamma", "solver",
    "seed", "heuristic", "time", "status", "objective", "area",
    "density", "gap",
]

# ── Thread-safety pour les écritures CSV concurrentes ────────────────────────
_csv_lock = threading.Lock()


def init_csv(csv_path: str) -> None:
    """Crée (ou écrase) le CSV de résultats et écrit l'en-tête standard."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_HEADER).writeheader()


def append_csv_row(csv_path: str, row: Dict[str, Any]) -> None:
    """Ajoute une ligne de résultat à *csv_path* (thread-safe via verrou de module)."""
    with _csv_lock:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADER).writerow(row)


def write_json_log(log_dir: str, run_id: str, data: Dict[str, Any]) -> str:
    """
    Sérialise *data* en JSON indenté dans ``<log_dir>/<run_id_sûr>.json``.

    Les caractères de *run_id* qui ne sont pas alphanumériques ou dans
    ``-_.`` sont remplacés par ``_`` pour produire un nom de fichier sûr.

    Paramètres
    ----------
    log_dir : str  — répertoire de destination (créé si absent)
    run_id  : str  — identifiant unique de l'exécution
    data    : dict — contenu à sérialiser (``default=str`` gère les inconnus)

    Retourne
    --------
    str : chemin absolu du fichier écrit
    """
    os.makedirs(log_dir, exist_ok=True)
    safe_id = "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_" for c in run_id
    )
    path = os.path.join(log_dir, f"{safe_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path
