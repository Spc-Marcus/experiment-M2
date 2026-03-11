"""
T1/pipeline/config.py — Lecture et validation de la configuration.

Lit un fichier de configuration ``clé=valeur`` et retourne un dictionnaire
entièrement résolu avec des valeurs typées et des valeurs par défaut sensées.
Chaque valeur par défaut appliquée est enregistrée dans ``cfg["_assumptions"]``
afin de pouvoir être journalisée et stockée dans les fichiers JSON de logs.

API publique
------------
parse_file(path) -> Dict[str, str]   – paires clé/valeur brutes sous forme de chaînes
build(raw)       -> Dict[str, Any]   – configuration typée et validée
"""

import os
from typing import Any, Dict, List


# ── Fonctions utilitaires privées ──────────────────────────────────────────────

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
    """Convertit ``'L:200,C:200,density:0.1'`` en ``{'L':'200', ...}``."""
    specs: Dict[str, str] = {}
    for part in _parse_list(value):
        if ":" in part:
            k, _, v = part.partition(":")
            specs[k.strip()] = v.strip()
    return specs


# ── API publique ───────────────────────────────────────────────────────────────

def parse_file(path: str) -> Dict[str, str]:
    """
    Analyse un fichier de configuration ``clé=valeur`` et retourne un
    dictionnaire de chaînes brutes.

    Règles
    ------
    - Les lignes commençant par ``#`` sont ignorées.
    - Les lignes sans ``=`` sont ignorées.
    - Les valeurs restent des chaînes ; utiliser :func:`build` pour les types.

    Retourne un dict vide (avec un avertissement) si *path* n'existe pas.
    """
    config: Dict[str, str] = {}
    if not os.path.exists(path):
        import logging
        logging.warning("Fichier de configuration introuvable : %s — utilisation des valeurs par défaut.", path)
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
    Construit un dictionnaire de configuration entièrement résolu et typé
    à partir de paires clé=valeur brutes.

    Les valeurs par défaut sont appliquées lorsque les clés sont absentes ;
    chaque valeur par défaut appliquée est ajoutée à ``cfg["_assumptions"]``
    pour journalisation et audit.

    Paramètres
    ----------
    raw : dict[str, str]
        Résultat de :func:`parse_file` (ou un dict vide pour tout en défaut).

    Retourne
    --------
    dict[str, Any] avec les clés suivantes :

    instances_dir, instances, synthetic, L, C, density,
    repetitions, gammas, solvers, heuristics, heuristic_solver,
    timeout_exact, timeout_heuristic, output_dir,
    parallel_jobs, dry_run, quick_check, _assumptions
    """
    assumptions: List[str] = []
    cfg: Dict[str, Any] = {}

    # ── Source de données ──────────────────────────────────────────────────
    if "instances_dir" not in raw:
        assumptions.append("instances_dir absent → valeur par défaut 'Mat'")
    cfg["instances_dir"] = raw.get("instances_dir", "Mat")

    cfg["instances"] = _parse_list(raw.get("instances", ""))

    # ── Indicateur synthétique ─────────────────────────────────────────────
    cfg["synthetic"] = _parse_bool(raw.get("synthetic", "false"))

    # ── Spécifications de la matrice synthétique ───────────────────────────
    specs_raw = raw.get("synthetic_specs")
    if specs_raw is None:
        assumptions.append("synthetic_specs absent → valeur par défaut L=50,C=50,density=0.35")
        specs_raw = "L:50,C:50,density:0.35"
    specs = _parse_synthetic_specs(specs_raw)
    cfg["L"] = int(specs.get("L", 50))
    cfg["C"] = int(specs.get("C", 50))
    cfg["density"] = float(specs.get("density", 0.35))

    # ── Répétitions ────────────────────────────────────────────────────────
    # Nombre d'exécutions indépendantes par paire (instance, gamma). Chaque
    # exécution reçoit une graine générée aléatoirement, enregistrée dans le
    # CSV/log pour la reproductibilité.
    if "repetitions" not in raw:
        assumptions.append("repetitions absent → valeur par défaut 5")
    cfg["repetitions"] = max(1, int(raw.get("repetitions", 5)))

    # ── Gammas ─────────────────────────────────────────────────────────────
    gammas_raw = raw.get("gammas")
    if gammas_raw is None:
        assumptions.append("gammas absent → valeur par défaut [0.9, 0.95, 0.99, 1.0]")
        gammas_raw = "0.9,0.95,0.99,1.0"
    cfg["gammas"] = [float(g) for g in _parse_list(gammas_raw)]
    if not cfg["gammas"]:
        cfg["gammas"] = [0.95]
        assumptions.append("liste gammas vide → valeur par défaut [0.95]")

    # ── Solveurs / heuristiques ────────────────────────────────────────────
    cfg["solvers"] = _parse_list(raw.get("solvers", ""))
    cfg["heuristics"] = _parse_list(raw.get("heuristics", ""))

    # ── Solveur pour les heuristiques ──────────────────────────────────────
    # Quelle classe de solveur injecter comme ``model_class`` dans les
    # heuristiques.
    # 'ALL' (insensible à la casse) → chaque solveur configuré est utilisé
    # pour chaque exécution heuristique (une ligne par paire heuristique×solveur).
    # Un nom de classe spécifique → seul ce solveur est utilisé ; il doit
    # figurer dans ``solvers``. Retombe sur ALL avec un avertissement si absent.
    cfg["heuristic_solver"] = raw.get("heuristic_solver", "ALL").strip()

    # ── Délais d'expiration ────────────────────────────────────────────────
    cfg["timeout_exact"] = int(raw.get("timeout_exact", 600))
    cfg["timeout_heuristic"] = int(raw.get("timeout_heuristic", 150))

    # ── Sortie / parallélisme / indicateurs ───────────────────────────────
    cfg["output_dir"] = raw.get("output_dir", "T1/results")
    cfg["parallel_jobs"] = int(raw.get("parallel_jobs", 1))
    cfg["dry_run"] = _parse_bool(raw.get("dry_run", "false"))
    cfg["quick_check"] = _parse_bool(raw.get("quick_check", "false"))

    cfg["_assumptions"] = assumptions
    return cfg
