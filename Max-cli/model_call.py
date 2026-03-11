"""
model_call.py – Couche d'appel aux modèles ILP pour Max-cli.

Expose une fonction unique ``find_dense_submatrix`` qui sélectionne
le modèle demandé (max_one_v2 ou max_e_r_v2) et retourne les indices
de lignes/colonnes de la plus grande sous-matrice dense trouvée.
"""

import os
import sys
import logging

import numpy as np

# ── Ajout du chemin racine pour les imports relatifs ──────────────────────────
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Les fonctions Gurobi sont dans No-error/ilp_grb.py
_NO_ERROR = os.path.join(_ROOT, "No-error")
if _NO_ERROR not in sys.path:
    sys.path.insert(0, _NO_ERROR)

from ilp_grb import (
    find_quasi_biclique_max_one_V2,
    find_quasi_biclique_max_e_r_V2,
)

logger = logging.getLogger("experiment.model_call")

_MODELS = {
    "max_one_v2": find_quasi_biclique_max_one_V2,
    "max_e_r_v2": find_quasi_biclique_max_e_r_V2,
}


def find_dense_submatrix(
    matrix: np.ndarray,
    model: str = "max_one_v2",
    gamma: float = 0.975,
) -> tuple:
    """
    Trouve la plus grande sous-matrice dense dans ``matrix``.

    Parameters
    ----------
    matrix : np.ndarray
        Matrice binaire (0/1) de forme (L, C).
    model : str
        Modèle à utiliser : ``"max_one_v2"`` (défaut) ou ``"max_e_r_v2"``.
    gamma : float
        Densité minimale souhaitée (0 < gamma <= 1). Correspond à
        ``1 - error_rate`` dans la formulation ILP interne.
        Par défaut 0.975 (2.5 % de zéros tolérés).

    Returns
    -------
    rows : list[int]
        Indices des lignes retenues.
    cols : list[int]
        Indices des colonnes retenues.
    success : bool
        True si une sous-matrice valide a été trouvée.

    Raises
    ------
    ValueError
        Si le nom de modèle n'est pas reconnu.
    """
    if model not in _MODELS:
        raise ValueError(
            f"Modèle inconnu : '{model}'. "
            f"Valeurs autorisées : {list(_MODELS.keys())}"
        )

    error_rate = max(0.0, 1.0 - gamma)
    logger.info(
        "Appel du modèle '%s' (gamma=%.4f, error_rate=%.4f) "
        "sur une matrice %dx%d.",
        model, gamma, error_rate, matrix.shape[0], matrix.shape[1],
    )

    rows, cols, success = _MODELS[model](matrix, error_rate=error_rate)

    if success:
        density = (
            matrix[np.ix_(rows, cols)].mean() if (rows and cols) else 0.0
        )
        logger.info(
            "Sous-matrice trouvée : %d lignes × %d colonnes "
            "(densité réelle = %.4f).",
            len(rows), len(cols), density,
        )
    else:
        logger.warning("Aucune sous-matrice dense trouvée.")

    return rows, cols, success
