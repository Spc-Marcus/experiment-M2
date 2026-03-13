"""
T1/pipeline/metrics.py — Calcul des métriques pour les exécutions de biclustering.

Toutes les fonctions sont pures (sans effets de bord) et peuvent être
testées unitairement de façon indépendante.

API publique
------------
matrix_to_model_inputs(matrix) -> (rows_data, cols_data, edges)
compute_metrics(matrix, row_indices, col_indices) -> (objective, area, density)
compute_gap(best_known, objective) -> float | 'NA'
"""

from typing import Any, List, Optional, Tuple

import numpy as np


def matrix_to_model_inputs(
    matrix: np.ndarray,
) -> Tuple[
    List[Tuple[int, int]],
    List[Tuple[int, int]],
    List[Tuple[int, int]],
]:
    """
    Convertit une matrice NumPy binaire en trois entrées requises par
    ``BiclusterModelBase.__init__``.

    Paramètres
    ----------
    matrix : np.ndarray  forme (m, n), dtype int (valeurs 0/1)

    Retourne
    --------
    rows_data : liste de (indice_ligne, degré_ligne)
    cols_data : liste de (indice_colonne, degré_colonne)
    edges     : liste de (indice_ligne, indice_colonne) pour chaque cellule == 1
    """
    m, n = matrix.shape
    rows_data: List[Tuple[int, int]] = [
        (i, int(matrix[i, :].sum())) for i in range(m)
    ]
    cols_data: List[Tuple[int, int]] = [
        (j, int(matrix[:, j].sum())) for j in range(n)
    ]
    edges: List[Tuple[int, int]] = [
        (i, j) for i in range(m) for j in range(n) if matrix[i, j] == 1
    ]
    return rows_data, cols_data, edges


def compute_metrics(
    matrix: np.ndarray,
    row_indices: List[int],
    col_indices: List[int],
) -> Tuple[int, int, float]:
    """
    Calcule l'objectif, la surface et la densité d'une sous-matrice sélectionnée.

    Paramètres
    ----------
    matrix      : matrice binaire d'entrée complète
    row_indices : indices de lignes sélectionnées (peuvent être non triés)
    col_indices : indices de colonnes sélectionnées (peuvent être non triés)

    Retourne
    --------
    objective : nombre de 1 dans la sous-matrice sélectionnée
    area      : #lignes_sélectionnées × #colonnes_sélectionnées
    density   : objective / area  (0.0 si area == 0)
    """
    if not row_indices or not col_indices:
        return 0, 0, 0.0
    sub = matrix[np.ix_(sorted(row_indices), sorted(col_indices))]
    objective = int(sub.sum())
    area = len(row_indices) * len(col_indices)
    density = objective / area if area > 0 else 0.0
    return objective, area, density


def compute_gap(best_known: Optional[float], objective: Any) -> Any:
    """
    Retourne ``100 × (best_known − objective) / best_known`` en pourcentage.

    Retourne ``'NA'`` lorsque :
      - *best_known* est None ou ≤ 0
      - *objective* n'est pas un entier (ex. ``'NA'`` d'une exécution échouée)
    """
    if best_known is None or best_known <= 0 or not isinstance(objective, int):
        return "NA"
    return round(100.0 * (best_known - objective) / best_known, 4)
