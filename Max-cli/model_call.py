"""
Models possibles:
- Max_one
- Max_surface
"""

import logging

from model.final.max_one_final import MaxOneModel
from model.final.max_surface_final import MaxSurfaceModel
from model.heuristics.heuristicA import heuristic
import numpy as np
from gurobipy import GRB

logger = logging.getLogger(__name__)


def find_dense_submatrix(matrix, model="max_one", gamma=0, use_heuristic=0) -> tuple[list[int], list[int], bool]:
    """
    Trouve une sous-matrice dense dans la matrice donnée en utilisant le modèle spécifié.

    ARGUMENTS:
    ----------
    * matrix: matrice d'entrée (numpy array).
    * model: modèle à utiliser pour trouver la sous-matrice dense (string).
    * gamma: error rate max dans la sous matrice (float entre 0 et 1).
    * use_heuristic: 0 pour exact, 1 pour heuristique (int).

    RETOURNE:
    ----------
    * row_indices: indices des lignes de la sous-matrice trouvée.
    * col_indices: indices des colonnes de la sous-matrice trouvée.
    * success: booléen indiquant si une solution satisfaisant le seuil gamma a été trouvée.
    """
    logger.info(
        "find_dense_submatrix appelé — modèle='%s', gamma=%.4f, use_heuristic=%d, "
        "matrice=%dx%d.",
        model, gamma, use_heuristic, matrix.shape[0], matrix.shape[1],
    )

    if model == "max_one":
        model_instance = MaxOneModel
    elif model == "max_surface":
        model_instance = MaxSurfaceModel
    else:
        logger.error("Modèle inconnu : '%s'. Valeurs acceptées : 'max_one', 'max_surface'.", model)
        raise ValueError(f"Modèle inconnu : {model}")

    if use_heuristic == 1:
        logger.info("Mode heuristique activé.")
        row_indices, col_indices, success = heuristic(matrix, model_instance, error_rate=gamma)
    else:
        logger.debug("Construction des données du modèle exact (lignes, colonnes, arêtes).")
        rows_data = [(i, int(np.sum(matrix[i, :]))) for i in range(matrix.shape[0])]
        cols_data = [(j, int(np.sum(matrix[:, j]))) for j in range(matrix.shape[1])]
        edges = [(i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if matrix[i, j] == 1]
        logger.debug("%d lignes, %d colonnes, %d arêtes transmises au modèle.", len(rows_data), len(cols_data), len(edges))

        m = model_instance(rows_data, cols_data, edges, gamma)
        m.setParam('OutputFlag', 0)
        logger.info("Lancement de l'optimisation exacte (modèle '%s').", model)
        m.optimize()

        row_indices, col_indices, success = [], [], False
        if m.status == GRB.OPTIMAL:
            row_indices = m.get_selected_rows()
            col_indices = m.get_selected_cols()
            success = True
            logger.info(
                "Solution optimale trouvée : %d lignes × %d colonnes sélectionnées.",
                len(row_indices), len(col_indices),
            )
        else:
            logger.warning(
                "Optimisation terminée sans solution optimale (status Gurobi=%d).", m.status
            )

    if success:
        logger.info(
            "Résultat final : sous-matrice %dx%d trouvée (success=True).",
            len(row_indices), len(col_indices),
        )
    else:
        logger.warning("Aucune sous-matrice dense trouvée (success=False).")

    return row_indices, col_indices, success