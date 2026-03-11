import logging
from typing import List, Tuple, Type

import numpy as np

from model.base import BiclusterModelBase

logger = logging.getLogger(__name__)


def heuristic(
    input_matrix: np.ndarray,
    model_class: Type[BiclusterModelBase],
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    V3c: Reconstruction du modèle à chaque phase avec seulement les seeds, SANS WarmStart.

    Stratégie: Crée un nouveau modèle à chaque phase avec uniquement
    les lignes/colonnes potentielles. Pas de WarmStart.
    """
    X_problem = input_matrix.copy()
    cols_sorted = np.argsort(X_problem.sum(axis=0))[::-1]
    rows_sorted = np.argsort(X_problem.sum(axis=1))[::-1]
    m = len(rows_sorted)
    n = len(cols_sorted)

    logger.info(
        "Heuristique démarrée — matrice %dx%d, error_rate=%.4f, modèle=%s.",
        m, n, error_rate, model_class.__name__,
    )

    if m == 0 or n == 0:
        logger.warning("Matrice vide (m=%d, n=%d). Abandon.", m, n)
        return [], [], False

    seed_cols = max(n // 3, 2)
    if n > 50:
        step_n = 10
    else:
        step_n = 2
    for x in range(m // 3, m, 10):
        for y in range(seed_cols, n, step_n):
            nb_of_ones = 0
            for row in rows_sorted[:x]:
                for col in cols_sorted[:y]:
                    nb_of_ones += X_problem[row, col]
            ratio_ones = nb_of_ones / (x * y) if (x * y) > 0 else 0
            if ratio_ones > 0.99:
                seed_cols = y

    logger.debug("seed_cols déterminé : %d.", seed_cols)

    try:
        # PHASE 1: Modèle uniquement sur seed_rows x seed_cols
        logger.info("Phase 1 : modèle initial sur %d lignes × %d colonnes (seed).", m, seed_cols)
        seed_row_indices = rows_sorted
        seed_col_indices = cols_sorted[:seed_cols]

        row_degrees = np.sum(X_problem[seed_row_indices, :][:, seed_col_indices] == 1, axis=1)
        col_degrees = np.sum(X_problem[seed_row_indices, :][:, seed_col_indices] == 1, axis=0)
        rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(seed_row_indices)]
        cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(seed_col_indices)]
        edges = []
        for r in seed_row_indices:
            for c in seed_col_indices:
                if X_problem[r, c] == 1:
                    edges.append((int(r), int(c)))

        model = model_class(rows_data, cols_data, edges, 0.0)
        model.setParam('OutputFlag', 0)
        model.setParam('MIPGap', 0.05)
        #model.setParam('TimeLimit', 20)
        model.setParam('Seed', 1)
        model.setParam('IntFeasTol', 1e-9)
        model.setParam('FeasibilityTol', 1e-9)
        model.setParam('OptimalityTol', 1e-9)
        model.setParam('NumericFocus', 1)
        model.optimize()
        if model.status != 2:
            logger.warning("Phase 1 : status non-optimal (%d). Abandon.", model.status)
            return [], [], False
        rw = model.get_selected_rows()
        cl = model.get_selected_cols()
        logger.info("Phase 1 terminée : %d lignes × %d colonnes sélectionnées.", len(rw), len(cl))

        # PHASE 2: Nouveau modèle pour extension colonnes
        logger.info("Phase 2 : extension des colonnes (relâchement de toutes les colonnes restantes).")
        rem_cols = [c for c in cols_sorted if c not in cl]
        # Relâcher toutes les colonnes restantes (ne pas filtrer seulement les 'potential')
        potential_cols = rem_cols

        logger.debug("Phase 2 : %d colonnes candidates à l'extension.", len(potential_cols))

        if potential_cols:
            all_col_indices = cl + potential_cols
            row_degrees = np.sum(X_problem[rw, :][:, all_col_indices] == 1, axis=1)
            rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(rw)]
            col_degrees = np.sum(X_problem[rw, :][:, all_col_indices] == 1, axis=0)
            cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(all_col_indices)]
            edges = []
            for r in rw:
                for c in all_col_indices:
                    if X_problem[r, c] == 1:
                        edges.append((int(r), int(c)))
            model2 = model_class(rows_data, cols_data, edges, error_rate)
            model2.setParam('OutputFlag', 0)
            model2.setParam('MIPGap', 0.05)
            model2.setParam('TimeLimit', 180)
            model2.setParam('Seed', 1)
            model2.setParam('IntFeasTol', 1e-9)
            model2.setParam('FeasibilityTol', 1e-9)
            model2.setParam('OptimalityTol', 1e-9)
            model2.setParam('NumericFocus', 1)
            # PAS DE WARMSTART
            model2.optimize()
            if model2.status == 2:
                rw = model2.get_selected_rows()
                cl = model2.get_selected_cols()
                logger.info("Phase 2 terminée : %d lignes × %d colonnes.", len(rw), len(cl))
            else:
                logger.warning(
                    "Phase 2 : status non-optimal (%d). Retour au résultat de phase 1.",
                    model2.status,
                )
                return rw, cl, True
        else:
            logger.info("Phase 2 : aucune colonne restante, extension ignorée.")

        # PHASE 3: Nouveau modèle pour extension lignes
        logger.info("Phase 3 : extension des lignes (relâchement de toutes les lignes restantes).")
        rem_rows = [r for r in rows_sorted if r not in rw]
        # Relâcher toutes les lignes restantes (ne pas filtrer seulement les 'potential')
        potential_rows = rem_rows

        logger.debug("Phase 3 : %d lignes candidates à l'extension.", len(potential_rows))

        if potential_rows:
            all_row_indices = rw + potential_rows
            row_degrees = np.sum(X_problem[all_row_indices, :][:, cl] == 1, axis=1)
            rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(all_row_indices)]
            col_degrees = np.sum(X_problem[all_row_indices, :][:, cl] == 1, axis=0)
            cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(cl)]
            edges = []
            for r in all_row_indices:
                for c in cl:
                    if X_problem[r, c] == 1:
                        edges.append((int(r), int(c)))
            model3 = model_class(rows_data, cols_data, edges, error_rate)
            model3.setParam('OutputFlag', 0)
            model3.setParam('MIPGap', 0.05)
            model3.setParam('TimeLimit', 20)
            model3.setParam('Seed', 1)
            model3.setParam('IntFeasTol', 1e-9)
            model3.setParam('FeasibilityTol', 1e-9)
            model3.setParam('OptimalityTol', 1e-9)
            model3.setParam('NumericFocus', 1)
            # PAS DE WARMSTART
            model3.optimize()
            if model3.status == 2:
                rw = model3.get_selected_rows()
                cl = model3.get_selected_cols()
                logger.info("Phase 3 terminée : %d lignes × %d colonnes.", len(rw), len(cl))
            else:
                logger.warning(
                    "Phase 3 : status non-optimal (%d). Retour au résultat de phase 2.",
                    model3.status,
                )
                return rw, cl, True
        else:
            logger.info("Phase 3 : aucune ligne restante, extension ignorée.")

        if rw and cl:
            logger.info("Heuristique terminée avec succès : %d lignes × %d colonnes.", len(rw), len(cl))
            return rw, cl, True
        else:
            logger.warning("Heuristique terminée sans résultat valide (rw ou cl vide).")
            return [], [], False

    except Exception:
        logger.exception("Exception inattendue dans l'heuristique V3c.")
        return [], [], False
    