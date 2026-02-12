import numpy as np
from typing import List, Tuple
from ilp_grb import find_quasi_biclique_max_one_V3c


def clustering_full_matrix(input_matrix: np.ndarray,
                           regions: list[int],
                           only_ones: bool = False,
                           min_row_quality: int = 5,
                           min_col_quality: int = 3,
                           error_rate: float = 0.025
                           ) -> tuple[list[tuple[list[int], list[int], list[int]]], dict]:
    """
    Biclustering itératif exhaustif sur une matrice binaire.
    
    Parameters
    ----------
    only_ones : bool
        Si True, utilise clustering_step_only_ones (recherche uniquement les 1).
        Si False, utilise clustering_step_alternating (alterne 1 et 0).
    """
    steps_result = []
    metrics_list = []
    remain_cols = regions
    status = True

    if len(remain_cols) >= min_col_quality:
        while len(remain_cols) >= min_col_quality and status:
            if only_ones:
                (reads1, reads0, cols), step_metrics = clustering_step_only_ones(
                    input_matrix[:, remain_cols],
                    error_rate=error_rate,
                    min_row_quality=min_row_quality,
                    min_col_quality=min_col_quality)
            else:
                (reads1, reads0, cols), step_metrics = clustering_step_alternating(
                    input_matrix[:, remain_cols],
                    error_rate=error_rate,
                    min_row_quality=min_row_quality,
                    min_col_quality=min_col_quality)

            cols = [remain_cols[c] for c in cols]
            if len(reads1) + len(reads0) < 0.9 * len(input_matrix[:, remain_cols]):
                status = False
            elif len(cols) < min_col_quality:
                status = False
            else:
                steps_result.append((reads1, reads0, cols))
                metrics_list.append(step_metrics)
                remain_cols = [c for c in remain_cols if c not in cols]

    # Métriques globales
    if metrics_list:
        nb_ilp_steps = sum(m.get('nb_ilp_steps', 0) for m in metrics_list)
        max_ilp_cluster_size = max((m.get('max_ilp_cluster_size', -1) for m in metrics_list), default=-1)
        dens0 = [m.get('density_cluster0', -1) for m in metrics_list if m.get('density_cluster0', -1) >= 0]
        dens1 = [m.get('density_cluster1', -1) for m in metrics_list if m.get('density_cluster1', -1) >= 0]
        metrics = {
            "nb_ilp_steps": nb_ilp_steps,
            "max_ilp_cluster_size": max_ilp_cluster_size,
            "min_density_cluster0": min(dens0) if dens0 else -1,
            "max_density_cluster0": max(dens0) if dens0 else -1,
            "mean_density_cluster0": sum(dens0) / len(dens0) if dens0 else -1,
            "min_density_cluster1": min(dens1) if dens1 else -1,
            "max_density_cluster1": max(dens1) if dens1 else -1,
            "mean_density_cluster1": sum(dens1) / len(dens1) if dens1 else -1,
            "nb_strips_from_ilp": len(steps_result),
            "found": True,
        }
    else:
        metrics = {
            "nb_ilp_steps": 0, "max_ilp_cluster_size": -1,
            "min_density_cluster0": -1, "max_density_cluster0": -1, "mean_density_cluster0": -1,
            "min_density_cluster1": -1, "max_density_cluster1": -1, "mean_density_cluster1": -1,
            "nb_strips_from_ilp": 0, "found": False,
        }
    return steps_result, metrics


# =============================================================================
# VERSION ORIGINALE : alterne recherche de 1 puis de 0
# =============================================================================
def clustering_step_alternating(input_matrix: np.ndarray,
                                error_rate: float = 0.025,
                                min_row_quality: int = 5,
                                min_col_quality: int = 3,
                                ) -> tuple[tuple[list[int], list[int], list[int]], dict]:
    """
    Clustering step ORIGINAL : alterne entre recherche de 1 et recherche de 0.
    """
    nb_ilp_steps = 0
    max_ilp_cluster_size = 0
    found = False

    matrix1 = input_matrix.copy()
    matrix1[matrix1 == -1] = 0

    matrix0 = input_matrix.copy()
    matrix0[matrix0 == -1] = 1
    matrix0 = (matrix0 - 1) * -1

    remain_rows = list(range(matrix1.shape[0]))
    current_cols = list(range(matrix1.shape[1]))
    clustering_1 = True  # Commence par les 1, puis alterne
    status = True
    rw1, rw0 = [], []

    while len(remain_rows) >= min_row_quality and len(current_cols) >= min_col_quality and status:
        if clustering_1:
            rw, cl, status = find_quasi_biclique_max_one_V3c(
                matrix1[remain_rows][:, current_cols], error_rate)
        else:
            rw, cl, status = find_quasi_biclique_max_one_V3c(
                matrix0[remain_rows][:, current_cols], error_rate)
        nb_ilp_steps += 1

        rw = [remain_rows[r] for r in rw]
        cl = [current_cols[c] for c in cl]

        if len(cl) < min_col_quality:
            status = False
        else:
            current_cols = cl
            if status and len(cl) > 0:
                found = True
                if len(rw) * len(cl) > max_ilp_cluster_size:
                    max_ilp_cluster_size = len(rw) * len(cl)
                if clustering_1:
                    rw1.extend(rw)
                else:
                    rw0.extend(rw)

        remain_rows = [r for r in remain_rows if r not in rw]
        clustering_1 = not clustering_1  # ALTERNE

    if isinstance(current_cols, range):
        current_cols = list(current_cols)

    density_cluster0, density_cluster1 = -1, -1
    if found:
        if len(rw0) > 0:
            sub0 = input_matrix[rw0, :][:, current_cols]
            if sub0.size > 0:
                density_cluster0 = sub0.sum() / sub0.size
        if len(rw1) > 0:
            sub1 = input_matrix[rw1, :][:, current_cols]
            if sub1.size > 0:
                density_cluster1 = sub1.sum() / sub1.size

    metrics = {
        "nb_ilp_steps": nb_ilp_steps,
        "max_ilp_cluster_size": max_ilp_cluster_size,
        "density_cluster0": density_cluster0,
        "density_cluster1": density_cluster1,
        "found": found,
    }
    return (rw0, rw1, current_cols), metrics


# =============================================================================
# NOUVELLE VERSION : recherche UNIQUEMENT les 1 jusqu'à épuisement
# =============================================================================
def clustering_step_only_ones(input_matrix: np.ndarray,
                              error_rate: float = 0.025,
                              min_row_quality: int = 5,
                              min_col_quality: int = 3,
                              ) -> tuple[tuple[list[int], list[int], list[int]], dict]:
    """
    Clustering step MODIFIÉ : cherche UNIQUEMENT les patterns de 1 de manière
    itérative, sans jamais chercher les 0. Les rows restants deviennent rw0.
    """
    nb_ilp_steps = 0
    max_ilp_cluster_size = 0
    found = False

    matrix1 = input_matrix.copy()
    matrix1[matrix1 == -1] = 0

    remain_rows = list(range(matrix1.shape[0]))
    current_cols = list(range(matrix1.shape[1]))
    status = True
    rw1 = []

    # Cherche UNIQUEMENT les 1, pas d'alternance
    while len(remain_rows) >= min_row_quality and len(current_cols) >= min_col_quality and status:
        rw, cl, status = find_quasi_biclique_max_one_V3c(
            matrix1[remain_rows][:, current_cols], error_rate)
        nb_ilp_steps += 1

        rw = [remain_rows[r] for r in rw]
        cl = [current_cols[c] for c in cl]

        if len(cl) < min_col_quality:
            status = False
        else:
            current_cols = cl
            if status and len(cl) > 0:
                found = True
                if len(rw) * len(cl) > max_ilp_cluster_size:
                    max_ilp_cluster_size = len(rw) * len(cl)
                rw1.extend(rw)

        remain_rows = [r for r in remain_rows if r not in rw]
        # PAS D'ALTERNANCE - on continue à chercher des 1

    # Tous les rows restants vont dans rw0
    rw0 = remain_rows

    if isinstance(current_cols, range):
        current_cols = list(current_cols)

    density_cluster0, density_cluster1 = -1, -1
    if found:
        if len(rw0) > 0:
            sub0 = input_matrix[rw0, :][:, current_cols]
            if sub0.size > 0:
                density_cluster0 = sub0.sum() / sub0.size
        if len(rw1) > 0:
            sub1 = input_matrix[rw1, :][:, current_cols]
            if sub1.size > 0:
                density_cluster1 = sub1.sum() / sub1.size

    metrics = {
        "nb_ilp_steps": nb_ilp_steps,
        "max_ilp_cluster_size": max_ilp_cluster_size,
        "density_cluster0": density_cluster0,
        "density_cluster1": density_cluster1,
        "found": found,
    }
    return (rw0, rw1, current_cols), metrics
