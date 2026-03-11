"""
T1/pipeline/metrics.py — Metric computation for biclustering runs.

All functions are pure (no side effects) and can be unit-tested independently.

Public API
----------
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
    Convert a binary NumPy matrix to the three inputs required by
    ``BiclusterModelBase.__init__``.

    Parameters
    ----------
    matrix : np.ndarray  shape (m, n), dtype int (0/1 values)

    Returns
    -------
    rows_data : list of (row_index, row_degree)
    cols_data : list of (col_index, col_degree)
    edges     : list of (row_index, col_index)  for every cell == 1
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
    Compute objective, area, and density for a selected sub-matrix.

    Parameters
    ----------
    matrix      : full input binary matrix
    row_indices : selected row indices (may be unsorted)
    col_indices : selected column indices (may be unsorted)

    Returns
    -------
    objective : number of 1s in the selected sub-matrix
    area      : #selected_rows × #selected_cols
    density   : objective / area  (0.0 when area == 0)
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
    Return ``100 × (best_known − objective) / best_known`` as a percentage.

    Returns ``'NA'`` when:
      - *best_known* is None or ≤ 0
      - *objective* is not an integer (e.g. ``'NA'`` from a failed run)
    """
    if best_known is None or best_known <= 0 or not isinstance(objective, int):
        return "NA"
    return round(100.0 * (best_known - objective) / best_known, 4)
