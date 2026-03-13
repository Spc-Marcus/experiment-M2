import os
import numpy as np
from typing import List, Tuple
from model.max_one_grb_v3 import MaxOneModel as max_Ones_v3
import contextlib
import sys

# Configuration cloud Gurobi
os.environ['GRB_WLSACCESSID'] = 'X'
os.environ['GRB_WLSSECRET'] = 'X'
os.environ['GRB_LICENSEID'] = 'X'
import gurobipy as grb


@contextlib.contextmanager
def suppress_gurobi_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def find_quasi_biclique_max_one_V3c(
    input_matrix: np.ndarray,
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    V3c: Reconstruction du modèle à chaque phase avec seulement les seeds, SANS WarmStart.
    """
    X_problem = input_matrix.copy()
    cols_sorted = np.argsort(X_problem.sum(axis=0))[::-1]
    rows_sorted = np.argsort(X_problem.sum(axis=1))[::-1]
    m = len(rows_sorted)
    n = len(cols_sorted)
    if m == 0 or n == 0:
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
    
    try:
        with suppress_gurobi_output():
            # PHASE 1: Modèle uniquement sur seed_rows x seed_cols
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

            model = max_Ones_v3(rows_data, cols_data, edges, 0.0)
            model.model.setParam('OutputFlag', 0)
            model.model.setParam('MIPGap', 0.05)
            model.model.setParam('TimeLimit', 20)
            model.model.setParam('Seed', 1)
            model.model.setParam('IntFeasTol', 1e-9)
            model.model.setParam('FeasibilityTol', 1e-9)
            model.model.setParam('OptimalityTol', 1e-9)
            model.model.setParam('NumericFocus', 1)
            model.optimize()
            if model.status != 2:
                return [], [], False
            rw = model.get_selected_rows()
            cl = model.get_selected_cols()

            # PHASE 2: Nouveau modèle pour extension colonnes
            rem_cols = [c for c in cols_sorted if c not in cl]
            if len(rw) > 0:
                rem_cols_sum = X_problem[rw][:, rem_cols].sum(axis=0)
                potential_cols = [c for idx, c in enumerate(rem_cols) if rem_cols_sum[idx] > 0.9 * len(rw)]
            else:
                potential_cols = []

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
                model2 = max_Ones_v3(rows_data, cols_data, edges, error_rate)
                model2.model.setParam('OutputFlag', 0)
                model2.model.setParam('MIPGap', 0.05)
                model2.model.setParam('TimeLimit', 180)
                model2.model.setParam('Seed', 1)
                model2.model.setParam('IntFeasTol', 1e-9)
                model2.model.setParam('FeasibilityTol', 1e-9)
                model2.model.setParam('OptimalityTol', 1e-9)
                model2.model.setParam('NumericFocus', 1)
                model2.optimize()
                if model2.status == 2:
                    rw = model2.get_selected_rows()
                    cl = model2.get_selected_cols()
                else:
                    return rw, cl, True

            # PHASE 3: Nouveau modèle pour extension lignes
            rem_rows = [r for r in rows_sorted if r not in rw]
            if len(cl) > 0:
                rem_rows_sum = X_problem[rem_rows][:, cl].sum(axis=1)
                potential_rows = [r for idx, r in enumerate(rem_rows) if rem_rows_sum[idx] > 0.5 * len(cl)]
            else:
                potential_rows = []

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
                model3 = max_Ones_v3(rows_data, cols_data, edges, error_rate)
                model3.model.setParam('OutputFlag', 0)
                model3.model.setParam('MIPGap', 0.05)
                model3.model.setParam('TimeLimit', 20)
                model3.model.setParam('Seed', 1)
                model3.model.setParam('IntFeasTol', 1e-9)
                model3.model.setParam('FeasibilityTol', 1e-9)
                model3.model.setParam('OptimalityTol', 1e-9)
                model3.model.setParam('NumericFocus', 1)
                model3.optimize()
                if model3.status == 2:
                    rw = model3.get_selected_rows()
                    cl = model3.get_selected_cols()
                else:
                    return rw, cl, True

            return (rw, cl, True) if (rw and cl) else ([], [], False)
    except Exception:
        print("Exception in V3c")
        print(sys.exc_info())
        return [], [], False
