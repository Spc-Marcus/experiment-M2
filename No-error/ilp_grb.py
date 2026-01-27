import os
import numpy as np
from typing import List, Tuple
from model.max_one_grb import max_Ones_gurobi
from model.max_one_grb_v2 import MaxOneModel
from model.max_one_grb_v3 import MaxOneModel as max_Ones_v3
import contextlib
import sys
# Configuration cloud Gurobi : ces variables d'environnement doivent être définies AVANT l'import de gurobipy
os.environ['GRB_WLSACCESSID'] = 'af4b8280-70cd-47bc-aeef-69ecf14ecd10'
os.environ['GRB_WLSSECRET'] = '04da6102-8eb3-4e38-ba06-660ea8f87bf2'
os.environ['GRB_LICENSEID'] = '2669217'
import gurobipy as grb

from model.max_e_r_V2_grb import MaxERModel

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

def find_quasi_dens_matrix_max_ones(
    input_matrix: np.ndarray,
    error_rate: float = 0.025
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique in a binary matrix using integer linear programming optimization (Gurobi).
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
            # --- PHASE 1: SEED ---
            seed_row_indices = rows_sorted
            seed_col_indices = cols_sorted[:seed_cols]
            row_degrees = np.sum(X_problem[seed_row_indices, :][:, seed_col_indices] == 1, axis=1)
            col_degrees = np.sum(X_problem[seed_row_indices, :][:, seed_col_indices] == 1, axis=0)
            rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(seed_row_indices)]
            cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(seed_col_indices)]
            edges = []
            for i, r in enumerate(seed_row_indices):
                for j, c in enumerate(seed_col_indices):
                    if X_problem[r, c] == 1:
                        edges.append((int(r), int(c)))
            
            model = max_Ones_gurobi(rows_data, cols_data, edges, 0)
            model.setParam('OutputFlag', 0)
            model.setParam('MIPGap', 0.05)
            model.setParam('TimeLimit', 20)
            # Deterministic and strict tolerances
            model.setParam('Seed', 1)
            model.setParam('IntFeasTol', 1e-9)
            model.setParam('FeasibilityTol', 1e-9)
            model.setParam('OptimalityTol', 1e-9)
            model.setParam('NumericFocus', 1)
            model.optimize()
            
            status = model.Status
            if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
                return [], [], False
            elif status == grb.GRB.TIME_LIMIT or status == grb.GRB.OPTIMAL:
                rw = []
                cl = []
                for v in model.getVars():
                    if v.VarName.startswith('row_') and v.X > 0.5:
                        rw.append(int(v.VarName.split('_')[1]))
                    elif v.VarName.startswith('col_') and v.X > 0.5:
                        cl.append(int(v.VarName.split('_')[1]))
            else:
                return [], [], False
            
            # --- PHASE 2: EXTENSION COLONNES ---
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
                for i, r in enumerate(rw):
                    for j, c in enumerate(all_col_indices):
                        if X_problem[r, c] == 1:
                            edges.append((int(r), int(c)))
                
                model = max_Ones_gurobi(rows_data, cols_data, edges, error_rate)
                model.setParam('OutputFlag', 0)
                model.setParam('MIPGap', 0.05)
                model.setParam('TimeLimit', 180)
                # Deterministic and strict tolerances
                model.setParam('Seed', 1)
                model.setParam('IntFeasTol', 1e-9)
                model.setParam('FeasibilityTol', 1e-9)
                model.setParam('OptimalityTol', 1e-9)
                model.setParam('NumericFocus', 1)
                model.optimize()
                
                status = model.Status
                if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
                    return [], [], False
                elif status == grb.GRB.TIME_LIMIT or status == grb.GRB.OPTIMAL:
                    rw = []
                    cl = []
                    for v in model.getVars():
                        if v.VarName.startswith('row_') and v.X > 0.5:
                            rw.append(int(v.VarName.split('_')[1]))
                        elif v.VarName.startswith('col_') and v.X > 0.5:
                            cl.append(int(v.VarName.split('_')[1]))
                else:
                    return [], [], False
            
            # --- PHASE 3: EXTENSION LIGNES ---
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
                for i, r in enumerate(all_row_indices):
                    for j, c in enumerate(cl):
                        if X_problem[r, c] == 1:
                            edges.append((int(r), int(c)))
                
                model = max_Ones_gurobi(rows_data, cols_data, edges, error_rate)
                model.setParam('OutputFlag', 0)
                model.setParam('MIPGap', 0.05)
                model.setParam('TimeLimit', 20)
                # Deterministic and strict tolerances
                model.setParam('Seed', 1)
                model.setParam('IntFeasTol', 1e-9)
                model.setParam('FeasibilityTol', 1e-9)
                model.setParam('OptimalityTol', 1e-9)
                model.setParam('NumericFocus', 1)
                model.optimize()
                
                status = model.Status
                if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
                    return [], [], False
                elif status == grb.GRB.TIME_LIMIT or status == grb.GRB.OPTIMAL:
                    rw = []
                    cl = []
                    for v in model.getVars():
                        if v.VarName.startswith('row_') and v.X > 0.5:
                            rw.append(int(v.VarName.split('_')[1]))
                        elif v.VarName.startswith('col_') and v.X > 0.5:
                            cl.append(int(v.VarName.split('_')[1]))
                else:
                    return [], [], False
            
            status = model.Status
            if status in (grb.GRB.INF_OR_UNBD, grb.GRB.INFEASIBLE, grb.GRB.UNBOUNDED):
                return [], [], False
            elif status == grb.GRB.TIME_LIMIT:
                return rw, cl, True
            elif status != grb.GRB.OPTIMAL:
                return [], [], False
            return rw, cl, True
    except Exception:
        return [], [], False

def find_quasi_biclique_max_one_V2(
    input_matrix: np.ndarray,
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique using MaxOneModel.
    """
    X_problem = input_matrix.copy()
    cols_sorted = np.argsort(X_problem.sum(axis=0))[::-1]
    rows_sorted = np.argsort(X_problem.sum(axis=1))[::-1]
    m = len(rows_sorted)
    n = len(cols_sorted)
    if m == 0 or n == 0:
        return [], [], False
    # Compute seed_cols similarly to V1
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
            # --- PHASE 1: SEED (identique à V1: lignes triées, seed_col_indices) ---
            # Construire modèle uniquement sur seed_rows x seed_cols
            row_degrees = np.sum(X_problem[rows_sorted, :][:, cols_sorted] == 1, axis=1)
            col_degrees = np.sum(X_problem[rows_sorted, :][:, cols_sorted] == 1, axis=0)
            rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(rows_sorted)]
            cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(cols_sorted)]
            edges = []
            for r in rows_sorted:
                for c in cols_sorted:
                    if X_problem[r, c] == 1:
                        edges.append((int(r), int(c)))

            model = MaxOneModel(rows_data, cols_data, edges, 0.0)
            model.model.setParam('OutputFlag', 0)
            model.model.setParam('MIPGap', 0.05)
            model.model.setParam('TimeLimit', 20)
            # Deterministic and strict tolerances
            model.model.setParam('Seed', 1)
            model.model.setParam('IntFeasTol', 1e-9)
            model.model.setParam('FeasibilityTol', 1e-9)
            model.model.setParam('OptimalityTol', 1e-9)
            model.model.setParam('NumericFocus', 1)
            model.add_forced_cols_zero(cols_sorted[seed_cols:])
            model.optimize()
            if model.status != 2:
                return [], [], False
            rw = model.get_selected_rows()
            cl = model.get_selected_cols()

            # --- PHASE 2: EXTENSION COLONNES (identique à V1) ---
            rem_cols = [c for c in cols_sorted if c not in cl]
            if len(rw) > 0:
                rem_cols_sum = X_problem[rw][:, rem_cols].sum(axis=0)
                potential_cols = [c for idx, c in enumerate(rem_cols) if rem_cols_sum[idx] > 0.9 * len(rw)]
            else:
                potential_cols = []

            if potential_cols:
                all_col_indices = cl + potential_cols
                select_rows = rw
                model.remove_forced_cols_zero(cols_sorted[seed_cols:])
                model.add_forced_cols_zero([c for c in range(n) if c not in all_col_indices])
                model.add_forced_rows_zero([r for r in range(m) if r not in select_rows])
                model.set_error_rate(error_rate)
                model.model.setParam('TimeLimit', 180)
                model.optimize()
                if model.status == 2:
                    rw = model.get_selected_rows()
                    cl = model.get_selected_cols()
                else:
                    return rw, cl, True

            # --- PHASE 3: EXTENSION LIGNES (identique à V1) ---
            rem_rows = [r for r in rows_sorted if r not in rw]
            if len(cl) > 0:
                rem_rows_sum = X_problem[rem_rows][:, cl].sum(axis=1)
                potential_rows = [r for idx, r in enumerate(rem_rows) if rem_rows_sum[idx] > 0.5 * len(cl)]
            else:
                potential_rows = []
            if potential_rows:
                all_row_indices = rw + potential_rows
                select_cols = cl
                model.remove_forced_rows_zero([r for r in range(m) if r not in rw])
                model.remove_forced_cols_zero([c for c in range(n) if c not in cl])
                model.add_forced_rows_zero([r for r in range(m) if r not in all_row_indices])
                model.add_forced_cols_zero([c for c in range(n) if c not in select_cols])
                model.set_error_rate(error_rate)
                model.model.setParam('TimeLimit', 180)
                model.optimize()
                if model.status == 2:
                    rw = model.get_selected_rows()
                    cl = model.get_selected_cols()
                else:
                    return rw, cl, True
            return (rw, cl, True) if (rw and cl) else ([], [], False)
    except Exception:
        print("Exception in Gurobi optimization")
        print(sys.exc_info())
        return [], [], False
    


def find_quasi_biclique_max_e_r_V2(
    input_matrix: np.ndarray,
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique using MaxERModel.
    """
    X_problem = input_matrix.copy()
    n_rows, n_cols = X_problem.shape
    if n_rows == 0 or n_cols == 0:
        return [], [], False

    try:
        with suppress_gurobi_output():
            row_degrees = np.sum(X_problem == 1, axis=1)
            rows_data = [(r, int(row_degrees[r])) for r in range(n_rows)]
            col_degrees = np.sum(X_problem == 1, axis=0)
            cols_data = [(int(c), int(col_degrees[c])) for c in range(n_cols)]
            edges = []
            cols_sums = X_problem.sum(axis=0)
            cols_sorted = np.argsort(cols_sums)[::-1]
            seed_cols = min(max(n_cols // 3, min(n_cols, 5)), 50)
            no_use_cols_seed = cols_sorted[seed_cols:]

            for r in range(n_rows):
                for c in range(n_cols):
                    if X_problem[r, c] == 1:
                        edges.append((int(r), int(c)))
            
            model = MaxERModel(rows_data, cols_data, edges)
            model.setParam('OutputFlag', 0)
            model.build_max_e_r(3, 2)
            model.add_density_constraints(0)
            model.add_forced_cols_zero(no_use_cols_seed)
            model.optimize()

            if model.status == 2:
                rw = model.get_selected_rows()
                cl = model.get_selected_cols()
            else:
                return [], [], False
            
            # PHASE 2: EXTENSION COLONNES
            no_use_rows_seed = [r for r in range(n_rows) if r not in rw]
            model.remove_forced_cols_zero(no_use_cols_seed)
            model.add_forced_rows_zero(no_use_rows_seed)
            model.add_improvement_constraint(model.ObjVal)
            model.update_density_constraints(error_rate)
            model.optimize()
            
            if model.status == 2:
                rw = model.get_selected_rows()
                cl = model.get_selected_cols()
            else:
                return rw, cl, True if (rw and cl) else ([], [], False)
            
            # PHASE 3: EXTENSION LIGNES (relâchement supplémentaire)
            no_use_cols_phase3 = [c for c in range(n_cols) if c not in cl]
            model.remove_forced_rows_zero(no_use_rows_seed)
            model.add_forced_cols_zero(no_use_cols_phase3)
            model.add_improvement_constraint(model.ObjVal)
            model.setParam('TimeLimit', 180)
            model.optimize()
            
            if model.status == 2:
                rw = model.get_selected_rows()
                cl = model.get_selected_cols()
            
            if rw and cl:
                return rw, cl, True
            return [], [], False
    except Exception:
        return [], [], False    


def find_quasi_biclique_max_one_V3(
    input_matrix: np.ndarray,
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    Find a quasi-biclique using MaxOneModel (V3).
    
    Cette version utilise la même approche que V2 (un seul modèle avec contraintes
    dynamiques) mais avec le WarmStart comme optimisation supplémentaire.
    Cela garantit des résultats identiques à V1/V2 tout en bénéficiant du WarmStart.
    """
    X_problem = input_matrix.copy()
    cols_sorted = np.argsort(X_problem.sum(axis=0))[::-1]
    rows_sorted = np.argsort(X_problem.sum(axis=1))[::-1]
    m = len(rows_sorted)
    n = len(cols_sorted)
    if m == 0 or n == 0:
        return [], [], False
    # Compute seed_cols similarly to V1
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
            # --- CONSTRUIRE UN SEUL MODÈLE AVEC TOUTES LES DONNÉES (comme V2) ---
            row_degrees = np.sum(X_problem[rows_sorted, :][:, cols_sorted] == 1, axis=1)
            col_degrees = np.sum(X_problem[rows_sorted, :][:, cols_sorted] == 1, axis=0)
            rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(rows_sorted)]
            cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(cols_sorted)]
            edges = []
            for r in rows_sorted:
                for c in cols_sorted:
                    if X_problem[r, c] == 1:
                        edges.append((int(r), int(c)))

            model = max_Ones_v3(rows_data, cols_data, edges, 0.0)
            model.model.setParam('OutputFlag', 0)
            model.model.setParam('MIPGap', 0.05)
            model.model.setParam('TimeLimit', 20)
            # Deterministic and strict tolerances
            model.model.setParam('Seed', 1)
            model.model.setParam('IntFeasTol', 1e-9)
            model.model.setParam('FeasibilityTol', 1e-9)
            model.model.setParam('OptimalityTol', 1e-9)
            model.model.setParam('NumericFocus', 1)
            
            # --- PHASE 1: SEED - Forcer les colonnes hors seed à zéro ---
            cols_to_force_zero = list(cols_sorted[seed_cols:])
            for col in cols_to_force_zero:
                if col in model.lp_cols:
                    model._forced_col_constrs[col] = model.model.addConstr(
                        model.lp_cols[col][0] == 0, name=f"force_col_{col}_zero")
            
            model.optimize()
            if model.status != 2:
                return [], [], False
            rw = model.get_selected_rows()
            cl = model.get_selected_cols()

            # --- PHASE 2: EXTENSION COLONNES (identique à V2) ---
            rem_cols = [c for c in cols_sorted if c not in cl]
            if len(rw) > 0:
                rem_cols_sum = X_problem[rw][:, rem_cols].sum(axis=0)
                potential_cols = [c for idx, c in enumerate(rem_cols) if rem_cols_sum[idx] > 0.9 * len(rw)]
            else:
                potential_cols = []

            if potential_cols:
                all_col_indices = cl + potential_cols
                select_rows = rw
                
                # Libérer les anciennes contraintes sur colonnes
                for col in list(model._forced_col_constrs.keys()):
                    model.model.remove(model._forced_col_constrs[col])
                    del model._forced_col_constrs[col]
                
                # Forcer les colonnes hors all_col_indices à zéro
                for c in cols_sorted:
                    if c not in all_col_indices and c not in model._forced_col_constrs:
                        if c in model.lp_cols:
                            model._forced_col_constrs[c] = model.model.addConstr(
                                model.lp_cols[c][0] == 0, name=f"force_col_{c}_zero")
                
                # Forcer les lignes hors select_rows à zéro
                for r in rows_sorted:
                    if r not in select_rows and r not in model._forced_row_constrs:
                        if r in model.lp_rows:
                            model._forced_row_constrs[r] = model.model.addConstr(
                                model.lp_rows[r][0] == 0, name=f"force_row_{r}_zero")
                
                # Mettre à jour error_rate
                model.error_rate = error_rate
                if hasattr(model, '_err_rate_constr') and model._err_rate_constr is not None:
                    model.model.remove(model._err_rate_constr)
                model._err_rate_constr = model.model.addConstr(
                    model._err_expr <= model.error_rate * model._tot_expr, name='err_rate')
                
                model.model.setParam('TimeLimit', 180)
                model.add_WarmStart(rw, cl)
                model.optimize()
                
                if model.status == 2:
                    rw = model.get_selected_rows()
                    cl = model.get_selected_cols()
                else:
                    return rw, cl, True

            # --- PHASE 3: EXTENSION LIGNES (identique à V2) ---
            rem_rows = [r for r in rows_sorted if r not in rw]
            if len(cl) > 0:
                rem_rows_sum = X_problem[rem_rows][:, cl].sum(axis=1)
                potential_rows = [r for idx, r in enumerate(rem_rows) if rem_rows_sum[idx] > 0.5 * len(cl)]
            else:
                potential_rows = []
            
            if potential_rows:
                all_row_indices = rw + potential_rows
                select_cols = cl
                
                # Libérer les anciennes contraintes sur lignes
                for r in list(model._forced_row_constrs.keys()):
                    model.model.remove(model._forced_row_constrs[r])
                    del model._forced_row_constrs[r]
                
                # Libérer les anciennes contraintes sur colonnes
                for c in list(model._forced_col_constrs.keys()):
                    model.model.remove(model._forced_col_constrs[c])
                    del model._forced_col_constrs[c]
                
                # Forcer les lignes hors all_row_indices à zéro
                for r in rows_sorted:
                    if r not in all_row_indices and r not in model._forced_row_constrs:
                        if r in model.lp_rows:
                            model._forced_row_constrs[r] = model.model.addConstr(
                                model.lp_rows[r][0] == 0, name=f"force_row_{r}_zero")
                
                # Forcer les colonnes hors select_cols à zéro
                for c in cols_sorted:
                    if c not in select_cols and c not in model._forced_col_constrs:
                        if c in model.lp_cols:
                            model._forced_col_constrs[c] = model.model.addConstr(
                                model.lp_cols[c][0] == 0, name=f"force_col_{c}_zero")
                
                model.model.setParam('TimeLimit', 20)
                model.add_WarmStart(rw, cl)
                model.optimize()
                
                if model.status == 2:
                    rw = model.get_selected_rows()
                    cl = model.get_selected_cols()
                else:
                    return rw, cl, True

            return (rw, cl, True) if (rw and cl) else ([], [], False)
    except Exception:
        print("Exception in Gurobi optimization")
        print(sys.exc_info())
        return [], [], False


# =============================================================================
# VARIANTES V3 POUR TESTER DIFFÉRENTES STRATÉGIES D'OPTIMISATION
# =============================================================================

def find_quasi_biclique_max_one_V3a(
    input_matrix: np.ndarray,
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    V3a: Un seul modèle avec contraintes dynamiques, SANS WarmStart.
    
    Stratégie: Réutilise le même modèle Gurobi à travers les phases,
    utilise des contraintes pour forcer les variables à zéro.
    Pas de WarmStart pour comparer les performances.
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
            # Construire un seul modèle avec toutes les données
            row_degrees = np.sum(X_problem[rows_sorted, :][:, cols_sorted] == 1, axis=1)
            col_degrees = np.sum(X_problem[rows_sorted, :][:, cols_sorted] == 1, axis=0)
            rows_data = [(int(r), int(row_degrees[i])) for i, r in enumerate(rows_sorted)]
            cols_data = [(int(c), int(col_degrees[i])) for i, c in enumerate(cols_sorted)]
            edges = []
            for r in rows_sorted:
                for c in cols_sorted:
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
            
            # PHASE 1: SEED - Forcer les colonnes hors seed à zéro
            cols_to_force_zero = list(cols_sorted[seed_cols:])
            for col in cols_to_force_zero:
                if col in model.lp_cols:
                    model._forced_col_constrs[col] = model.model.addConstr(
                        model.lp_cols[col][0] == 0, name=f"force_col_{col}_zero")
            
            model.optimize()
            if model.status != 2:
                return [], [], False
            rw = model.get_selected_rows()
            cl = model.get_selected_cols()

            # PHASE 2: EXTENSION COLONNES
            rem_cols = [c for c in cols_sorted if c not in cl]
            if len(rw) > 0:
                rem_cols_sum = X_problem[rw][:, rem_cols].sum(axis=0)
                potential_cols = [c for idx, c in enumerate(rem_cols) if rem_cols_sum[idx] > 0.9 * len(rw)]
            else:
                potential_cols = []

            if potential_cols:
                all_col_indices = cl + potential_cols
                select_rows = rw
                
                for col in list(model._forced_col_constrs.keys()):
                    model.model.remove(model._forced_col_constrs[col])
                    del model._forced_col_constrs[col]
                
                for c in cols_sorted:
                    if c not in all_col_indices and c not in model._forced_col_constrs:
                        if c in model.lp_cols:
                            model._forced_col_constrs[c] = model.model.addConstr(
                                model.lp_cols[c][0] == 0, name=f"force_col_{c}_zero")
                
                for r in rows_sorted:
                    if r not in select_rows and r not in model._forced_row_constrs:
                        if r in model.lp_rows:
                            model._forced_row_constrs[r] = model.model.addConstr(
                                model.lp_rows[r][0] == 0, name=f"force_row_{r}_zero")
                
                model.error_rate = error_rate
                if hasattr(model, '_err_rate_constr') and model._err_rate_constr is not None:
                    model.model.remove(model._err_rate_constr)
                model._err_rate_constr = model.model.addConstr(
                    model._err_expr <= model.error_rate * model._tot_expr, name='err_rate')
                
                model.model.setParam('TimeLimit', 180)
                # PAS DE WARMSTART
                model.optimize()
                
                if model.status == 2:
                    rw = model.get_selected_rows()
                    cl = model.get_selected_cols()
                else:
                    return rw, cl, True

            # PHASE 3: EXTENSION LIGNES
            rem_rows = [r for r in rows_sorted if r not in rw]
            if len(cl) > 0:
                rem_rows_sum = X_problem[rem_rows][:, cl].sum(axis=1)
                potential_rows = [r for idx, r in enumerate(rem_rows) if rem_rows_sum[idx] > 0.5 * len(cl)]
            else:
                potential_rows = []
            
            if potential_rows:
                all_row_indices = rw + potential_rows
                select_cols = cl
                
                for r in list(model._forced_row_constrs.keys()):
                    model.model.remove(model._forced_row_constrs[r])
                    del model._forced_row_constrs[r]
                
                for c in list(model._forced_col_constrs.keys()):
                    model.model.remove(model._forced_col_constrs[c])
                    del model._forced_col_constrs[c]
                
                for r in rows_sorted:
                    if r not in all_row_indices and r not in model._forced_row_constrs:
                        if r in model.lp_rows:
                            model._forced_row_constrs[r] = model.model.addConstr(
                                model.lp_rows[r][0] == 0, name=f"force_row_{r}_zero")
                
                for c in cols_sorted:
                    if c not in select_cols and c not in model._forced_col_constrs:
                        if c in model.lp_cols:
                            model._forced_col_constrs[c] = model.model.addConstr(
                                model.lp_cols[c][0] == 0, name=f"force_col_{c}_zero")
                
                model.model.setParam('TimeLimit', 20)
                # PAS DE WARMSTART
                model.optimize()
                
                if model.status == 2:
                    rw = model.get_selected_rows()
                    cl = model.get_selected_cols()
                else:
                    return rw, cl, True

            return (rw, cl, True) if (rw and cl) else ([], [], False)
    except Exception:
        print("Exception in V3a")
        print(sys.exc_info())
        return [], [], False


def find_quasi_biclique_max_one_V3b(
    input_matrix: np.ndarray,
    error_rate: float = 0.025,
) -> Tuple[List[int], List[int], bool]:
    """
    V3b: Reconstruction du modèle à chaque phase avec seulement les seeds, AVEC WarmStart.
    
    Stratégie: Crée un nouveau modèle à chaque phase avec uniquement
    les lignes/colonnes potentielles. Utilise WarmStart pour accélérer.
    C'est l'ancienne version V3 originale.
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
                model2.add_WarmStart(rw, cl)  # AVEC WARMSTART
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
                model3.add_WarmStart(rw, cl)  # AVEC WARMSTART
                model3.optimize()
                if model3.status == 2:
                    rw = model3.get_selected_rows()
                    cl = model3.get_selected_cols()
                else:
                    return rw, cl, True

            return (rw, cl, True) if (rw and cl) else ([], [], False)
    except Exception:
        print("Exception in V3b")
        print(sys.exc_info())
        return [], [], False


def find_quasi_biclique_max_one_V3c(
    input_matrix: np.ndarray,
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
                # PAS DE WARMSTART
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
                # PAS DE WARMSTART
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
    