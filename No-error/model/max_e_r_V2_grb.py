import gurobipy as gp
from gurobipy import GRB

class MaxERModel:
    """
    Wrapper pour le problème de maximum edge recovery (max_e_r et max_e_wr),
    qui encapsule un gp.Model et expose les méthodes principales de Gurobi.
    """
    def __init__(self, rows_data, cols_data, edges, model_name="max_e_r_v2"):
        self.rows_data = rows_data
        self.cols_data = cols_data
        self.edges = edges
        self.model = gp.Model(model_name)
        self.model.setAttr('ModelSense', GRB.MAXIMIZE)
        self.lp_rows = self._create_row_variables()
        self.lp_cols = self._create_col_variables()
        self.lp_cells = self._create_cell_variables()
        self._is_built = False
        self._forced_row_constrs = {}
        self._forced_col_constrs = {}
        self._improvement_constr = None
        self._density_constrs = []

    def _create_row_variables(self):
        return {
            row: (self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=1, name=f'row_{row}'), degree)
            for row, degree in self.rows_data
        }

    def _create_col_variables(self):
        return {
            col: (self.model.addVar(vtype=GRB.INTEGER, lb=0, ub=1, name=f'col_{col}'), degree)
            for col, degree in self.cols_data
        }

    def _create_cell_variables(self):
        return {
            (row, col): self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'cell_{row}_{col}')
            for row, col in self.edges
        }

    def add_forced_rows_zero(self, rows):
        for row in rows:
            if row in self.lp_rows and row not in self._forced_row_constrs:
                constr = self.model.addConstr(self.lp_rows[row][0] == 0, f"force_row_{row}_zero")
                self._forced_row_constrs[row] = constr

    def remove_forced_rows_zero(self, rows):
        for row in rows:
            if row in self._forced_row_constrs:
                self.model.remove(self._forced_row_constrs[row])
                del self._forced_row_constrs[row]

    def add_forced_cols_zero(self, cols):
        for col in cols:
            if col in self.lp_cols and col not in self._forced_col_constrs:
                constr = self.model.addConstr(self.lp_cols[col][0] == 0, f"force_col_{col}_zero")
                self._forced_col_constrs[col] = constr

    def remove_forced_cols_zero(self, cols):
        for col in cols:
            if col in self._forced_col_constrs:
                self.model.remove(self._forced_col_constrs[col])
                del self._forced_col_constrs[col]

    def add_improvement_constraint(self, prev_obj):
        if self._improvement_constr is not None:
            self.model.remove(self._improvement_constr)
        self._improvement_constr = self.model.addConstr(gp.quicksum(self.lp_cells.values()) >= prev_obj , "improvement")

    def remove_improvement_constraint(self):
        if self._improvement_constr is not None:
            self.model.remove(self._improvement_constr)
            self._improvement_constr = None

    def _add_threshold_constraints(self, row_threshold, col_threshold):
        self.model.addConstr(gp.quicksum(lpvar for lpvar, _ in self.lp_rows.values()) >= row_threshold, "row_threshold")
        self.model.addConstr(gp.quicksum(lpvar for lpvar, _ in self.lp_cols.values()) >= col_threshold, "col_threshold")

    def _add_matrix_structure_constraints(self):
        for row, col in self.edges:
            if row in self.lp_rows and col in self.lp_cols and (row, col) in self.lp_cells:
                self.model.addConstr(self.lp_rows[row][0] >= self.lp_cells[(row, col)], f'cell_{row}_{col}_1')
                self.model.addConstr(self.lp_cols[col][0] >= self.lp_cells[(row, col)], f'cell_{row}_{col}_2')

    def add_density_constraints(self, delta):
        self._density_constrs = []
        big_m = len(self.rows_data) + len(self.cols_data) + 1
        for col, _ in self.cols_data:
            col_edges = [u for u, v in self.edges if v == col]
            if col_edges:
                constr = self.model.addConstr(
                    gp.quicksum(self.lp_rows[row][0] for row in col_edges) -
                    (1 - delta) * gp.quicksum(self.lp_rows[row][0] for row, _ in self.rows_data) >=
                    (self.lp_cols[col][0] - 1) * big_m,
                    f"col_density_{col}"
                )
                self._density_constrs.append(constr)

    def remove_density_constraints(self):
        for constr in self._density_constrs:
            self.model.remove(constr)
        self._density_constrs = []

    def update_density_constraints(self, new_delta):
        self.remove_density_constraints()
        self.add_density_constraints(new_delta)

    def build_max_e_r(self, row_threshold=1, col_threshold=1, delta=0.0):
        if self._is_built:
            raise RuntimeError("Le modèle a déjà été construit.")
        self.model.setObjective(gp.quicksum(self.lp_cells.values()), GRB.MAXIMIZE)
        self._add_threshold_constraints(row_threshold, col_threshold)
        self._add_matrix_structure_constraints()
        self.add_density_constraints(delta)
        self._is_built = True

    def optimize(self):
        self.model.optimize()

    def getVars(self):
        return self.model.getVars()

    def setParam(self, *args, **kwargs):
        return self.model.setParam(*args, **kwargs)

    @property
    def status(self):
        return self.model.status

    @property
    def ObjVal(self):
        return self.model.ObjVal

    def get_selected_rows(self):
        return [int(v.VarName.split('_')[1]) for v in self.getVars() if v.VarName.startswith('row_') and v.X > 0.5]

    def get_selected_cols(self):
        return [int(v.VarName.split('_')[1]) for v in self.getVars() if v.VarName.startswith('col_') and v.X > 0.5]

    def reset_model(self):
        self.model.remove(self.model.getConstrs())
        self.model.remove(self.model.getVars())
        self.lp_rows = self._create_row_variables()
        self.lp_cols = self._create_col_variables()
        self.lp_cells = self._create_cell_variables()
        self._forced_row_constrs = {}
        self._forced_col_constrs = {}
        self._improvement_constr = None
        self._is_built = False