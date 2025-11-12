import gurobipy as gp
from gurobipy import GRB


class MaxOneModel:
    def __init__(self, rows_data, cols_data, edges, error_rate: float,env=None):
        # Store data
        self.rows_data = rows_data
        self.cols_data = cols_data
        self.edges = set(edges)
        self.error_rate = float(error_rate)

        # Create model
        if env is not None:
            self.model = gp.Model("max_one_grb_v2",env=env)
        else:
            self.model = gp.Model("max_one_grb_v2")
        self.model.setAttr('ModelSense', GRB.MAXIMIZE)

        # Variables
        self.lp_rows = {row: (self.model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'row_{row}'), degree)
                        for row, degree in self.rows_data}
        self.lp_cols = {col: (self.model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'col_{col}'), degree)
                        for col, degree in self.cols_data}
        self.lp_cells = {}
        for row, _ in self.rows_data:
            for col, _ in self.cols_data:
                var = self.model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'cell_{row}_{col}')
                val = 1 if (row, col) in self.edges else 0
                self.lp_cells[(row, col)] = (var, val)

        # Objective (same as v1)
        self._ones_expr = gp.quicksum(cell_val * var for var, cell_val in self.lp_cells.values())
        self.model.setObjective(self._ones_expr, GRB.MAXIMIZE)

        # Structure constraints (same as v1)
        for (row, col), (cell_var, _) in self.lp_cells.items():
            self.model.addConstr(self.lp_rows[row][0] >= cell_var, name=f'cell_{row}_{col}_1')
            self.model.addConstr(self.lp_cols[col][0] >= cell_var, name=f'cell_{row}_{col}_2')
            self.model.addConstr(self.lp_rows[row][0] + self.lp_cols[col][0] - 1 <= cell_var, name=f'cell_{row}_{col}_3')

        # Error-rate constraint (same as v1)
        self._err_expr = gp.quicksum((1 - cell_val) * var for var, cell_val in self.lp_cells.values())
        self._tot_expr = gp.quicksum(var for var, _ in self.lp_cells.values())
        self._err_rate_constr = self.model.addConstr(self._err_expr <= self.error_rate * self._tot_expr, name='err_rate')

        # Dynamic helpers
        self._forced_row_constrs = {}
        self._forced_col_constrs = {}
        self._improvement_constr = None

    def add_WarmStart(self, rows, cols):
        # Provide a complete and consistent MIP start for all vars (rows, cols, cells)
        rows_set = set(rows)
        cols_set = set(cols)

        # Initialize row/col variable starts to 0 by default, 1 if selected
        for r, (var_r, _) in self.lp_rows.items():
            var_r.start = 1 if r in rows_set else 0
        for c, (var_c, _) in self.lp_cols.items():
            var_c.start = 1 if c in cols_set else 0

        # Initialize cell variable starts consistent with logical AND of row/col selection
        # cell_{r,c} = 1 iff row_r = 1 and col_c = 1
        for (r, c), (var_cell, _) in self.lp_cells.items():
            var_cell.start = 1 if (r in rows_set and c in cols_set) else 0

        # Push starts into the model
        self.model.update()

    
    # Solver helpers
    def optimize(self):
        self.model.optimize()

    def getVars(self):
        return self.model.getVars()

    def setParam(self, param, value):
        self.model.setParam(param, value)

    @property
    def ObjVal(self):
        return self.model.ObjVal

    @property
    def status(self):
        return self.model.Status

    def get_selected_rows(self):
        return [int(v.VarName.split('_')[1]) for v in self.getVars() if v.VarName.startswith('row_') and v.X > 0.5]

    def get_selected_cols(self):
        return [int(v.VarName.split('_')[1]) for v in self.getVars() if v.VarName.startswith('col_') and v.X > 0.5]