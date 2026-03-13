import gurobipy as gp
from gurobipy import GRB

class MaxERSolver:
    """Solver for maximum edge recovery problems with density constraints."""
    
    def __init__(self, debug_level=0):
        self.debug = debug_level
    
    def create_base_model(self, rows_data, cols_data, edges, model_name):
        """Create base model with variables and basic structure."""
        model = gp.Model(model_name)
        model.setAttr('ModelSense', GRB.MAXIMIZE)
        
        # Create variables
        lp_rows = self._create_row_variables(model, rows_data)
        lp_cols = self._create_col_variables(model, cols_data)
        lp_cells = self._create_cell_variables(model, edges)
        
        # Set objective
        model.setObjective(gp.quicksum(lp_cells.values()), GRB.MAXIMIZE)
        
        return model, lp_rows, lp_cols, lp_cells
    
    def _create_row_variables(self, model, rows_data):
        """Create row variables."""
        return {
            row: (model.addVar(vtype=GRB.INTEGER, lb=0, ub=1, name=f'row_{row}'), degree)
            for row, degree in rows_data
        }
    
    def _create_col_variables(self, model, cols_data):
        """Create column variables."""
        return {
            col: (model.addVar(vtype=GRB.INTEGER, lb=0, ub=1, name=f'col_{col}'), degree)
            for col, degree in cols_data
        }
    
    def _create_cell_variables(self, model, edges):
        """Create cell variables."""
        return {
            (row, col): model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'cell_{row}_{col}')
            for row, col in edges
        }
    
    def _add_threshold_constraints(self, model, lp_rows, lp_cols, row_threshold, col_threshold):
        """Add row and column threshold constraints."""
        model.addConstr(gp.quicksum(lpvar for lpvar, _ in lp_rows.values()) >= row_threshold, "row_threshold")
        model.addConstr(gp.quicksum(lpvar for lpvar, _ in lp_cols.values()) >= col_threshold, "col_threshold")
    
    def _add_matrix_structure_constraints(self, model, edges, lp_rows, lp_cols, lp_cells):
        """Add constraints ensuring matrix structure consistency."""
        missing_cols = {col for _, col in edges} - {col for col, _ in lp_cols.items()}
        if missing_cols and self.debug >= 1:
            print(f"Warning: Missing columns in cols_data: {missing_cols}")
        
        for row, col in edges:
            if row in lp_rows and col in lp_cols and (row, col) in lp_cells:
                model.addConstr(lp_rows[row][0] >= lp_cells[(row, col)], f'cell_{row}_{col}_1')
                model.addConstr(lp_cols[col][0] >= lp_cells[(row, col)], f'cell_{row}_{col}_2')
            elif self.debug >= 1:
                print(f"Warning: row={row}, col={col} not found in variables!")
    
    def _add_density_constraints(self, model, rows_data, cols_data, edges, lp_rows, lp_cols, delta):
        """Add row and column density constraints."""
        self._add_row_density_constraints(rows_data, cols_data, edges, model, lp_rows, lp_cols, delta)
        self._add_col_density_constraints(rows_data, cols_data, edges, model, lp_rows, lp_cols, delta)
    
    def _print_debug_info(self, model_name, rows_data, cols_data, edges, delta, thresholds=None):
        """Print debug information."""
        if self.debug >= 3:
            print('\n' + '-' * 40)
            print(f"******** Solving model {model_name} with delta = {delta} ********")
            print(f"# rows_data = {len(rows_data)}, # cols_data = {len(cols_data)}, # edges = {len(edges)}")
            if thresholds:
                print(f"row_threshold = {thresholds[0]}, col_threshold = {thresholds[1]}")
            print('-' * 40)
    
    def max_e_r(self, rows_data, cols_data, edges, delta):
        """
        Create the basic maximum edge recovery model.
        
        Arguments:
        ----------
        rows_data: list of tuples (row, degree) of rows in the matrix.
        cols_data: list of tuples (col, degree) of columns in the matrix.
        edges: list of tuples (row, col) corresponding to the ones of the matrix.
        delta: float, error tolerance for density constraints.

        Returns:
        --------
        gp.Model: The ILP model.
        """
        model, lp_rows, lp_cols, lp_cells = self.create_base_model(
            rows_data, cols_data, edges, "max_e_r"
        )
        
        # Set thresholds
        row_threshold, col_threshold = 1, 1
        
        self._print_debug_info("max_e_r", rows_data, cols_data, edges, delta, (row_threshold, col_threshold))
        
        # Add constraints
        self._add_threshold_constraints(model, lp_rows, lp_cols, row_threshold, col_threshold)
        self._add_matrix_structure_constraints(model, edges, lp_rows, lp_cols, lp_cells)
        self._add_density_constraints(model, rows_data, cols_data, edges, lp_rows, lp_cols, delta)
        
        return model
    
    def max_e_wr(self, rows_data, cols_data, edges, rows_res, cols_res, prev_obj, delta):
        """
        Create the warm-start maximum edge recovery model.
        
        Arguments:
        ----------
        rows_data: list of tuples (row, degree) of rows in the matrix.
        cols_data: list of tuples (col, degree) of columns in the matrix.
        edges: list of tuples (row, col) corresponding to the ones of the matrix.
        rows_res: list of indices of rows (input solution)
        cols_res: list of indices of cols (input solution)
        prev_obj: previous objective value for improvement constraint
        delta: float, error tolerance for density constraints.

        Returns:
        --------
        gp.Model: The ILP model with warm start.
        """
        model, lp_rows, lp_cols, lp_cells = self.create_base_model(
            rows_data, cols_data, edges, "max_e_wr"
        )
        
        # Set thresholds
        row_threshold, col_threshold = 2, 2
        
        self._print_debug_info("max_e_wr", rows_data, cols_data, edges, delta, (row_threshold, col_threshold))
        
        # Set warm start values
        self._set_warm_start_values(lp_rows, lp_cols, lp_cells, rows_res, cols_res)
        
        # Add constraints
        self._add_threshold_constraints(model, lp_rows, lp_cols, row_threshold, col_threshold)
        self._add_improvement_constraint(model, lp_cells, prev_obj)
        self._add_matrix_structure_constraints(model, edges, lp_rows, lp_cols, lp_cells)
        self._add_density_constraints(model, rows_data, cols_data, edges, lp_rows, lp_cols, delta)
        
        return model
    
    def _set_warm_start_values(self, lp_rows, lp_cols, lp_cells, rows_res, cols_res):
        """Set initial values for warm start."""
        rows_res_set = set(map(int, rows_res))
        cols_res_set = set(map(int, cols_res))
        
        if self.debug >= 2:
            print(f"Setting warm start with {len(rows_res_set)} rows and {len(cols_res_set)} columns")
        
        # Set row initial values
        row_initial_values = {}
        for row, (var, _) in lp_rows.items():
            val = 1 if row in rows_res_set else 0
            var.Start = val
            row_initial_values[row] = val
        
        # Set column initial values
        col_initial_values = {}
        for col, (var, _) in lp_cols.items():
            val = 1 if col in cols_res_set else 0
            var.Start = val
            col_initial_values[col] = val
        
        # Set cell initial values
        for (row, col), var in lp_cells.items():
            val = 1 if row in rows_res_set and col in cols_res_set else 0
            var.Start = val
        
        if self.debug >= 3:
            print("\nInitial point before solving:")
            print(f"ROWS: {row_initial_values}")
            print(f"COLS: {col_initial_values}")
    
    def _add_improvement_constraint(self, model, lp_cells, prev_obj):
        """Add constraint for objective improvement."""
        model.addConstr(gp.quicksum(lp_cells.values()) >= prev_obj + 1, "improvement")
    
    def _add_row_density_constraints(self, rows_data, cols_data, edges, model, lp_rows, lp_cols, delta):
        """Add row density constraints to the model."""
        big_m = len(rows_data) + len(cols_data) + 1
        
        for row, _ in rows_data:
            row_edges = [v for u, v in edges if u == row]
            if row_edges:  # Only add constraint if row has edges
                model.addConstr(
                    gp.quicksum(lp_cols[col][0] for col in row_edges) - 
                    (1 - delta) * gp.quicksum(lp_cols[col][0] for col, _ in cols_data) >= 
                    (lp_rows[row][0] - 1) * big_m,
                    f"row_density_{row}"
                )
    
    def _add_col_density_constraints(self, rows_data, cols_data, edges, model, lp_rows, lp_cols, delta):
        """Add column density constraints to the model."""
        big_m = len(rows_data) + len(cols_data) + 1
        
        for col, _ in cols_data:
            col_edges = [u for u, v in edges if v == col]
            if col_edges:  # Only add constraint if column has edges
                model.addConstr(
                    gp.quicksum(lp_rows[row][0] for row in col_edges) - 
                    (1 - delta) * gp.quicksum(lp_rows[row][0] for row, _ in rows_data) >= 
                    (lp_cols[col][0] - 1) * big_m,
                    f"col_density_{col}"
                )
