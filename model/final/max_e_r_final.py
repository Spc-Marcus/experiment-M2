import logging
import gurobipy as gp
from gurobipy import GRB
from model.base import BiclusterModelBase

logger = logging.getLogger(__name__)


class MaxERModel(BiclusterModelBase):
    def __init__(self, rows_data, cols_data, edges, error_rate: float, env=None):
        # Stockage des données
        self.rows_data = rows_data
        self.cols_data = cols_data
        self.edges = set(edges)  # Keep as set for O(1) checking if needed, but we iterate edges mostly
        self.edges_list = list(edges) # Keep ordered list for reproducible iteration
        self.error_rate = float(error_rate)

        logger.info(
            "MaxERModel : initialisation — %d lignes, %d colonnes, %d arêtes, error_rate=%.4f.",
            len(rows_data), len(cols_data), len(self.edges), error_rate,
        )

        # Création du modèle
        if env is not None:
            self.model = gp.Model("max_e_r_final", env=env)
        else:
            self.model = gp.Model("max_e_r_final")
        self.model.setAttr('ModelSense', GRB.MAXIMIZE)

        # Variables
        self.lp_rows = {row: (self.model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'row_{row}'), degree)
                        for row, degree in self.rows_data}
        self.lp_cols = {col: (self.model.addVar(vtype=GRB.BINARY, lb=0, ub=1, name=f'col_{col}'), degree)
                        for col, degree in self.cols_data}
        
        # Cell variables: In MaxER we typically only model the edges we want to recover.
        # MaxER objective is maximizing sum of covered edges.
        self.lp_cells = {}
        for row, col in self.edges_list:
            # Using CONTINUOUS variable for relaxation performance (integrality implied by rows/cols)
            # as seen in max_e_r_grb.py
            var = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f'cell_{row}_{col}')
            self.lp_cells[(row, col)] = var

        # Objectif: Maximize sum of cells (covered edges)
        self.model.setObjective(gp.quicksum(self.lp_cells.values()), GRB.MAXIMIZE)

        # Contraintes de structure
        # cell_{ij} <= row_i AND cell_{ij} <= col_j
        for (row, col), cell_var in self.lp_cells.items():
            if row in self.lp_rows and col in self.lp_cols:
                self.model.addConstr(self.lp_rows[row][0] >= cell_var, name=f'cell_{row}_{col}_1')
                self.model.addConstr(self.lp_cols[col][0] >= cell_var, name=f'cell_{row}_{col}_2')
            else:
                logger.warning(f"Edge ({row}, {col}) references missing row/col variables")

        # Contraintes de densité (Row/Col Density >= 1 - delta)
        self._add_density_constraints()

        # Contraintes de seuil minimal (au moins 1 ligne et 1 colonne)
        self.model.addConstr(gp.quicksum(v for v, _ in self.lp_rows.values()) >= 1, "row_min_1")
        self.model.addConstr(gp.quicksum(v for v, _ in self.lp_cols.values()) >= 1, "col_min_1")

    def _add_density_constraints(self):
        """Add row and column density constraints adapted from max_e_r_grb.py"""
        delta = self.error_rate
        lp_rows = self.lp_rows
        lp_cols = self.lp_cols
        
        # Pre-compute adjacency for efficiency
        row_adj = {r: [] for r, _ in self.rows_data}
        col_adj = {c: [] for c, _ in self.cols_data}
        for r, c in self.edges_list:
            if r in row_adj: row_adj[r].append(c)
            if c in col_adj: col_adj[c].append(r)

        big_m = len(self.rows_data) + len(self.cols_data) + 1
        
        # Row density
        for row, _ in self.rows_data:
            row_edges = row_adj.get(row, [])
            if row_edges:
                self.model.addConstr(
                    gp.quicksum(lp_cols[col][0] for col in row_edges) - 
                    (1 - delta) * gp.quicksum(v for v, _ in lp_cols.values()) >= 
                    (lp_rows[row][0] - 1) * big_m,
                    f"row_density_{row}"
                )
        
        # Col density
        for col, _ in self.cols_data:
            col_edges = col_adj.get(col, [])
            if col_edges:
                self.model.addConstr(
                    gp.quicksum(lp_rows[row][0] for row in col_edges) - 
                    (1 - delta) * gp.quicksum(v for v, _ in lp_rows.values()) >= 
                    (lp_cols[col][0] - 1) * big_m,
                    f"col_density_{col}"
                )

    def add_WarmStart(self, rows, cols):
        logger.debug("MaxERModel : WarmStart fourni (%d lignes, %d colonnes).", len(rows), len(cols))
        rows_set = set(rows)
        cols_set = set(cols)

        # Start variables
        for r, (var_r, _) in self.lp_rows.items():
            var_r.Start = 1 if r in rows_set else 0
        for c, (var_c, _) in self.lp_cols.items():
            var_c.Start = 1 if c in cols_set else 0
        
        # Cell starts
        for (r, c), var_cell in self.lp_cells.items():
            var_cell.Start = 1 if (r in rows_set and c in cols_set) else 0

        self.model.update()

    def optimize(self):
        logger.info("MaxERModel : lancement de l'optimisation.")
        self.model.optimize()
        logger.info(
            "MaxERModel : optimisation terminée — status=%d%s.",
            self.model.Status,
            f", ObjVal={self.model.ObjVal:.4f}" if self.model.Status == GRB.OPTIMAL else "",
        )

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
