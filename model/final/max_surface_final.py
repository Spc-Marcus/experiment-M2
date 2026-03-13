import logging

import gurobipy as gp
from gurobipy import GRB

from model.base import BiclusterModelBase

logger = logging.getLogger(__name__)


class MaxSurfaceModel(BiclusterModelBase):
    def __init__(self, rows_data, cols_data, edges, error_rate: float, env=None):
        # Stockage des données
        self.rows_data = rows_data
        self.cols_data = cols_data
        self.edges = set(edges)
        self.error_rate = float(error_rate)

        logger.info(
            "MaxSurfaceModel : initialisation — %d lignes, %d colonnes, %d arêtes, error_rate=%.4f.",
            len(rows_data), len(cols_data), len(list(edges)), error_rate,
        )

        # Création du modèle
        if env is not None:
            self.model = gp.Model("max_surface_grb_v3", env=env)
        else:
            self.model = gp.Model("max_surface_grb_v3")
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

        # Objectif
        self._surface_expr = gp.quicksum(cell_val * var for var, cell_val in self.lp_cells.values())
        self.model.setObjective(self._surface_expr, GRB.MAXIMIZE)

        # Contraintes de structure
        for (row, col), (cell_var, _) in self.lp_cells.items():
            self.model.addConstr(self.lp_rows[row][0] >= cell_var, name=f'cell_{row}_{col}_1')
            self.model.addConstr(self.lp_cols[col][0] >= cell_var, name=f'cell_{row}_{col}_2')
            self.model.addConstr(self.lp_rows[row][0] + self.lp_cols[col][0] - 1 <= cell_var, name=f'cell_{row}_{col}_3')

        # Contrainte de taux d'erreur
        self._err_expr = gp.quicksum((1 - cell_val) * var for var, cell_val in self.lp_cells.values())
        self._tot_expr = gp.quicksum(var for var, _ in self.lp_cells.values())
        self._err_rate_constr = self.model.addConstr(self._err_expr <= self.error_rate * self._tot_expr, name='err_rate')

    def add_WarmStart(self, rows, cols):
        logger.debug("MaxSurfaceModel : WarmStart fourni (%d lignes, %d colonnes).", len(rows), len(cols))
        # Fournit un démarrage chaud MIP complet et cohérent pour toutes les variables
        rows_set = set(rows)
        cols_set = set(cols)

        # Initialise les starts des variables ligne/colonne à 0 par défaut, 1 si sélectionnée
        for r, (var_r, _) in self.lp_rows.items():
            var_r.start = 1 if r in rows_set else 0
        for c, (var_c, _) in self.lp_cols.items():
            var_c.start = 1 if c in cols_set else 0

        # Initialise les starts des variables cellule cohérents avec le ET logique ligne/colonne
        # cell_{r,c} = 1 ssi row_r = 1 et col_c = 1
        for (r, c), (var_cell, _) in self.lp_cells.items():
            var_cell.start = 1 if (r in rows_set and c in cols_set) else 0

        # Pousse les starts dans le modèle
        self.model.update()

    # Fonctions utilitaires du solveur
    def optimize(self):
        logger.info("MaxSurfaceModel : lancement de l'optimisation.")
        self.model.optimize()
        logger.info(
            "MaxSurfaceModel : optimisation terminée — status=%d%s.",
            self.model.Status,
            f", ObjVal={self.model.ObjVal:.4f}" if self.model.Status == GRB.OPTIMAL else "",
        )
        if self.model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            logger.warning("MaxSurfaceModel : status inattendu (%d).", self.model.Status)

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
        rows = [int(v.VarName.split('_')[1]) for v in self.getVars() if v.VarName.startswith('row_') and v.X > 0.5]
        logger.debug("MaxSurfaceModel : %d ligne(s) sélectionnée(s).", len(rows))
        return rows

    def get_selected_cols(self):
        cols = [int(v.VarName.split('_')[1]) for v in self.getVars() if v.VarName.startswith('col_') and v.X > 0.5]
        logger.debug("MaxSurfaceModel : %d colonne(s) sélectionnée(s).", len(cols))
        return cols