from abc import ABC, abstractmethod
from typing import Any, List, Tuple


# Type aliases for the data fed to every model
RowsData = List[Tuple[int, int]]   # [(row_index, degree), ...]
ColsData = List[Tuple[int, int]]   # [(col_index, degree), ...]
Edges = List[Tuple[int, int]]      # [(row_index, col_index), ...]


class BiclusterModelBase(ABC):
    """
    Interface générique pour tous les modèles de biclustering (MaxOne, MaxSurface, …).

    Chaque implémentation doit accepter la même signature de constructeur
    et exposer les méthodes/propriétés listées ici, ce qui permet de
    typer ``model_class: Type[BiclusterModelBase]`` dans la heuristique
    ou tout autre code appelant.
    """

    @abstractmethod
    def __init__(
        self,
        rows_data: RowsData,
        cols_data: ColsData,
        edges: Edges,
        error_rate: float,
    ) -> None: ...

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    @abstractmethod
    def optimize(self) -> None:
        """Lance la résolution du modèle."""

    @abstractmethod
    def setParam(self, param: str, value: Any) -> None:
        """Transmet un paramètre au solveur sous-jacent."""

    # ------------------------------------------------------------------
    # Résultats
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def status(self) -> int:
        """Statut du solveur après optimisation (ex. GRB.Status)."""

    @property
    @abstractmethod
    def ObjVal(self) -> float:
        """Valeur de l'objectif après optimisation."""

    @abstractmethod
    def get_selected_rows(self) -> List[int]:
        """Retourne les indices des lignes sélectionnées (variable = 1)."""

    @abstractmethod
    def get_selected_cols(self) -> List[int]:
        """Retourne les indices des colonnes sélectionnées (variable = 1)."""


# Alias de rétro-compatibilité
MaxOneModelBase = BiclusterModelBase
