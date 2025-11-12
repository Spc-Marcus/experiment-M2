"""Utilitaires pour générer, étendre, bruiter et mélanger des matrices binaires.

Ce module fournit:
- create_simple_matrix: génère une matrice binaire (0/1) avec lignes et colonnes uniques
- extend_matrix: duplique aléatoirement lignes et colonnes et retourne les mappings
- add_noise_to_matrix: applique un bruit de type flip bit à un taux donné
- mix_matrices: permute aléatoirement lignes et colonnes en conservant les clusters

Les fonctions renvoient des structures riches (dataclasses) pour faciliter les
post-traitements (clustering, regroupements, débruitage, etc.).

Astuce reproductibilité: utilisez random.seed(S) et np.random.seed(S) avant
d'appeler ces fonctions pour obtenir des résultats identiques d'un run à l'autre.
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict

def create_simple_matrix(rows, cols) -> list[list[int]]:
    """Crée une matrice binaire avec lignes et colonnes uniques.

    Génère une matrice binaire de dimension rows x cols telle que:
    - toutes les lignes sont différentes entre elles
    - toutes les colonnes sont différentes entre elles

    Implémentation: on échantillonne aléatoirement des lignes parmi toutes les
    possibilités (2^cols), puis on valide l'unicité des colonnes. On répète
    jusqu'à trouver une configuration valide ou dépasser un nombre d'essais.

    Args:
        rows: nombre de lignes désiré (int)
        cols: nombre de colonnes désiré (int)

    Returns:
        Liste de listes d'entiers 0/1, de taille rows x cols.
    """
    if rows > 2 ** cols or cols > 2 ** rows:
        raise ValueError("Impossible de garantir unicité des lignes et colonnes avec ces dimensions.")

    all_possible_rows = [list(map(int, format(i, f'0{cols}b'))) for i in range(2 ** cols)]
    tries = 10000
    for _ in range(tries):
        candidate_rows = random.sample(all_possible_rows, rows)
        columns = list(zip(*candidate_rows))
        if len({tuple(col) for col in columns}) == cols:
            return candidate_rows
    raise ValueError("Impossible de générer une matrice avec lignes et colonnes toutes uniques pour ces dimensions après plusieurs essais.")


@dataclass
class ExtendedMatrixInfo:
    """Informations détaillées sur la matrice étendue.

    Attributes:
        matrix: matrice étendue (np.ndarray de forme (R_ext, C_ext)).
        row_multiplicities: longueur = R_orig; nb de copies choisies pour chaque
            ligne originale (dans [min_rows, max_rows]).
        row_origin_indices: longueur = R_ext; pour chaque ligne étendue, l'indice
            (0..R_orig-1) de la ligne originale dont elle est issue.
        col_multiplicities: longueur = C_orig; nb de copies choisies pour chaque
            colonne originale (dans [min_cols, max_cols]).
        col_origin_indices: longueur = C_ext; pour chaque colonne étendue, l'indice
            (0..C_orig-1) de la colonne originale dont elle est issue.
    """
    matrix: np.ndarray
    row_multiplicities: List[int]
    row_origin_indices: List[int]
    col_multiplicities: List[int]
    col_origin_indices: List[int]


def extend_matrix(matrix: list[list[int]], size_cols: list[int], size_rows: list[int]) -> ExtendedMatrixInfo:
    """Étend une matrice binaire en dupliquant lignes et colonnes.

    Pour chaque ligne originale i, tire un nombre nb_copies_i ~ U[min_rows, max_rows]
    et répète la ligne nb_copies_i fois. Pour chaque colonne originale j, tire un
    nb_copies_j ~ U[min_cols, max_cols] et répète la colonne nb_copies_j fois.

    Args:
        matrix: matrice binaire d'entrée (liste de listes 0/1), de forme (R_orig, C_orig)
        size_cols: [min_cols, max_cols], bornes inclusives pour la duplication des colonnes
        size_rows: [min_rows, max_rows], bornes inclusives pour la duplication des lignes

    Returns:
        ExtendedMatrixInfo contenant la matrice étendue et les mappings de clusters.

    Raises:
        ValueError: si size_cols ou size_rows ne sont pas des listes de longueur 2
            ou si min > max.

    Example:
        >>> base = [[0,1,0],[1,0,1]]
        >>> random.seed(0); np.random.seed(0)
        >>> info = extend_matrix(base, [1,2], [2,3])
        >>> info.matrix.shape[0] >= 2 and info.matrix.shape[1] >= 2
        True
    """
    if not isinstance(size_cols, list) or not isinstance(size_rows, list) or len(size_cols) != 2 or len(size_rows) != 2:
        raise ValueError("size_cols et size_rows doivent être des listes [min, max] de longueur 2.")
    min_cols, max_cols = size_cols
    min_rows, max_rows = size_rows
    if min_cols > max_cols or min_rows > max_rows:
        raise ValueError("Pour chaque intervalle [min,max], on doit avoir min <= max.")
    if not matrix:
        return ExtendedMatrixInfo(np.array([]), [], [], [], [])

    min_cols, max_cols = size_cols
    min_rows, max_rows = size_rows

    # Étendre les lignes
    extended_rows: List[List[int]] = []
    row_multiplicities: List[int] = []
    row_origin_indices: List[int] = []  # pour chaque ligne étendue, indice de la ligne source
    for original_idx, row in enumerate(matrix):
        nb_copies = random.randint(min_rows, max_rows)
        row_multiplicities.append(nb_copies)
        for _ in range(nb_copies):
            extended_rows.append(row[:])
            row_origin_indices.append(original_idx)

    if not extended_rows:
        return ExtendedMatrixInfo(np.array([]), row_multiplicities, row_origin_indices, [], [])

    # Étendre les colonnes
    num_cols = len(extended_rows[0])
    col_multiplicities: List[int] = [random.randint(min_cols, max_cols) for _ in range(num_cols)]

    final_matrix_rows: List[List[int]] = []
    col_origin_indices: List[int] = []  # mapping colonne étendue -> colonne originale
    for row in extended_rows:
        new_row: List[int] = []
        for j, col_val in enumerate(row):
            # ajouter les copies successives
            new_row.extend([col_val] * col_multiplicities[j])
            # enregistrer mapping colonne originale -> colonnes étendues (une entrée par colonne étendue)
        final_matrix_rows.append(new_row)
    # Construire col_origin_indices en parcourant les colonnes originales
    for original_col_idx, multiplicity in enumerate(col_multiplicities):
        col_origin_indices.extend([original_col_idx] * multiplicity)

    final_matrix = np.array(final_matrix_rows)

    return ExtendedMatrixInfo(
        matrix=final_matrix,
        row_multiplicities=row_multiplicities,
        row_origin_indices=row_origin_indices,
        col_multiplicities=col_multiplicities,
        col_origin_indices=col_origin_indices,
    )

def add_noise_to_matrix(matrix: np.ndarray, error_rate: float) -> np.ndarray:
    """Ajoute du bruit de type flip bit à une matrice binaire.

    À partir d'une matrice 0/1, on sélectionne aléatoirement un nombre de cases
    égal à floor(error_rate * matrix.size) et on inverse leur valeur (0->1, 1->0).

    Args:
        matrix: np.ndarray binaire (0/1)
        error_rate: taux d'erreur dans [0.0, 1.0]. Par ex 0.01 => ~1% des cases.

    Returns:
        np.ndarray de même forme que matrix, bruitée.

    Raises:
        ValueError: si error_rate est hors de [0.0, 1.0].
    """
    if not (0.0 <= error_rate <= 1.0):
        raise ValueError("error_rate doit être dans [0.0, 1.0].")
    noisy_matrix = matrix.copy()
    n_errors = int(error_rate * matrix.size)
    
    # Choisir des positions aléatoirement et inverser leur valeur
    flat_indices = np.random.choice(matrix.size, n_errors, replace=False)
    row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)
    
    for r, c in zip(row_indices, col_indices):
        noisy_matrix[r, c] = 1 - noisy_matrix[r, c]
    
    return noisy_matrix

@dataclass
class MixedMatrixInfo:
    """Informations sur la matrice mélangée (permutation aléatoire).

    Attributes:
        matrix: matrice mélangée (np.ndarray de forme (R_ext, C_ext)).
        row_origin_indices: longueur = R_ext; pour chaque ligne mélangée, l'indice
            (0..R_orig-1) de la ligne originale d'où elle provient (hérité d'extend).
        col_origin_indices: longueur = C_ext; pour chaque colonne mélangée, l'indice
            (0..C_orig-1) de la colonne originale d'où elle provient (hérité d'extend).
        row_clusters: dictionnaire {idx_ligne_originale -> [indices_lignes_mélangées]}
        col_clusters: dictionnaire {idx_colonne_originale -> [indices_colonnes_mélangées]}
    """
    matrix: np.ndarray
    row_origin_indices: List[int]
    col_origin_indices: List[int]
    row_clusters: Dict[int, List[int]]
    col_clusters: Dict[int, List[int]]


def mix_matrices(extended_info: ExtendedMatrixInfo) -> MixedMatrixInfo:
    """Mélange lignes et colonnes d'une matrice étendue et retourne les clusters.

    Cette fonction prend en entrée l'objet retourné par extend_matrix (ou la même
    structure où la matrice a été bruitée) et applique indépendamment:
      - une permutation aléatoire des lignes
      - une permutation aléatoire des colonnes
    Elle reconstruit ensuite les clusters de lignes/colonnes en fonction des
    indices d'origine hérités d'extend_matrix.

    Args:
        extended_info: ExtendedMatrixInfo (peut contenir une matrice bruitée)

    Returns:
        MixedMatrixInfo contenant la matrice mélangée, les mappings et les clusters.
    """
    matrix = extended_info.matrix.copy()
    if matrix.size == 0:
        return MixedMatrixInfo(matrix, [], [], {}, {})

    # Permutation des lignes
    row_perm = np.random.permutation(matrix.shape[0])
    shuffled_matrix = matrix[row_perm, :]
    shuffled_row_origin_indices = [extended_info.row_origin_indices[i] for i in row_perm]

    # Construire nouveaux clusters de lignes
    row_clusters: Dict[int, List[int]] = {}
    for new_row_idx, original_row_idx in enumerate(shuffled_row_origin_indices):
        row_clusters.setdefault(original_row_idx, []).append(new_row_idx)

    # Permutation des colonnes
    col_perm = np.random.permutation(shuffled_matrix.shape[1])
    shuffled_matrix = shuffled_matrix[:, col_perm]
    shuffled_col_origin_indices = [extended_info.col_origin_indices[i] for i in col_perm]

    # Construire nouveaux clusters de colonnes
    col_clusters: Dict[int, List[int]] = {}
    for new_col_idx, original_col_idx in enumerate(shuffled_col_origin_indices):
        col_clusters.setdefault(original_col_idx, []).append(new_col_idx)

    return MixedMatrixInfo(
        matrix=shuffled_matrix,
        row_origin_indices=shuffled_row_origin_indices,
        col_origin_indices=shuffled_col_origin_indices,
        row_clusters=row_clusters,
        col_clusters=col_clusters,
    )

if __name__ == "__main__":
    # Exemple d'utilisation amélioré
    rows = 3
    cols = 3
    matrix = create_simple_matrix(rows, cols)
    print("Matrice initiale :")
    for row in matrix:
        print(row)

    size_cols = [2, 5]
    size_rows = [3, 5]
    extended_info = extend_matrix(matrix, size_cols, size_rows)
    print("\nMatrice étendue :")
    print(extended_info.matrix)
    print("Multiplicités des lignes originales:", extended_info.row_multiplicities)
    print("Mapping lignes étendues -> index original:", extended_info.row_origin_indices)
    print("Multiplicités des colonnes originales:", extended_info.col_multiplicities)
    print("Mapping colonnes étendues -> index original:", extended_info.col_origin_indices)

    # Ajouter du bruit
    error_rate = 0.01
    noisy_matrix = add_noise_to_matrix(extended_info.matrix, error_rate)
    print("\nMatrice avec bruit :")
    print(noisy_matrix)

    # Mettre à jour l'objet pour le mélange (on garde les mappings, seule la matrice est bruitée)
    extended_info_noisy = ExtendedMatrixInfo(
        matrix=noisy_matrix,
        row_multiplicities=extended_info.row_multiplicities,
        row_origin_indices=extended_info.row_origin_indices,
        col_multiplicities=extended_info.col_multiplicities,
        col_origin_indices=extended_info.col_origin_indices,
    )

    mixed_info = mix_matrices(extended_info_noisy)
    print("\nMatrice mélangée :")
    print(mixed_info.matrix)
    print("Clusters de lignes (original -> lignes mélangées):", mixed_info.row_clusters)
    print("Clusters de colonnes (original -> colonnes mélangées):", mixed_info.col_clusters)