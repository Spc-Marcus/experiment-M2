from create_matrix import *


def make_all_steps(extended_info: MixedMatrixInfo,base : list[list[int]]) -> list[tuple[list[int], list[int], list[int]]]:
    """
    Génère toutes les étapes de biclustering nécessaires pour reconstruire la matrice de base à partir de la matrice étendue.

    Arguments:
    ----------
    extended_info : MixedMatrixInfo
        Informations sur la matrice étendue.
    base : list of list of int
        Matrice binaire de base à reconstruire.

    Returns:
    --------
    List of tuples (reads1, reads0, cols)
        Liste des étapes de biclustering.
    """
    steps: list[tuple[list[int], list[int], list[int]]] = []
    ext_matrix = extended_info.matrix
    # Dans MixedMatrixInfo, nous avons des listes d'indices d'origine
    row_origin_indices = extended_info.row_origin_indices  # len = nb lignes étendues (mélangées)
    col_origin_indices = extended_info.col_origin_indices  # len = nb colonnes étendues (mélangées)

    base_n_rows = len(base)
    base_n_cols = len(base[0]) if base_n_rows > 0 else 0

    for base_col in range(base_n_cols):
        # Trouver les colonnes correspondantes dans la matrice étendue
        ext_cols = [ext_col for ext_col, orig_col in enumerate(col_origin_indices) if orig_col == base_col]
        if not ext_cols:
            continue  # Pas de colonnes correspondantes trouvées

        reads1 = []
        reads0 = []
        for ext_row in range(ext_matrix.shape[0]):
            # récupérer l'indice de la ligne originale source
            if ext_row >= len(row_origin_indices):
                continue
            orig_row = row_origin_indices[ext_row]
            if orig_row is None or orig_row >= base_n_rows:
                continue  # Ligne étendue sans correspondance dans la matrice de base

            if all(ext_matrix[ext_row, ext_col] == base[orig_row][base_col] for ext_col in ext_cols):
                if base[orig_row][base_col] == 1:
                    reads1.append(ext_row)
                else:
                    reads0.append(ext_row)

        steps.append((reads1, reads0, ext_cols))

    return steps

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

    # Générer les étapes de biclustering pour reconstruire la matrice de base
    steps = make_all_steps(mixed_info, matrix)
    print("\nÉtapes de biclustering générées :")
    for step_idx, (reads1, reads0, cols) in enumerate(steps):
        print(f"Étape {step_idx}: reads1={reads1}, reads0={reads0}, cols={cols}")