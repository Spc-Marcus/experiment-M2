from typing import List
import random

def create_matrix(L: int, C: int, density: float, seed: int) -> List[List[int]]:
    """
    Crée une matrice binaire avec une densité exacte de uns et reproduction déterministe.

    Parameters
    - L, C: dimensions
    - density: fraction de uns (0.0 - 1.0)
    - seed: entier utilisé pour initialiser un RNG local (garantit la reproduction parfaite)
    """
    rng = random.Random(seed)
    total = L * C
    ones = int(round(density * total))
    ones = max(0, min(total, ones))
    zeros = total - ones
    flat = [1] * ones + [0] * zeros
    rng.shuffle(flat)
    matrix = [flat[i * C:(i + 1) * C] for i in range(L)]
    return matrix

if __name__ == "__main__":
    L = 5  # Nombre de lignes
    C = 5  # Nombre de colonnes
    density = 0.35  # Densité souhaitée de uns (35%)
    seed = 42  # Seed pour la reproductibilité
    matrix = create_matrix(L, C, density, seed)
    actual_ones = sum(row.count(1) for row in matrix)
    total = L * C
    expected_ones = int(round(density * total))
    print(f"Expected ones: {expected_ones}/{total}, actual ones: {actual_ones}, density: {actual_ones/total:.6f}")