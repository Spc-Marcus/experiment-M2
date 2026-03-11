"""
main.py – Point d'entrée principal de Max-cli.

Usage
-----
    python Max-cli/main.py [config_file]

Si ``config_file`` n'est pas fourni, le programme lit ``Max-cli/config.arg``.

Fonctionnement
--------------
1. Lecture de la configuration (.arg).
2. Génération d'une matrice aléatoire (new-gen=1) ou chargement depuis un
   fichier CSV (new-gen=0).
3. Appel du modèle ILP sélectionné pour trouver la plus grande sous-matrice
   dense (au sens du paramètre gamma).
4. Sauvegarde de la sous-matrice trouvée dans un fichier texte lisible.
"""

import csv
import os
import sys
import time

import numpy as np

# ── Ajout du chemin racine pour les imports relatifs ──────────────────────────
_ROOT = os.path.dirname(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.parser import parse_arg_file
from utils.logger import setup_logger
from utils.create_matrix_V2 import create_matrix

# Import model_call depuis le même dossier (Max-cli contient un tiret, on utilise le chemin)
_CLI_DIR = os.path.join(_ROOT, "Max-cli")
if _CLI_DIR not in sys.path:
    sys.path.insert(0, _CLI_DIR)
from model_call import find_dense_submatrix 


def load_matrix_from_csv(path: str) -> np.ndarray:
    """Charge une matrice binaire depuis un fichier CSV (valeurs 0/1)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append([int(v.strip()) for v in row])
    return np.array(rows, dtype=int)


def save_matrix_csv(matrix: np.ndarray, path: str) -> None:
    """Sauvegarde une matrice numpy en CSV."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(row.tolist())


def save_submatrix(
    matrix: np.ndarray,
    row_indices: list,
    col_indices: list,
    output_path: str,
    gamma: float,
    model: str,
    model_time: float = None,
) -> None:
    """Sauvegarde la sous-matrice dans un fichier texte lisible."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    sorted_rows = sorted(row_indices)
    sorted_cols = sorted(col_indices)
    sub = matrix[np.ix_(sorted_rows, sorted_cols)] if (sorted_rows and sorted_cols) else np.array([])
    density = sub.mean() if sub.size > 0 else 0.0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Résultat Max-cli – Sous-matrice dense\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Modèle utilisé : {model}\n")
        if model_time is not None:
            f.write(f"Temps de résolution du modèle : {model_time:.2f} s\n")
        f.write(f"Gamma (densité min) : {gamma}\n")
        f.write(f"Dimensions originales : {matrix.shape[0]} lignes × {matrix.shape[1]} colonnes\n\n")
        f.write(f"Sous-matrice trouvée : {len(sorted_rows)} lignes × {len(sorted_cols)} colonnes\n")
        f.write(f"Densité réelle : {density:.4f}\n\n")
        f.write(f"Indices de lignes : {sorted_rows}\n")
        f.write(f"Indices de colonnes : {sorted_cols}\n\n")
        f.write("Matrice originale :\n")
        f.write("-" * 40 + "\n")
        for i, row in enumerate(matrix):
            row_vals = " ".join(str(v) for v in row)
            f.write(f"  ligne {i:>4} │ {row_vals}\n")
        f.write("-" * 40 + "\n\n")
        f.write("Sous-matrice (lignes=sélectionnées, colonnes=sélectionnées) :\n")
        f.write("-" * 40 + "\n")
        if sub.size > 0:
            for r_idx, r in enumerate(sorted_rows):
                row_vals = " ".join(str(v) for v in sub[r_idx])
                f.write(f"  ligne {r:>4} │ {row_vals}\n")
        else:
            f.write("  (aucune sous-matrice)\n")
        f.write("-" * 40 + "\n")


def main(config_path: str = "Max-cli/config.arg") -> None:
    conf = parse_arg_file(config_path)

    # ── Paramètres de logging ──────────────────────────────────────────────────
    log_level = str(conf.get("log_level", "INFO"))
    log_file = conf.get("log_file", None)
    if isinstance(log_file, str) and log_file.strip() == "":
        log_file = None
    logger = setup_logger("experiment", level=log_level, log_file=log_file)

    logger.info("Configuration chargée depuis '%s' : %s", config_path, conf)

    # ── Paramètres généraux ────────────────────────────────────────────────────
    new_gen = int(conf.get("new-gen", 1))
    source = conf.get("source", "Mat/toto.csv")
    nom = conf.get("nom", "output")
    seed_raw = conf.get("seed", None)
    seed = int(seed_raw) if seed_raw not in (None, "", 0) else int(time.time())
    heuristic = int(conf.get("heuristic", 0))
    gamma = float(conf.get("gamma", 0.975))
    # `model` peut être une chaîne ou une liste (selon utils/parser.py).
    model_conf = conf.get("model", "max_one")
    if isinstance(model_conf, list):
        # Si la liste contient un seul élément, renvoyer cet élément (comportement demandé).
        if len(model_conf) == 1:
            model = str(model_conf[0]).strip()
        else:
            # Normaliser chaque entrée en chaîne sans espaces superflus.
            model = [str(m).strip() for m in model_conf]
    else:
        model = str(model_conf).strip()
    output_dir = str(conf.get("output_dir", "Max-cli/results"))

    # ── Génération ou chargement de la matrice ─────────────────────────────────
    if new_gen:
        rows_count = int(conf.get("rows", 20))
        cols_count = int(conf.get("cols", 15))
        density = float(conf.get("density", 0.7))
        logger.info(
            "Génération d'une matrice aléatoire %dx%d "
            "(densité=%.2f, seed=%d).",
            rows_count, cols_count, density, seed,
        )
        matrix_list = create_matrix(rows_count, cols_count, density, seed)
        matrix = np.array(matrix_list, dtype=int)
        # Sauvegarde de la matrice générée dans source
        save_matrix_csv(matrix, source)
        logger.info("Matrice générée sauvegardée dans '%s'.", source)
    else:
        logger.info("Chargement de la matrice depuis '%s'.", source)
        if not os.path.exists(source):
            logger.error("Fichier source introuvable : '%s'.", source)
            sys.exit(1)
        matrix = load_matrix_from_csv(source)

    logger.info(
        "Matrice prête : %d lignes × %d colonnes, densité globale=%.4f.",
        matrix.shape[0], matrix.shape[1], matrix.mean(),
    )

    # ── Appel du modèle ────────────────────────────────────────────────────────
    start = time.time()
    results = []
    # Supporter un seul modèle ou une liste de modèles. Mesurer le temps par modèle.
    if isinstance(model, list):
        for m in model:
            m_start = time.time()
            row_indices, col_indices, success = find_dense_submatrix(
                matrix, model=m, gamma=gamma, use_heuristic=heuristic
            )
            m_elapsed = time.time() - m_start
            results.append((m, row_indices, col_indices, success, m_elapsed))
    else:
        m_start = time.time()
        row_indices, col_indices, success = find_dense_submatrix(
            matrix, model=model, gamma=gamma, use_heuristic=heuristic
        )
        m_elapsed = time.time() - m_start
        results.append((model, row_indices, col_indices, success, m_elapsed))
    elapsed = time.time() - start

    logger.info("Temps de résolution : %.2f s.", elapsed)

    # ── Sauvegarde des résultats ───────────────────────────────────────────────
    timestamp = int(time.time())
    os.makedirs(output_dir, exist_ok=True)

    for m_name, row_indices, col_indices, success, m_time in results:
        # Construire un nom de fichier sûr à partir du nom du modèle.
        safe_model = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(m_name))
        output_file = os.path.join(output_dir, f"{nom}_{safe_model}_{timestamp}.txt")
        save_submatrix(matrix, row_indices, col_indices, output_file, gamma, m_name, model_time=m_time)
        logger.info("Résultats sauvegardés dans '%s'.", output_file)

        if not success:
            logger.warning("Aucune sous-matrice dense (gamma=%.4f) trouvée pour le modèle '%s'.", gamma, m_name)
        else:
            print(
                f"Sous-matrice trouvée pour '{m_name}': {len(row_indices)} lignes × {len(col_indices)} colonnes "
                f"— résultats dans '{output_file}' (temps={m_time:.2f}s)"
            )


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "Max-cli/config.arg"
    main(cfg)
