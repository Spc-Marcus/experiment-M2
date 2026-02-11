
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.create_matrix import *
from utils.reconstruire import *
from utils.parser import parse_arg_file
from ilp import *
from post_processing import post_processing
import csv
import time
import numpy as np
import pprint
from typing import List, Tuple

def clusters_equal(clusters1, clusters2):
    sorted1 = sorted([tuple(sorted(cluster)) for cluster in clusters1])
    sorted2 = sorted([tuple(sorted(cluster)) for cluster in clusters2])
    return sorted1 == sorted2

"""
Point d'entrée principal du programme.
conf file doit être un fichier .arg avec des paires clé: valeur.
nb_it doit être un entier indiquant le nombre d'itérations.
stripe une liste de valeur pour le nb de stripes.
haplotypes une liste de valeurs pour le nb d'haplotypes.
min_row et max_row des entiers pour les bornes du nb de reads. (pour la mult)
min_col et max_col des entiers pour les bornes du nb de SNPs. (pour la mult)
max_one, max_one_v2, max_one_v3 ,max_e_r 1 : utiliser le model 0 si non
resultat_file un fichier de sortie pour les résultats en .csv
[optionnel] distance un float pour la distance de fusion.

best file doit être un fichier .csv avec les colonnes:
Error-Rate, Threshold, Distance

"""

def main(conf_file: dict):
    # Exemple d'utilisation des fonctions importées
    print("Configuration chargée :", conf_file)
     # En-tête CSV
    header_cols = [
        "Error-Rate","Strip","Haplotype",
        "Matrix-Rows","Matrix-Cols",
    ]

    def _is_selected(val):
        if val is None:
            return False
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on")
        return bool(val)

    # Liste des modèles disponibles:
    # - max_one: V1 original - Nouveau modèle à chaque phase
    # - max_one_v2: V2 - Un seul modèle avec contraintes dynamiques
    # - max_one_v3: V3 - Un seul modèle + contraintes dynamiques + WarmStart (comme V2 mais avec WarmStart)
    # - max_one_v3a: V3a - Un seul modèle + contraintes dynamiques, SANS WarmStart
    # - max_one_v3b: V3b - Reconstruction modèle (seeds only) + WarmStart
    # - max_one_v3c: V3c - Reconstruction modèle (seeds only), SANS WarmStart
    # - max_e_r: Modèle avec error rate dans l'objectif
    model_keys = ["max_one", "max_one_v2", "max_one_v3", "max_one_v3a", "max_one_v3b", "max_one_v3c", "max_e_r"]
    selected_models = [k for k in model_keys if _is_selected(conf_file.get(k))]

    for model_name in selected_models:
        header_cols.extend([
            f"{model_name}-Time", f"{model_name}-Haplotypes",
            f"{model_name}-Stripe", f"{model_name}-Orfelin", f"{model_name}-Is-Equal",
        ])

    print("En-tête CSV :", header_cols)
    # Récupération des paramètres depuis le fichier de configuration
    distance_thresh = conf_file.get('distance', 0.1)
    stripe_values = conf_file.get('stripe')
    haplotype_values = conf_file.get('haplotypes')
    size_cols = [conf_file.get('min_col'), conf_file.get('max_col')]
    size_rows = [conf_file.get('min_row'), conf_file.get('max_row')]
    nb_it = conf_file.get('nb_it')
    resultat_file = conf_file.get('resultat_file')
    best_file = conf_file.get('best')
    log_error = conf_file.get('log_error', 0)
    if resultat_file:
        resultat_file = os.path.join("No-error", resultat_file)
    if best_file:
        best_file = os.path.join("No-error", best_file)
    error_rate = 0.0
    if best_file and os.path.exists(best_file):
        with open(best_file, 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader, {})
            error_rate = float(row.get('Error-Rate', 0.0))

    with open(resultat_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_cols)
        writer.writeheader()

        for stripe in stripe_values:
            for haplotypes in haplotype_values:
                print(f"Traitement pour stripe={stripe}, haplotypes={haplotypes}")
                for i in range(nb_it):
                    # Appel des fonctions de création de matrice et de reconstruction
                    matrix = create_simple_matrix(haplotypes, stripe)
                    extended_info = extend_matrix(matrix, size_rows, size_cols)
                    matrixCopy = extended_info.matrix.copy()
                    mixed_info = mix_matrix(extended_info)
                    # Génération des étapes de biclustering pour reconstruire la matrice de base
                    reconstructed = make_all_steps(mixed_info, matrix)
                    read_names = [f"read_{i}" for i in range(mixed_info.matrix.shape[0])]
                    baseline_clusters, _, baseline_orphans, _ = post_processing(matrixCopy, reconstructed, read_names, distance_thresh,1)
                    # utiliser les modèles ILP sélectionnés
                    row = {
                        'Error-Rate': error_rate,
                        'Strip': stripe,
                        'Haplotype': haplotypes,
                        'Matrix-Rows': mixed_info.matrix.shape[0],
                        'Matrix-Cols': mixed_info.matrix.shape[1],
                    }
                    model_data = {}
                    for model_name in selected_models:
                        print(f"Utilisation du modèle {model_name} pour l'itération {i+1}")
                        # Préparer paramètres pour clustering_full_matrix
                        regions = list(range(mixed_info.matrix.shape[1]))  # chaque colonne comme région par défaut
                        version_map = {
                            "max_one": 1,
                            "max_one_v2": 2,
                            "max_one_v3": 4,       # Un seul modèle + contraintes dynamiques + WarmStart
                            "max_one_v3a": 5,      # Un seul modèle + contraintes dynamiques, SANS WarmStart
                            "max_one_v3b": 6,      # Reconstruction modèle (seeds only) + WarmStart
                            "max_one_v3c": 7,      # Reconstruction modèle (seeds only), SANS WarmStart
                            "max_e_r": 3,
                        }
                        version = version_map.get(model_name, 1)
                        min_row_q = size_rows[0]
                        min_col_q = size_cols[0]
                        # Appel à la nouvelle fonction
                        start = time.time()
                        steps, info = clustering_full_matrix(matrixCopy,
                                                    regions=regions,
                                                    version=version,
                                                    min_row_quality=1,
                                                    min_col_quality=1,
                                                    error_rate=error_rate)
                        end = time.time()
                        elapsed_time = end - start
                        model_clusters, _, model_orphans, _ = post_processing(matrixCopy, steps, read_names,distance_thresh, 1)
                        is_equal = clusters_equal(baseline_clusters, model_clusters)
                        model_data[model_name] = {
                            'time': elapsed_time,
                            'haplotypes': len(model_clusters),
                            'stripes': len(steps),
                            'orphans': len(model_orphans),
                            'is_equal': is_equal,
                            'steps': steps,
                            'clusters': model_clusters,
                            'orphans_list': model_orphans
                        }
                        row[f'{model_name}-Time'] = elapsed_time
                        row[f'{model_name}-Haplotypes'] = len(model_clusters)
                        row[f'{model_name}-Stripe'] = len(steps)
                        row[f'{model_name}-Orfelin'] = len(model_orphans)
                        row[f'{model_name}-Is-Equal'] = 1 if is_equal else 0
                    if log_error and (not any(d['is_equal'] for d in model_data.values()) or any(d['haplotypes'] != haplotypes for d in model_data.values())):
                        os.makedirs("temp", exist_ok=True)
                        filename = f"temp/{int(time.time())}.txt"
                        with open(filename, 'w') as f:
                            f.write(f"Stripe: {stripe}\n")
                            f.write(f"Haplotypes expected: {haplotypes}\n")
                            f.write(f"Extended matrix shape: {extended_info.matrix.shape}\n")
                            f.write("Extended matrix:\n")
                            for r in extended_info.matrix:
                                f.write(" ".join(map(str, r)) + "\n")
                            f.write(f"Mixed matrix shape: {mixed_info.matrix.shape}\n")
                            f.write("Mixed matrix:\n")
                            for r in mixed_info.matrix:
                                f.write(" ".join(map(str, r)) + "\n")
                            f.write(f"Row origin indices:\n{pprint.pformat(mixed_info.row_origin_indices, width=1000)}\n")
                            f.write(f"Col origin indices:\n{pprint.pformat(mixed_info.col_origin_indices, width=1000)}\n")
                            f.write(f"Row clusters:\n{pprint.pformat(mixed_info.row_clusters)}\n")
                            f.write(f"Col clusters:\n{pprint.pformat(mixed_info.col_clusters)}\n")
                            f.write(f"Reconstructed steps:\n{pprint.pformat(reconstructed, width=1000)}\n")
                            f.write(f"Baseline clusters:\n{pprint.pformat(baseline_clusters, width=1000)}\n")
                            f.write(f"Baseline orphans:\n{pprint.pformat(baseline_orphans, width=1000)}\n")
                            for model_name, d in model_data.items():
                                f.write(f"Model {model_name}:\n")
                                f.write(f"Time: {d['time']}\n")
                                f.write(f"Haplotypes found: {d['haplotypes']}\n")
                                f.write(f"Stripes: {d['stripes']}\n")
                                f.write(f"Orphans: {d['orphans']}\n")
                                f.write(f"Is-Equal: {d['is_equal']}\n")
                                f.write(f"Steps:\n{pprint.pformat(d['steps'], width=1000)}\n")
                                f.write(f"Clusters:\n{pprint.pformat(d['clusters'], width=1000)}\n")
                                f.write(f"Orphans list:\n{pprint.pformat(d['orphans_list'], width=1000)}\n")
                    writer.writerow(row)
                    # Forcer l'écriture sur le disque afin que le fichier soit mis à jour
                    # en continu et visible même avant la fermeture du contexte.
                    try:
                        csvfile.flush()
                        os.fsync(csvfile.fileno())
                    except Exception:
                        # Si fsync échoue (par ex. sur certains systèmes de fichiers),
                        # on ignore silencieusement mais le flush est déjà utile.
                        pass



if __name__ == "__main__":
    var_file = "No-error/config.arg"
    args = parse_arg_file(var_file)
    main(args)