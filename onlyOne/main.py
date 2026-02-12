import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.create_matrix import *
from utils.reconstruire import *
from utils.parser import parse_arg_file
from ilp import clustering_full_matrix
from post_processing import post_processing
from pre_processing import pre_processing
import csv
import time
import pprint


def clusters_equal(clusters1, clusters2):
    sorted1 = sorted([tuple(sorted(cluster)) for cluster in clusters1])
    sorted2 = sorted([tuple(sorted(cluster)) for cluster in clusters2])
    return sorted1 == sorted2


def main(conf_file: dict):
    print("Configuration chargée :", conf_file)

    # En-tête CSV : colonnes communes + colonnes pour chaque approche
    header_cols = [
        "Error-Rate", "Strip", "Haplotype", "Matrix-Rows", "Matrix-Cols",
        "PreProc-Strips", "PreProc-Ambiguous-Cols",
        # Approche alternante (1 puis 0 puis 1...)
        "alt-Time", "alt-Haplotypes", "alt-Stripe", "alt-Orfelin", "alt-Is-Equal",
        # Approche only-ones (uniquement les 1)
        "only1-Time", "only1-Haplotypes", "only1-Stripe", "only1-Orfelin", "only1-Is-Equal",
    ]

    # Paramètres
    distance_thresh = conf_file.get('distance', 0.0)
    stripe_values = conf_file.get('stripe')
    haplotype_values = conf_file.get('haplotypes')
    size_cols = [conf_file.get('min_col'), conf_file.get('max_col')]
    size_rows = [conf_file.get('min_row'), conf_file.get('max_row')]
    nb_it = conf_file.get('nb_it')
    resultat_file = conf_file.get('resultat_file')
    log_error = conf_file.get('log_error', 0)
    certitude = float(conf_file.get('certitude', 0.2))
    # Taux d'erreur : peut être un float unique ou une liste de floats
    error_rate_values = conf_file.get('error_rate', 0.0)
    if not isinstance(error_rate_values, list):
        error_rate_values = [float(error_rate_values)]

    if resultat_file:
        resultat_file = os.path.join("onlyOne", resultat_file)

    with open(resultat_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header_cols)
        writer.writeheader()

        for stripe in stripe_values:
            for haplotypes in haplotype_values:
              for error_rate in error_rate_values:
                print(f"--- stripe={stripe}, haplotypes={haplotypes}, error_rate={error_rate} ---")
                for i in range(nb_it):
                    # Génération de la matrice
                    matrix = create_simple_matrix(haplotypes, stripe)
                    extended_info = extend_matrix(matrix, size_rows, size_cols)
                    mixed_info = mix_matrix(extended_info)
                    mixedMatrix = mixed_info.matrix.copy()
                    reconstructed = make_all_steps(mixed_info, matrix)
                    read_names = [f"read_{j}" for j in range(mixed_info.matrix.shape[0])]
                    baseline_clusters, _, baseline_orphans, _ = post_processing(
                        mixedMatrix, reconstructed, read_names, distance_thresh, 1)

                    # Appliquer le bruit à la matrice mélangée avant résolution
                    if error_rate > 0:
                        matrixNoisy = add_noise_to_matrix(mixedMatrix, error_rate)
                    else:
                        matrixNoisy = mixedMatrix

                    # === PRE-PROCESSING : identifier les strips évidents par HCA ===
                    ambiguous_cols, preproc_steps = pre_processing(
                        matrixNoisy, min_col_quality=1, certitude=certitude, error_rate=error_rate)

                    row = {
                        'Error-Rate': error_rate,
                        'Strip': stripe,
                        'Haplotype': haplotypes,
                        'Matrix-Rows': mixed_info.matrix.shape[0],
                        'Matrix-Cols': mixed_info.matrix.shape[1],
                        'PreProc-Strips': len(preproc_steps),
                        'PreProc-Ambiguous-Cols': len(ambiguous_cols),
                    }

                    # ILP reçoit uniquement les colonnes non résolues par le pré-processing
                    regions = ambiguous_cols

                    # ===== APPROCHE 1 : ALTERNANTE (1 puis 0 puis 1...) =====
                    start = time.time()
                    steps_alt_ilp, _ = clustering_full_matrix(
                        matrixNoisy, regions=regions, only_ones=False,
                        min_row_quality=1, min_col_quality=1, error_rate=error_rate)
                    time_alt = time.time() - start
                    # Fusionner les steps du pré-processing avec ceux de l'ILP
                    steps_alt = list(preproc_steps) + list(steps_alt_ilp)
                    clusters_alt, _, orphans_alt, _ = post_processing(
                        matrixNoisy, steps_alt, read_names, distance_thresh, 1)
                    eq_alt = clusters_equal(baseline_clusters, clusters_alt)

                    row['alt-Time'] = time_alt
                    row['alt-Haplotypes'] = len(clusters_alt)
                    row['alt-Stripe'] = len(steps_alt)
                    row['alt-Orfelin'] = len(orphans_alt)
                    row['alt-Is-Equal'] = 1 if eq_alt else 0

                    # ===== APPROCHE 2 : ONLY ONES (uniquement les 1) =====
                    start = time.time()
                    steps_o1_ilp, _ = clustering_full_matrix(
                        matrixNoisy, regions=regions, only_ones=True,
                        min_row_quality=1, min_col_quality=1, error_rate=error_rate)
                    time_o1 = time.time() - start
                    # Fusionner les steps du pré-processing avec ceux de l'ILP
                    steps_o1 = list(preproc_steps) + list(steps_o1_ilp)
                    clusters_o1, _, orphans_o1, _ = post_processing(
                        matrixNoisy, steps_o1, read_names, distance_thresh, 1)
                    eq_o1 = clusters_equal(baseline_clusters, clusters_o1)

                    row['only1-Time'] = time_o1
                    row['only1-Haplotypes'] = len(clusters_o1)
                    row['only1-Stripe'] = len(steps_o1)
                    row['only1-Orfelin'] = len(orphans_o1)
                    row['only1-Is-Equal'] = 1 if eq_o1 else 0

                    print(f"  it={i+1}  alt: {len(clusters_alt)}h/{len(steps_alt)}s eq={eq_alt}  |  "
                          f"only1: {len(clusters_o1)}h/{len(steps_o1)}s eq={eq_o1}")

                    # Log erreurs si une des deux approches échoue
                    if log_error and (not eq_alt or not eq_o1
                                      or len(clusters_alt) != haplotypes
                                      or len(clusters_o1) != haplotypes):
                        os.makedirs("temp", exist_ok=True)
                        filename = f"temp/{int(time.time())}.txt"
                        with open(filename, 'w') as f:
                            f.write(f"Stripe: {stripe}\n")
                            f.write(f"Haplotypes expected: {haplotypes}\n")
                            f.write(f"Matrix shape: {mixed_info.matrix.shape}\n")
                            f.write("Mixed matrix:\n")
                            for r in mixed_info.matrix:
                                f.write(" ".join(map(str, r)) + "\n")
                            f.write(f"\n=== ALTERNATING ===\n")
                            f.write(f"Time: {time_alt}\n")
                            f.write(f"Haplotypes: {len(clusters_alt)}, Stripes: {len(steps_alt)}, Equal: {eq_alt}\n")
                            f.write(f"Steps:\n{pprint.pformat(steps_alt, width=1000)}\n")
                            f.write(f"Clusters:\n{pprint.pformat(clusters_alt, width=1000)}\n")
                            f.write(f"\n=== ONLY ONES ===\n")
                            f.write(f"Time: {time_o1}\n")
                            f.write(f"Haplotypes: {len(clusters_o1)}, Stripes: {len(steps_o1)}, Equal: {eq_o1}\n")
                            f.write(f"Steps:\n{pprint.pformat(steps_o1, width=1000)}\n")
                            f.write(f"Clusters:\n{pprint.pformat(clusters_o1, width=1000)}\n")

                    writer.writerow(row)
                    try:
                        csvfile.flush()
                        os.fsync(csvfile.fileno())
                    except Exception:
                        pass


if __name__ == "__main__":
    var_file = "onlyOne/config.arg"
    args = parse_arg_file(var_file)
    main(args)
