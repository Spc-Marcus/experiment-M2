import numpy as np
from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering
import logging

logger = logging.getLogger(__name__)

def cluster_mean(cluster, matrix):
    cluster_matrix = matrix[cluster]
    # Plus besoin de gérer les -1, on suppose 0/1
    return np.rint(cluster_matrix.sum(axis=0) / len(cluster))

def hamming_distance_with_mask(read_vec, mean_vec):
    # Plus besoin de masque, tout est 0/1
    return np.mean(read_vec != mean_vec)

def merge_similar_clusters(clusters, means, distance_thresh):
    if len(clusters) <= 1:
        return clusters
    agglo = AgglomerativeClustering(n_clusters=None, metric='hamming', linkage='complete', distance_threshold=distance_thresh)
    labels = agglo.fit_predict(means)
    merged = {}
    for idx, label in enumerate(labels):
        if label not in merged:
            merged[label] = []
        merged[label].extend(clusters[idx])
    return list(merged.values())

def reassign_orphans(rem_, clusters, means, matrix, threshold=0.3):
    for r in rem_:
        read_vec = matrix[r]
        dists = [hamming_distance_with_mask(read_vec, mean) for mean in means]
        if len(dists) > 0:
            idx_most_similar = np.argmin(dists)
            if dists[idx_most_similar] < threshold:
                clusters[idx_most_similar].append(r)
    return clusters

def post_processing(matrix: np.ndarray, steps: List[Tuple[List[int], List[int], List[int]]], 
                   read_names: List[str], distance_thresh: float = 0.1,min_reads_per_cluster: int = 5) -> Tuple[List[np.ndarray], np.ndarray, list, list]:
    """
    Post-process clustering results and create final strain clusters.
    
    This function takes the output of the biclustering algorithm and creates final
    read clusters representing different microbial strains. It processes hierarchical
    clustering steps, filters small clusters, assigns orphaned reads, and merges
    similar clusters to produce biologically meaningful strain groups.
    
    Parameters
    ----------
    matrix : np.ndarray
        Binary variant matrix with shape (n_reads, n_positions) containing
        processed variant calls (0, 1)
    steps : List[Tuple[List[int], List[int], List[int]]]
        List of clustering steps from biclustering algorithm. Each step contains:
        - reads1: List of read indices assigned to cluster 1
        - reads0: List of read indices assigned to cluster 0  
        - cols: List of column indices used for this clustering decision
    read_names : List[str]
        List of read names corresponding to matrix rows
    distance_thresh : float, optional
        Distance threshold for merging similar clusters using Hamming distance.
        Default is 0.1 (10% difference threshold)
    min_reads_per_cluster : int, optional
        Minimum number of reads per cluster.
        Default is 5.
    Returns
    -------
    clusters : List[np.ndarray]
        Liste des clusters finaux, chaque cluster étant un tableau numpy des noms de reads.
    reduced_matrix : np.ndarray
        Matrice réduite de shape (nb_clusters, nb_steps), chaque ligne étant le profil consensus d'un cluster sur les stripes (une colonne par step).
    orphan_reads_names : list
        Liste des noms de reads non clusterisés à la fin (orphelins).
    unused_columns : list
        Liste des indices de colonnes non utilisées dans les steps (colonnes non "stripes").
    """
    logger.info(f"Post-processing {len(read_names)} reads with {len(steps)} clustering steps.")
    # 1. Initialisation : tous les reads dans un seul cluster
    clusters = [list(range(len(read_names)))]
    logger.debug(f"Initial clusters: {len(clusters)} cluster(s) with {len(read_names)} reads.")
    unused_columns = set(range(matrix.shape[1]))
    logger.debug(f"Initial unused columns: {unused_columns}")
    logger.debug(f"Initial clusters: {clusters}")
    logger.debug(f"Initial steps: {steps}")
    # 2. Application des étapes de biclustering
    for step_idx, step in enumerate(steps):
        reads1, reads0, cols = step
        new_clusters = []
        unused_columns = unused_columns - set(cols)
        for cluster_idx, cluster in enumerate(clusters):
            clust1 = [c for c in cluster if c in reads1]
            clust0 = [c for c in cluster if c in reads0]
            if len(clust1) > 0:
                new_clusters.append(clust1)
            if len(clust0) > 0:
                new_clusters.append(clust0)
            logger.debug(f"Step {step_idx}, cluster {cluster_idx}: split into {len(clust1)} (reads1) and {len(clust0)} (reads0) reads using cols {cols}.")
        clusters = new_clusters
        logger.info(f"After step {step_idx}: {len(clusters)} clusters.")
    for cluster_idx, cluster in enumerate(clusters):
        logger.info(f"Cluster {cluster_idx}: {cluster}")


    # 3. Calcul des moyennes de clusters
    mean_of_clusters = [cluster_mean(cluster, matrix) for cluster in clusters]
    logger.debug(f"Calculated mean vectors for {len(clusters)} clusters.")

    # 4. Fusion des clusters similaires (agglomératif sur les moyennes)
    n_before_merge = len(clusters)
    clusters = merge_similar_clusters(clusters, mean_of_clusters, distance_thresh)
    logger.info(f"Merged clusters: {n_before_merge} -> {len(clusters)} after agglomerative clustering (distance_thresh={distance_thresh}).")

    # 5. Première réaffectation des reads orphelins
    mean_of_clusters = [cluster_mean(cluster, matrix) for cluster in clusters]
    logger.debug(f"Calculated mean vectors for {len(clusters)} clusters.")

    all_clustered = set([r for cluster in clusters for r in cluster])
    rem_ = []
    for read in range(len(read_names)):
        if read not in all_clustered:
            rem_.append(read)
    if len(rem_) > 0 and len(clusters) > 0:
        logger.info(f"First reassignment: reassigning {len(rem_)} orphaned reads to closest clusters.")
        clusters = reassign_orphans(rem_, clusters, mean_of_clusters, matrix, distance_thresh)
        mean_of_clusters = [cluster_mean(cluster, matrix) for cluster in clusters]
    else:
        logger.info("No orphaned reads to reassign in first pass.")
    
    # 6. Filtrage des petits clusters
    rem_ = []
    big_clusters = []
    if min_reads_per_cluster is not None:
        for cluster in clusters:
            if len(cluster) < min_reads_per_cluster:
                rem_ += list(cluster)
            else:
                big_clusters.append(cluster)
        logger.info(f"Filtered out {len(clusters) - len(big_clusters)} clusters smaller than {min_reads_per_cluster}.")
        clusters = big_clusters
    else:
        logger.info("No minimum cluster size filtering applied.")
    
    # 7. Deuxième réaffectation des reads orphelins (après filtrage)
    if len(rem_) > 0 and len(clusters) > 0:
        logger.info(f"Second reassignment: reassigning {len(rem_)} orphaned reads from filtered clusters.")
        mean_of_clusters = [cluster_mean(cluster, matrix) for cluster in clusters]
        clusters = reassign_orphans(rem_, clusters, mean_of_clusters, matrix, distance_thresh)
    elif len(rem_) > 0:
        logger.info(f"{len(rem_)} reads became orphaned after filtering but no clusters remain to reassign them to.")
    else:
        logger.info("No orphaned reads after filtering.")
    
    # 8. Conversion des indices en noms et matrice réduite
    result_clusters = []
    reduced_matrix = []
    clustered_reads = set()
    for idx, cluster in enumerate(clusters):
        # On ne garde que les clusters non vides
        cluster = [r for r in cluster if 0 <= r < len(read_names)]
        if len(cluster) > 0:
            names = [read_names[r] for r in cluster]
            result_clusters.append(np.array(sorted(names)))
            clustered_reads.update(cluster)
            # Pour chaque step, calculer la moyenne sur les colonnes de ce step
            cluster_profile = []
            for step in steps:
                cols = step[2]
                cluster_matrix = matrix[cluster][:, cols]
                mean_val = np.rint(cluster_matrix.mean()) if cluster_matrix.size > 0 else 0
                cluster_profile.append(mean_val)
            reduced_matrix.append(cluster_profile)
            logger.info(f"Cluster {idx}: {len(cluster)} reads.")
            logger.info(f"Cluster {idx}: {cluster}")
    if reduced_matrix:
        reduced_matrix = np.array(reduced_matrix)
        logger.info(f"Reduced matrix shape: {reduced_matrix.shape}")
    else:
        reduced_matrix = np.array([])
        logger.warning("No reads in final clusters. Reduced matrix is empty.")
    # Orphelins : reads non clusterisés à la fin
    all_reads = set(range(len(read_names)))
    orphan_reads = all_reads - clustered_reads
    orphan_reads_names = [read_names[r] for r in sorted(orphan_reads)]
    if orphan_reads:
        logger.info(f"{len(orphan_reads)} reads are not present in any final cluster and are ignored in the reduced matrix.")
        logger.debug(f"Orphan reads: {orphan_reads}")
    # Always return unused_columns as a sorted list
    unused_columns = sorted(list(unused_columns))
    if len(unused_columns) > 0:
        logger.info(f"Unused columns: {unused_columns}")
    logger.info(f"Post-processing completed: {len(result_clusters)} clusters returned.")
    return result_clusters, reduced_matrix, orphan_reads_names, unused_columns
