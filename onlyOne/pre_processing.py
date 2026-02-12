import numpy as np
import logging
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

logger = logging.getLogger(__name__)

def hamming_distance_matrix(matrix):
    """
    Compute Hamming distance matrix between columns of the input matrix.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        Input matrix with shape (m, n)
        
    Returns
    -------
    numpy.ndarray
        Hamming distance matrix with shape (n, n)
    """
    logger.debug(f"Calcul de la matrice de distance de Hamming pour une matrice de forme {matrix.shape}")
    # Convert to binary if not already
    binary_matrix = (matrix != 0).astype(int)
    
    # Compute Hamming distance between all pairs of columns
    distances = pdist(binary_matrix.T, metric='hamming')
    distance_matrix = squareform(distances)
    logger.debug("Matrice de distance de Hamming calculée.")
    
    return distance_matrix

def is_strip(matrix, column_indices, error_rate=0.025):
    """
    Check if a group of columns forms a strip by applying HCA to partition reads.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        Input matrix with shape (m, n)
    column_indices : list
        List of column indices to check
    error_rate : float
        Error rate threshold for strip identification
        
    Returns
    -------
    bool
        True if the columns form a strip, False otherwise
    tuple or None
        If strip is found, returns (cluster0_indices, cluster1_indices, column_indices)
    """
    logger.debug(f"Test du strip sur les colonnes {column_indices} (erreur tolérée: {error_rate})")
    if len(column_indices) < 2:
        logger.debug("Moins de 2 colonnes, impossible de former un strip.")
        return False, None
    
    # Extract submatrix for the columns
    submatrix = matrix[:, column_indices]
    logger.debug(f"Sous-matrice extraite de forme {submatrix.shape}")
    
    if submatrix.sum() <= 0 + error_rate * submatrix.size:
        logger.info("Sous-matrice quasi-nulle : strip trivial de zéros.")
        return True, ([i for i in range(submatrix.shape[0])], [], column_indices)
    if submatrix.sum() >= submatrix.size - error_rate * submatrix.size:
        logger.info("Sous-matrice quasi-un : strip trivial de uns.")
        return True, ([], [i for i in range(submatrix.shape[0])], column_indices)
    # Apply hierarchical clustering to partition reads into two groups
    clustering = AgglomerativeClustering(
        n_clusters=2,
        linkage='complete',
        metric='hamming'
    )
    logger.debug("Clustering hiérarchique des lignes...")

    try:
        # Cluster the rows (reads)
        row_clusters = clustering.fit_predict(submatrix)
        logger.debug("Clustering des lignes terminé.")
        
        # Check if the clustering forms a strip pattern
        cluster0_mask = (row_clusters == 0)
        cluster1_mask = (row_clusters == 1)
        
        # Calculate the proportion of ones in each cluster
        cluster0_ones = np.sum(submatrix[cluster0_mask] == 1)
        cluster1_ones = np.sum(submatrix[cluster1_mask] == 1)
        
        cluster0_total = cluster0_mask.sum() * len(column_indices)
        cluster1_total = cluster1_mask.sum() * len(column_indices)
        # Avoid division by zero by checking that each total is > 0 before dividing
        if cluster0_total > 0 or cluster1_total > 0:
            if cluster0_total > 0:
                cluster0_ratio = cluster0_ones / cluster0_total
            else:
                cluster0_ratio = 0  # or np.nan, but 0 is safe for the logic below

            if cluster1_total > 0:
                cluster1_ratio = cluster1_ones / cluster1_total
            else:
                cluster1_ratio = 0  # or np.nan, but 0 is safe for the logic below

            logger.debug(f"Ratios de uns : cluster0={cluster0_ratio:.3f}, cluster1={cluster1_ratio:.3f}")
            # Check if one cluster has mostly zeros and the other has mostly ones
            # (with tolerance defined by certitude)
            if (cluster0_ratio <= error_rate and cluster1_ratio >= (1 - error_rate)) or \
               (cluster1_ratio <= error_rate and cluster0_ratio >= (1 - error_rate)):

                # Return the strip information
                cluster0_indices = np.where(cluster0_mask)[0].tolist()
                cluster1_indices = np.where(cluster1_mask)[0].tolist()
                logger.info(f"Strip détecté : {len(cluster0_indices)} lignes à 0, {len(cluster1_indices)} lignes à 1.")
                return True, (cluster0_indices, cluster1_indices, column_indices)
    
    except Exception as e:
        logger.warning(f"Error in strip identification: {e}")
    return False, None

def pre_processing(input_matrix: np.ndarray, min_col_quality: int = 5, certitude: float = 0.2, error_rate: float = 0.025) -> tuple:
    """
    Pre-processes the input matrix by identifying strips using hierarchical clustering.
    
    Steps performed:
    1. Hierarchical clustering of columns using Hamming distance and complete-linkage strategy
    2. Merge groups until all groups have more than 35% divergence with others
    3. Filter out groups with fewer than min_col_quality columns
    4. For each group, apply HCA to partition reads into two groups
    5. Identify "unambiguous" strips (1-y bicluster of zeros and y-bicluster of ones)
    
    Parameters
    ----------
    input_matrix : numpy.ndarray
        The input matrix to be pre-processed, with shape (m, n).
    min_col_quality : int, optional
        Minimum number of columns required for a region to be considered significant (default is 5).
    certitude : float, optional
        Certainty threshold for strip identification (default is 0.2).
    error_rate : float, optional
        Error rate threshold for strip identification (default is 0.025).
        
    Returns
    -------
    inhomogenious_regions : list of int
        List of column indices identified as inhomogeneous/ambiguous.
    steps : list of tuple
        List of tuples describing the identified strips. Each tuple contains:
            (cluster0_indices, cluster1_indices, region_columns)
    """
    # Validate input
    if input_matrix.ndim != 2:
        logger.error("Input matrix must be 2-dimensional")
        raise ValueError("Input matrix must be 2-dimensional")
    
    if error_rate < 0 or error_rate >= 0.5:
        logger.error("error_rate must be in [0, 0.5)")
        raise ValueError("error_rate must be in [0, 0.5)")
    
    logger.info(f"Starting pre-processing with matrix shape: {input_matrix.shape}")
    
    # Handle edge cases
    if input_matrix.size == 0:
        logger.info("Matrice vide : rien à traiter.")
        return [], []
    
    if input_matrix.shape[1] == 1:
        logger.info("Matrice à une seule colonne : considérée comme inhomogène.")
        return [0], []
    
    # For small matrices or when min_col_quality is greater than matrix width,
    # return all columns as inhomogeneous
    if input_matrix.shape[1] < min_col_quality:
        logger.info(f"Trop peu de colonnes ({input_matrix.shape[1]}) pour min_col_quality={min_col_quality} : toutes inhomogènes.")
        return list(range(input_matrix.shape[1])), []
    
    # Step 1: Hierarchical clustering of columns using Hamming distance
    logger.info("Computing Hamming distance matrix...")
    binary_matrix = (input_matrix != 0).astype(int)
    condensed_distances = pdist(binary_matrix.T, metric='hamming')
    logger.debug(f"Distance de Hamming condensée : shape {condensed_distances.shape}")
    
    # Apply hierarchical clustering with complete linkage
    logger.info("Applying hierarchical clustering...")
    linkage_matrix = linkage(condensed_distances, method='complete')
    logger.debug(f"Matrice de linkage : shape {linkage_matrix.shape}")
    
    # Cut the dendrogram to get clusters with >35% divergence (distance > 0.35)
    cluster_labels = fcluster(linkage_matrix, t=certitude, criterion='distance')
    logger.debug(f"Labels de clusters obtenus : {np.unique(cluster_labels)}")
    
    # Group columns by cluster
    unique_clusters = np.unique(cluster_labels)
    column_groups = []
    for cluster_id in unique_clusters:
        cluster_columns = np.where(cluster_labels == cluster_id)[0].tolist()
        if len(cluster_columns) >= min_col_quality:
            column_groups.append(cluster_columns)
    logger.info(f"Found {len(column_groups)} column groups after filtering")
    
    identified_strips = []
    ambiguous_columns = set()
    used_columns = set()
    
    for group in column_groups:
        logger.debug(f"Checking group with {len(group)} columns: {group}")
        is_strip_result, strip_info = is_strip(input_matrix, group, error_rate)
        if is_strip_result:
            logger.info(f"Identified strip with {len(group)} columns")
            identified_strips.append(strip_info)
            used_columns.update(group)
        else:
            logger.debug(f"Group is ambiguous, adding to inhomogeneous regions")
            ambiguous_columns.update(group)
            used_columns.update(group)
    
    # Any columns not used in strips or ambiguous groups are ambiguous
    all_columns = set(range(input_matrix.shape[1]))
    ambiguous_columns.update(all_columns - used_columns)
    
    # If no strips found, all columns are ambiguous
    if not identified_strips:
        ambiguous_columns = all_columns
    
    # If ambiguous_columns covers all columns, steps must be empty
    if ambiguous_columns == all_columns:
        identified_strips = []
    
    logger.info(f"Identified {len(identified_strips)} strips and {len(ambiguous_columns)} ambiguous columns")
    return sorted(list(ambiguous_columns)), identified_strips