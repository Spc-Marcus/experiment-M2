import numpy as np
from typing import List, Tuple
from ilp_grb import *
def clustering_full_matrix(input_matrix:np.ndarray, 
        regions :list[int],
        version:int=1,
        min_row_quality:int=5,
        min_col_quality:int = 3,
        error_rate : float = 0.025) -> tuple[list[tuple[list[int], list[int], list[int]]], dict]:
    """
    Perform exhaustive iterative biclustering on a binary matrix to extract all significant patterns.
    
    This function systematically processes predefined regions of a binary matrix to identify 
    all possible separations between rows based on their column patterns. It applies binary 
    clustering iteratively until no more significant patterns can be found.
    
    Parameters
    ----------
    input_matrix : np.ndarray
        Input binary matrix with values (0, 1) where rows and columns represent 
        data points and features respectively. Values indicate feature presence (1), 
        absence (0).
    regions : List[List[int]]
        List of column indices to process.
    min_row_quality : int, optional
        Minimum number of rows required for a cluster to be considered valid.
        Default is 5.
    min_col_quality : int, optional
        Minimum number of columns required for a region to be processed.
        Regions with fewer columns are skipped. Default is 3.
    error_rate : float, optional
        Tolerance level for pattern detection, allowing for noise and imperfections
        in the binary patterns. Default is 0.025 (2.5%).
    
    Returns
    -------
    List[Tuple[List[int], List[int], List[int]]]
        Complete list of all valid clustering steps found. Each tuple contains:
        - [0] : List[int] - Row indices in first group (pattern match)(0s)
        - [1] : List[int] - Row indices in second group (pattern opposite)(1s)
        - [2] : List[int] - Column indices where this separation is significant
        
        Only returns steps where both groups are non-empty and column count
        meets minimum quality threshold.
    dict
        Dictionary containing the following metrics:
        - "nb_ilp_steps": int
        - "max_ilp_cluster_size": int
        - "min_density_cluster0": float
        - "max_density_cluster0": float
        - "mean_density_cluster0": float
        - "min_density_cluster1": float
        - "max_density_cluster1": float
        - "mean_density_cluster1": float
        - "nb_strips_from_ilp": int
        

    
    Algorithm
    ---------
    1. **Initialization**: Start with existing clustering steps
    
    2. **Region Iteration**: Process each column region independently:
       - Skip regions with insufficient columns (< min_col_quality)
       - Initialize remaining columns for processing
       
    3. **Exhaustive Pattern Extraction**: For each region:
       - Apply binary clustering to find one significant row separation
       - Convert local column indices to global matrix coordinates
       - Save valid separations to results
       - Remove processed columns from remaining set
       - Continue until no significant patterns remain
    
    4. **Result Filtering**: Return only clustering steps that satisfy:
       - Both row groups contain at least one element
       - Column set meets minimum quality requirements
    """
    # Initialize result list with existing steps
    steps_result = []
    metrics_list = []
    remain_cols = regions
    status = True
            
    # Only process regions that meet minimum quality threshold
    if len(remain_cols) >= min_col_quality:
        # Iteratively extract patterns until no more significant ones found
        while len(remain_cols) >= min_col_quality and status:
                    # Apply clustering to current remaining columns
            (reads1, reads0, cols), metricclustering_steps = clustering_step(input_matrix[:, remain_cols], 
                                                          error_rate=error_rate,
                                                          min_row_quality=min_row_quality, 
                                                          min_col_quality=min_col_quality,
                                                          version=version)
                    
                    # Convert local column indices back to global matrix coordinates
            cols = [remain_cols[c] for c in cols]
            if len(reads1) + len(reads0) < 0.9 * len(input_matrix[:, remain_cols]):
                status = False

            # Check if valid pattern was found
            elif len(cols) < min_col_quality:
                status = False  # Ou continue, selon si tu veux continuer à chercher d'autres patterns
            else:
                # Save valid clustering step
                steps_result.append((reads1, reads0, cols))
                metrics_list.append(metricclustering_steps)

                    # Remove processed columns from remaining set
                remain_cols = [c for c in remain_cols if c not in cols]
    
    # Calcul des métriques globales
    if metrics_list:
        nb_ilp_steps = sum((m.get('nb_ilp_steps', 0) for m in metrics_list))
        max_ilp_cluster_size = max((m.get('max_ilp_cluster_size', -1) for m in metrics_list), default=-1)
        dens0_list = [m.get('density_cluster0', -1) for m in metrics_list if m.get('density_cluster0', -1) >= 0]
        dens1_list = [m.get('density_cluster1', -1) for m in metrics_list if m.get('density_cluster1', -1) >= 0]
        min_density_cluster0 = min(dens0_list) if dens0_list else -1
        max_density_cluster0 = max(dens0_list) if dens0_list else -1
        mean_density_cluster0 = sum(dens0_list)/len(dens0_list) if dens0_list else -1
        min_density_cluster1 = min(dens1_list) if dens1_list else -1
        max_density_cluster1 = max(dens1_list) if dens1_list else -1
        mean_density_cluster1 = sum(dens1_list)/len(dens1_list) if dens1_list else -1
        nb_strips_from_ilp = len(steps_result)
        found = True
    else:
        nb_ilp_steps = 0
        max_ilp_cluster_size = -1
        min_density_cluster0 = -1
        max_density_cluster0 = -1
        mean_density_cluster0 = -1
        min_density_cluster1 = -1
        max_density_cluster1 = -1
        mean_density_cluster1 = -1
        nb_strips_from_ilp = 0
        found = False
    metrics = {
        "nb_ilp_steps": nb_ilp_steps,
        "max_ilp_cluster_size": max_ilp_cluster_size,
        "min_density_cluster0": min_density_cluster0,
        "max_density_cluster0": max_density_cluster0,
        "mean_density_cluster0": mean_density_cluster0,
        "min_density_cluster1": min_density_cluster1,
        "max_density_cluster1": max_density_cluster1,
        "mean_density_cluster1": mean_density_cluster1,
        "nb_strips_from_ilp": nb_strips_from_ilp,
        "found": found
    }
    return steps_result, metrics

def clustering_step(input_matrix: np.ndarray,
                    version:int=1,
                    error_rate: float = 0.025,
                    min_row_quality: int = 5,
                    min_col_quality: int = 3,
                    ) -> tuple[tuple[list[int], list[int], list[int]], dict]:
    """
    Perform a single binary clustering step on a matrix to identify one significant row separation.
    
    This function applies alternating quasi-biclique detection to find groups of rows with 
    similar patterns across columns. It processes both positive (1s) and negative (0s) patterns 
    iteratively until a stable separation is found or no significant patterns remain.
    
    Parameters
    ----------
    input_matrix : np.ndarray
        Input binary matrix with values (0, 1) where rows represent data points 
        and columns represent features. Values indicate feature presence (1), 
        absence (0)
    error_rate : float, optional
        Tolerance level for quasi-biclique detection, allowing for noise and 
        imperfections in binary patterns. Default is 0.025 (2.5%).
    min_row_quality : int, optional
        Minimum number of rows required to continue processing. Algorithm stops 
        when fewer rows remain unprocessed. Default is 5.
    min_col_quality : int, optional
        Minimum number of columns required to continue processing. Algorithm stops 
        when fewer significant columns remain. Default is 3.
    
    Returns
    -------
    Tuple[List[int], List[int], List[int]]
        Triple containing the results of binary clustering:
        - [0] : List[int] - Row indices with positive pattern (predominantly 0s)
        - [1] : List[int] - Row indices with negative pattern (predominantly 1s)  
        - [2] : List[int] - Column indices where separation is most significant
        
        Empty lists are returned for categories where no significant patterns are found.
    dict
        Dictionary containing the following metrics:
        - "found": bool, True if a valid cluster was found
        - "nb_ilp_steps": int
        - "max_ilp_cluster_size": int
        - "density_cluster0": float
        - "density_cluster1": float
    
    Algorithm
    ---------
    1. **Matrix Preparation**:
       - Create negative matrix: invert values to detect 0-patterns
       
    2. **Alternating Pattern Detection**:
       - Start with all rows and columns available for processing
       - Alternate between searching for 1-patterns and 0-patterns
       - Apply quasi-biclique optimization to find dense sub-matrices
       
    3. **Noise Filtering**:
       - For each detected pattern, calculate column homogeneity
       - Retain only columns with homogeneity > 5 × error_rate
       - Remove extremely noisy or inconsistent columns
       
    4. **Iterative Refinement**:
       - Accumulate rows into positive or negative groups based on current pattern
       - Remove processed rows from remaining set
       - Continue until insufficient rows or columns remain
       
    5. **Termination Conditions**:
       - Stop when fewer than min_row_quality rows remain
       - Stop when fewer than min_col_quality columns remain  
       - Stop when no significant patterns are detected
    """
    # Variable to track the number of ILP steps
    nb_ilp_steps = 0
    max_ilp_cluster_size = 0
    found = False

    # Create binary matrices for pattern detection
    # Handle -1 values by treating them as 0s in positive pattern matrix
    matrix1 = input_matrix.copy()
    matrix1[matrix1 == -1] = 0  # Convert missing values to 0 for positive patterns
    
    # Create inverted matrix for negative pattern detection
    # Convert -1 to 1, then invert all values (0->1, 1->0)
    matrix0 = input_matrix.copy()
    matrix0[matrix0 == -1] = 1
    matrix0 = (matrix0 - 1) * -1  # Invert matrix for negative pattern detection
    
    # Initialize tracking variables for iterative clustering
    remain_rows = range(matrix1.shape[0])  # All rows initially available
    current_cols = range(matrix1.shape[1])  # All columns initially available
    clustering_1 = True  # Alternate between positive (True) and negative (False) patterns
    status = True  # Continue while valid patterns are found
    rw1, rw0 = [], []  # Accumulate rows for positive and negative groups

    # Iteratively extract patterns until insufficient data remains
    while len(remain_rows) >= min_row_quality and len(current_cols) >= min_col_quality and status:
        # Apply quasi-biclique detection on appropriate matrix
        if clustering_1:
            # Search for positive patterns (dense regions of 1s)
            rw, cl, status = ilp(matrix1[remain_rows][:, current_cols], error_rate,version)
        else:
            # Search for negative patterns (dense regions of 0s in original)
            rw, cl, status = ilp(matrix0[remain_rows][:, current_cols], error_rate,version)
        nb_ilp_steps += 1
        print("Col: " , cl, "Rows: " , rw, "IsOne: ", clustering_1)
        # Convert local indices back to global matrix coordinates
        rw = [remain_rows[r] for r in rw]  # Map row indices to original matrix
        cl = [current_cols[c] for c in cl]  # Map column indices to original matrix

        if len(cl) < min_col_quality:   
            status = False
        else:
            current_cols = cl  # Update working column set to detected significant columns
            
            # Accumulate rows into appropriate pattern group if valid pattern found
            if status and len(cl) > 0:
                found = True
                if len(rw) * len(cl) > max_ilp_cluster_size:
                    max_ilp_cluster_size = len(rw) * len(cl)
                if clustering_1:
                    rw1.extend(rw)  # Add to positive pattern group
                else:
                    rw0.extend(rw)  # Add to negative pattern group
                
        # Remove processed rows from remaining set for next iteration
        remain_rows = [r for r in remain_rows if r not in rw]
        # Alternate pattern detection type for next iteration
        clustering_1 = not clustering_1
        
    # Log final clustering statistics
    if isinstance(current_cols, range):
        current_cols = list(current_cols)
    if isinstance(rw0, range):
        rw0 = list(rw0)
    if isinstance(rw1, range):
        rw1 = list(rw1)
        
    if found:
        if len(rw0) > 0:
            submatrix0 = input_matrix[rw0, :][:, current_cols]
            if submatrix0.shape[0] > 0 and submatrix0.shape[1] > 0:
                density_cluster0 = submatrix0.sum() / (submatrix0.shape[0] * submatrix0.shape[1])
            else:
                density_cluster0 = -1
        else:
            density_cluster0 = -1
        if len(rw1) > 0:
            submatrix1 = input_matrix[rw1, :][:, current_cols]
            if submatrix1.shape[0] > 0 and submatrix1.shape[1] > 0:
                density_cluster1 = submatrix1.sum() / (submatrix1.shape[0] * submatrix1.shape[1])
            else:
                density_cluster1 = -1
        else:
            density_cluster1 = -1
    else:
        density_cluster0 = -1
        density_cluster1 = -1
    metrics = {
        "nb_ilp_steps": nb_ilp_steps,
        "max_ilp_cluster_size": max_ilp_cluster_size,
        "density_cluster0": density_cluster0,
        "density_cluster1": density_cluster1,
        "found": found,
    }
    return (rw0, rw1, current_cols), metrics

def ilp(input_matrix: np.ndarray, error_rate: float = 0.025, version:int=1) -> Tuple[List[int], List[int], bool]:
    """
    Wrapper function for quasi-biclique detection using integer linear programming.
    
    Parameters
    ----------
    input_matrix : np.ndarray
        Input binary matrix
    error_rate : float, optional
        Tolerance level for pattern detection. Default is 0.025.
    version : int, optional
        Version of the ILP model to use. Default is 1.
        
        Versions disponibles:
        - 1: V1 original (max_Ones_gurobi) - Nouveau modèle à chaque phase
        - 2: V2 (MaxOneModel) - Un seul modèle avec contraintes dynamiques
        - 3: max_e_r - Modèle avec error rate dans l'objectif
        - 4: V3 - Un seul modèle + contraintes dynamiques + WarmStart (comme V2 mais avec WarmStart)
        - 5: V3a - Un seul modèle + contraintes dynamiques, SANS WarmStart
        - 6: V3b - Reconstruction modèle (seeds only) + WarmStart
        - 7: V3c - Reconstruction modèle (seeds only), SANS WarmStart
        
    Returns
    -------
    Tuple[List[int], List[int], bool]
        Row indices, column indices, and success status
    """
    if version == 1:
        return find_quasi_dens_matrix_max_ones(input_matrix, error_rate)
    if version == 2:
        return find_quasi_biclique_max_one_V2(input_matrix, error_rate)
    if version == 3:
        return find_quasi_biclique_max_e_r_V2(input_matrix, error_rate)
    if version == 4:
        return find_quasi_biclique_max_one_V3(input_matrix, error_rate)
    if version == 5:
        return find_quasi_biclique_max_one_V3a(input_matrix, error_rate)
    if version == 6:
        return find_quasi_biclique_max_one_V3b(input_matrix, error_rate)
    if version == 7:
        return find_quasi_biclique_max_one_V3c(input_matrix, error_rate)
    # Default to version 1
    return find_quasi_dens_matrix_max_ones(input_matrix, error_rate)