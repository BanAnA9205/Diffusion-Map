import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

def _core_diffusion_map(K, alpha, n_components, t, is_sparse):
    """
    Core engine for Diffusion Maps. Abstracts the normalization, Markov matrix
    symmetrization, and spectral decomposition math independent of kernel sparsity.
    
    Parameters:
    - K: The initial (possibly sparse) symmetric Kernel matrix.
    - alpha: Normalization parameter.
    - n_components: Target dimensions to extract.
    - t: Diffusion time steps.
    - is_sparse: Boolean indicating if K is a sparse matrix.
    
    Returns:
    - diffusion_coords, evals_k
    """
    # 1. Density (Alpha) Normalization 
    q = np.asarray(K.sum(axis=1)).flatten()
    if alpha == 1.0:
        np.divide(1.0, q, out=q)
    else:
        np.power(q, -alpha, out=q)

    # Scale K appropriately
    if is_sparse:
        D_q = sp.diags(q)
        K_alpha = D_q @ K @ D_q
    else:
        K_alpha = K * q[:, None] * q[None, :]

    # 2. Symmetrization & Markov Matrix Construction
    d = np.asarray(K_alpha.sum(axis=1)).flatten()
    np.sqrt(d, out=d)
    np.divide(1.0, d, out=d)
    
    if is_sparse:
        D_d = sp.diags(d)
        S = D_d @ K_alpha @ D_d
    else:
        S = K_alpha * d[:, None] * d[None, :]

    # 3. Eigendecomposition 
    if is_sparse:
        evals, evecs = eigsh(S, k=n_components + 1, which='LA')
    else:
        evals, evecs = eigh(S)

    # Flip to descending order so lambda_0 is at index 0
    evals = np.flip(evals, axis=0)
    evecs = np.flip(evecs, axis=1)

    # Drop the trivial stationary state (index 0) and slice the requested components
    evals_k = evals[1 : n_components + 1]
    phi_k = evecs[:, 1 : n_components + 1]

    # 4. Recover Right Eigenvectors (psi) and apply time scale t
    psi_k = phi_k * d[:, None]
    diffusion_coords = psi_k * (evals_k**t)

    return diffusion_coords, evals_k

def bgh(X, method='auto', k=None, sample_size=2000, num_eps=100):
    """
    Berry-Giannakis-Harlim (BGH) heuristic / Coifman error algorithm for
    automatically choosing the optimal epsilon bandwidth for the Gaussian kernel.

    Parameters:
    - X (ndarray): The input dataset of shape (n_samples, n_features).
    - method (str): 'auto', 'dense', or 'sparse'. Strategy to compute pairwise distances.
    - k (int, optional): The number of neighbors to compute if method is 'sparse'. Defaults to 10.
    - sample_size (int): Maximum number of samples to use for 'dense' distance computation.
    - num_eps (int): Number of epsilon values to test in the logarithmic scale grid.

    Returns:
    - float: The optimally chosen epsilon bandwidth.
    """
    N = X.shape[0]
    
    # Safe fallback if k was passed as None
    if k is None:
        k = 10
        
    if method == 'auto':
        method = 'sparse' if N > 2000 else 'dense'
        
    if method == 'sparse':
        actual_k = min(k + 1, N)
        nbrs = NearestNeighbors(n_neighbors=actual_k, algorithm='auto').fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # In-place square and 1D view (ravel) avoids 2 memory allocations!
        np.square(distances, out=distances)
        sq_dists = distances.ravel()
    else:
        # Subsample if dataset is too large to save memory on dense pdist
        if N > sample_size:
            indices = np.random.choice(N, sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        sq_dists = pdist(X_sample, metric='sqeuclidean')
    
    # Filter out absolute 0s to avoid log(0) issues
    nz_sq_dists = sq_dists[sq_dists > 0]
    if len(nz_sq_dists) == 0:
        return 1.0 # Fallback if all points are identical
        
    eps_min = np.min(nz_sq_dists)
    eps_max = np.max(sq_dists)
    
    # Create the grid of epsilons on a logarithmic scale
    eps_grid = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)
    
    L = np.zeros(num_eps)
    for i, eps in enumerate(eps_grid):
        # Sum of the unnormalized kernel matrix
        L[i] = np.sum(np.exp(-sq_dists / eps))
        
    log_eps = np.log(eps_grid)
    log_L = np.log(L)
    
    # Calculate the derivative d(log L) / d(log eps)
    derivatives = np.gradient(log_L, log_eps)
    
    # The optimal epsilon maximizes this derivative (linear region of log-log plot)
    opt_idx = np.argmax(derivatives)
    
    return eps_grid[opt_idx]

def nystrom_extension(X_new, X_train, embedding_train, eigval_train, epsilon, k=None):
    """
    Computes an efficient Nystrom extension of out-of-sample data points.
    
    Parameters:
    - X_new: Out-of-sample points to embed (n_new, n_features).
    - X_train: Original training points (n_samples, n_features).
    - embedding_train: Diffusion coordinates of X_train (n_samples, n_components).
    - eigval_train: Eigengaps/evals of the diffusion process (n_components).
    - epsilon: Gaussian kernel bandwidth.
    - k: If provided, uses sparse k-NN for efficiency, otherwise dense distances.
    
    Returns:
    - embedding_new: Estimated diffusion coordinates for X_new.
    """
    if k is not None:
        # Fast sparse implementation
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_train)
        distances, indices = nbrs.kneighbors(X_new)
        
        # Calculate kernel values
        affinities = np.exp(-np.square(distances) / epsilon)
        
        # Build sparse cross-kernel matrix
        n_new = X_new.shape[0]
        n_train = X_train.shape[0]
        row_indices = np.repeat(np.arange(n_new), k)
        col_indices = indices.ravel()
        
        K_cross = sp.csr_matrix((affinities.ravel(), (row_indices, col_indices)), shape=(n_new, n_train))
        
        # Row-normalize to get transition probabilities
        q_cross = np.asarray(K_cross.sum(axis=1)).flatten()
        # Avoid division by zero
        q_cross[q_cross == 0] = 1.0
        D_inv = sp.diags(1.0 / q_cross)
        P_cross = D_inv @ K_cross
        
        # Project
        embedding_new = (P_cross @ embedding_train) / eigval_train
    else:
        # Dense implementation
        from scipy.spatial.distance import cdist
        sq_dists = cdist(X_new, X_train, metric='sqeuclidean')
        K_cross = np.exp(-sq_dists / epsilon)
        
        q_cross = K_cross.sum(axis=1, keepdims=True)
        q_cross[q_cross == 0] = 1.0  # Safe division
        P_cross = K_cross / q_cross
        
        embedding_new = (P_cross @ embedding_train) / eigval_train
        
    return embedding_new
