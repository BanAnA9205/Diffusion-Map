import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

from helpers import bgh, _core_diffusion_map

def diffmap_dense(X, epsilon, alpha=1.0, n_components=2, t=1):
    """
    Computes the diffusion map embedding for a dense dataset X.

    Parameters:
    - X (ndarray): The input dataset of shape (n_samples, n_features).
    - epsilon (float): The local neighborhood structure bandwidth (Gaussian kernel variance).
    - alpha (float): The normalization parameter (1.0 corresponds to Laplace-Beltrami operator).
    - n_components (int): The number of diffusion dimensions to return.
    - t (int): The diffusion time step. Acts as a low-pass filter on the eigenvalues.

    Returns:
    - diffusion_coords (ndarray): The mapped dataset coordinates of shape (n_samples, n_components).
    - evals_k (ndarray): The top non-trivial eigenvalues.
    """
    # 1. Isotropic Gaussian Kernel
    sq_dists = squareform(pdist(X, metric='sqeuclidean'))
    K = np.exp(-sq_dists / epsilon)

    return _core_diffusion_map(K, alpha, n_components, t, is_sparse=False)

def diffmap_sparse(X, epsilon, k=10, alpha=1.0, n_components=2, t=1):
    """
    Computes the diffusion map embedding using a sparse k-NN graph. Very highly optimized 
    for large datasets.

    Parameters:
    - X (ndarray): The input dataset of shape (n_samples, n_features).
    - epsilon (float): The local neighborhood structure bandwidth (Gaussian kernel variance).
    - k (int): The number of nearest neighbors to connect in the sparse graph.
    - alpha (float): The normalization parameter.
    - n_components (int): The number of diffusion dimensions to return.
    - t (int): The diffusion time step.

    Returns:
    - diffusion_coords (ndarray): The mapped dataset coordinates.
    - evals_k (ndarray): The top non-trivial eigenvalues.
    """
    n_samples = X.shape[0]

    # 1. k-NN Search (O(N log N) using KD-Tree/Ball-Tree)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # 2. Sparse Kernel Construction (CSR format for fast row operations)
    row_indices = np.repeat(np.arange(n_samples), k+1)
    col_indices = indices.ravel()
    
    np.square(distances, out=distances)
    np.multiply(distances, -1.0 / epsilon, out=distances)
    np.exp(distances, out=distances)
    affinities = distances.ravel()

    K = sp.csr_matrix((affinities, (row_indices, col_indices)), shape=(n_samples, n_samples))

    # 3. Symmetrize the k-NN Graph
    K = K.maximum(K.T)

    return _core_diffusion_map(K, alpha, n_components, t, is_sparse=True)

def diffmap(X, epsilon=None, method='auto', k=None, alpha=1.0, n_components=2, t=1):
    """
    Wrapper for computing Diffusion Maps, supporting routing to sparse or dense variants,
    and automatically estimating epsilon if not provided.
    
    Parameters:
    - X (ndarray): The input dataset of shape (n_samples, n_features).
    - epsilon (float, optional): The Gaussian kernel bandwidth. Auto-estimated if None.
    - method (str): Strategy to use: 'auto', 'sparse', or 'dense'.
    - k (int, optional): Number of nearest neighbors for the sparse graph.
    - alpha (float): Normalization parameter (default: 1.0).
    - n_components (int): The number of dimensions for the embedding.
    - t (int): Diffusion time step (default: 1).
    
    Returns:
    - diffusion_coords (ndarray): The mathematically mapped dataset coordinates.
    - evals_k (ndarray): The top non-trivial eigenvalues.
    """
    n_samples = X.shape[0]
    
    if method == 'auto':
        method = 'sparse' if n_samples > 2000 or k is not None else 'dense'
        
    if method == 'sparse' and k is None:
        k = 10
        
    if epsilon is None:
        epsilon = bgh(X, method=method, k=k)
        print(f"Auto-selected epsilon ({method} BGH): {epsilon:.4f}")
        
    if method == 'sparse':
        actual_k = min(k, n_samples - 1)
        return diffmap_sparse(X, epsilon, k=actual_k, alpha=alpha, n_components=n_components, t=t)
    elif method == 'dense':
        return diffmap_dense(X, epsilon, alpha=alpha, n_components=n_components, t=t)
    else:
        raise ValueError("method must be 'sparse', 'dense', or 'auto'.")
