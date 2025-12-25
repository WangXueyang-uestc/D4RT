"""
Miscellaneous utility functions
"""

import numpy as np


def farthest_point_sample_py(points, n_samples):
    """
    Farthest Point Sampling (FPS) algorithm
    
    Args:
        points: (N, D) array of points
        n_samples: number of points to sample
    
    Returns:
        indices: (n_samples,) array of selected point indices
    """
    N, D = points.shape
    if n_samples >= N:
        return np.arange(N)
    
    # Initialize
    selected_indices = np.zeros(n_samples, dtype=np.int64)
    distances = np.ones(N) * np.inf
    
    # Start with a random point
    selected_indices[0] = np.random.randint(0, N)
    last_selected = points[selected_indices[0]]
    
    # Iteratively select farthest points
    for i in range(1, n_samples):
        # Update distances
        dists = np.sum((points - last_selected) ** 2, axis=1)
        distances = np.minimum(distances, dists)
        
        # Select farthest point
        selected_indices[i] = np.argmax(distances)
        last_selected = points[selected_indices[i]]
    
    return selected_indices



