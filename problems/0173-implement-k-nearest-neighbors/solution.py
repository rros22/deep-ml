import numpy as np
import heapq

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """

    points = np.asarray(points)

    distances = np.linalg.norm(points - query_point, axis=1)

    indices = np.argsort(distances, kind="stable")

    return [tuple(p) for p in points[indices[:k]]]
