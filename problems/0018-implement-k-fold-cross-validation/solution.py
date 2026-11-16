import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    
    # Your code here

    # 1. Generate a list of indices
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    splits = []
    # 2. Split into train and test
    start = 0
    for i in range(k):
        fold_size = n_samples // k + (i < n_samples % k)
        end = start + fold_size

        test = indices[start:end]
        train = np.concatenate((indices[:start], indices[end:]))

        start = end

        splits.append((train, test))
    
    return splits





