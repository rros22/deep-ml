import numpy as np

def stratified_kfold_indices(y, n_splits):
    """
    Generate train/test indices for stratified K-fold cross-validation.

    Args:
        y: 1D array-like of integer class labels
        n_splits: number of folds

    Returns:
        A list of [train_indices, test_indices] pairs, one per fold.
    """
    # Cast to numpy array
    y = np.asarray(y)
    indices = np.arange(len(y))

    test_folds = [[] for _ in range(n_splits)]

    for label in np.unique(y):
        class_indices = indices[y == label]

        for i, fold in enumerate(np.array_split(class_indices, n_splits)):
            test_folds[i].extend(fold.tolist())

    return [
        [np.setdiff1d(indices, test).tolist(), sorted(test)]
        for test in test_folds
    ]

