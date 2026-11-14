import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    # This will not work for constant features though.
    normalized_data = (data - data.min(axis=0))/(data.max(axis=0) - data.min(axis=0))
    standardized_data = (data - data.mean(axis=0))/(data.std(axis=0))

    return standardized_data, normalized_data