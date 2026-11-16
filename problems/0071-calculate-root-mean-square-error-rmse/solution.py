import numpy as np

def rmse(y_true, y_pred):
    if (y_true.shape != y_pred.shape) or (y_true.size == 0) or (y_pred.size == 0):
        raise ValueError("Arrays must have the same shape and must not be empty")

    return round(np.sqrt(np.mean((y_true - y_pred)**2)), 3)