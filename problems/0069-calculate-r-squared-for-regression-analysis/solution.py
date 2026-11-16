
import numpy as np

def r_squared(y_true, y_pred):
    MSE_mean = np.mean((y_true - y_true.mean())**2)
    MSE_linear = np.mean((y_true - y_pred)**2)
    return 1 - MSE_linear / MSE_mean
