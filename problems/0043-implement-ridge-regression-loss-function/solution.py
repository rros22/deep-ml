import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # Your code here
    return 1/X.shape[0]*np.sum((X@w - y)**2) + alpha*np.sum(w**2)