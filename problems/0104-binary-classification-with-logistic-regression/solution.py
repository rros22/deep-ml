import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # Sigmoid definition
    def sigmoid(x):
        # 1. Convert input to numpy array
        x = np.asarray(x, dtype=np.float32)
        # 2. Empty array to store result
        res = np.empty_like(x)
        # 3. Create boolean masks for positive and negative values
        pos = x >= 0
        neg = ~pos
        # 4. Fill array
        res[pos] = 1/(1+np.exp(-x[pos]))
        res[neg] = np.exp(x[neg])/(1 + np.exp(x[neg]))

        return res
        

    # Your code here
    return sigmoid(X@weights + bias) >= 0.5