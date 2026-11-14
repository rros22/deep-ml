import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    weights = np.array(weights, dtype=float, copy=True)

    def grad(Xb, yb, weights):
        """Compute the MSE gradient over the current batch."""
        k = Xb.shape[0]

        # Xb @ weights gives predictions; subtract yb for residuals.
        # Xb.T maps residuals back to the feature/weight dimension.
        return (2 / k) * (Xb.T @ (Xb @ weights - yb))

    if method == 'batch':
        # One update per epoch using all observations.
        for _ in range(n_epochs):
            weights -= learning_rate * grad(X, y, weights)

    elif method == 'mini_batch':
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        # Sequential batches; the final batch may be smaller.
        for _ in range(n_epochs):
            for i in range(0, len(X), batch_size):
                Xb = X[i:i + batch_size]
                yb = y[i:i + batch_size]
                weights -= learning_rate * grad(Xb, yb, weights)

    elif method == 'stochastic':
        # One update per observation, without shuffling.
        for _ in range(n_epochs):
            for i in range(len(X)):
                # Slicing preserves the 2D X and 1D y shapes.
                weights -= learning_rate * grad(
                    X[i:i + 1], y[i:i + 1], weights
                )

    else:
        raise ValueError(f"Unknown gradient descent method: {method}")

    return weights
