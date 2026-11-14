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
    # Your code here
    # Gradient function
    def grad(Xb, yb, weights):
        # k = # of training examples 
        k = Xb.shape[0]
        return 2/k*Xb.T@(Xb@weights - yb)

    if method == "batch":
        # Use the full dataset
        for _ in range(n_epochs):
            weights -= learning_rate*grad(X,y,weights)
    
    elif method == 'mini_batch':
        # 1. First partition the dataset into batches
        if batch_size < 0:
            print("Batch size needs to be greater than zero")
            return weights

        batch_indices = range(0, len(X), batch_size)

        for _ in range(n_epochs):
            for i in batch_indices:
                Xb, yb = X[i:i + batch_size], y[i:i + batch_size]
                weights -= learning_rate*grad(Xb,yb,weights)
    
    elif method == 'stochastic':
        for _ in range(n_epochs):
            for i in range(len(X)):
                weights -= learning_rate*grad(X[i:i+1], y[i:i+1], weights)
            

    return weights

