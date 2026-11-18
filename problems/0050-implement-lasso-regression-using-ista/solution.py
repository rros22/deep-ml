
import numpy as np

def soft_threshold(w, threshold):
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)


def l1_regularization_gradient_descent(
    X, y, alpha=0.1, learning_rate=0.01,
    max_iter=1000, tol=1e-4
):
    n_samples, n_features = X.shape

    weights = np.zeros(n_features)
    bias = 0.0

    for _ in range(max_iter):
        predictions = X @ weights + bias
        error = predictions - y

        grad_w = (X.T @ error) / n_samples
        grad_b = np.mean(error)

        new_weights = soft_threshold(
            weights - learning_rate * grad_w,
            learning_rate * alpha
        )

        bias -= learning_rate * grad_b

        if np.linalg.norm(new_weights - weights) < tol:
            weights = new_weights
            break

        weights = new_weights

    return weights, float(bias)
