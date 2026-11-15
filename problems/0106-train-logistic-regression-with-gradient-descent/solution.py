
import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], list[float]]:
    """Train logistic regression using full-batch gradient descent and BCE loss."""

    # Numerically stable sigmoid
    def sigmoid(x):
        x = np.asarray(x, dtype=float)
        result = np.empty_like(x)

        pos = x >= 0
        neg = ~pos

        result[pos] = 1 / (1 + np.exp(-x[pos]))

        # Equivalent form avoids overflow for negative inputs.
        exp_neg = np.exp(x[neg])
        result[neg] = exp_neg / (1 + exp_neg)

        return result

    # Add bias column and initialize weights.
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    theta = np.zeros(X.shape[1], dtype=float)
    losses = []

    for _ in range(iterations):
        z = X @ theta
        y_hat = sigmoid(z)

        # BCE = sum(log(1 + exp(z)) - y*z).
        # logaddexp computes log(1 + exp(z)) without overflow.
        loss = np.sum(np.logaddexp(0, z) - y * z)
        losses.append(loss)

        # Gradient of summed BCE (no 1/N factor).
        theta -= learning_rate * (X.T @ (y_hat - y))

    return theta.round(4).tolist(), np.round(losses, 4).tolist()
