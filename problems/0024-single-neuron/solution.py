import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int],
                        weights: list[float], bias: float) -> tuple[list[float], float]:

    X = np.asarray(features, dtype=float)
    w = np.asarray(weights, dtype=float)
    y = np.asarray(labels, dtype=float)

    # Numerically stable, vectorized sigmoid.
    def sigmoid(z):
        result = np.empty_like(z)
        pos = z >= 0
        neg = ~pos

        result[pos] = 1 / (1 + np.exp(-z[pos]))

        # Equivalent form avoids overflow for negative logits.
        exp_neg = np.exp(z[neg])
        result[neg] = exp_neg / (1 + exp_neg)

        return result

    # Forward pass.
    probabilities = sigmoid(X @ w + bias)

    # Mean squared error.
    mse = np.mean((probabilities - y) ** 2)

    return probabilities.round(4).tolist(), round(float(mse), 4)