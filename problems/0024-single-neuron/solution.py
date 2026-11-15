import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int],
                        weights: list[float], bias: float) -> tuple[list[float], float]:

    X = np.asarray(features, dtype=float)
    w = np.asarray(weights, dtype=float)
    y = np.asarray(labels, dtype=float)

    # Forward pass: weighted sum followed by sigmoid.
    probabilities = 1 / (1 + np.exp(-(X @ w + bias)))

    # Mean squared error across observations.
    mse = np.mean((probabilities - y) ** 2)

    return probabilities.round(4).tolist(), round(float(mse), 4)