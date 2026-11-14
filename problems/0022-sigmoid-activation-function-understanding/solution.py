
import math

def sigmoid(z: float) -> float:
    """Compute the sigmoid function with numerical stability."""

    if z >= 0:
        # exp(-z) <= 1, so exponential overflow is avoided.
        result = 1 / (1 + math.exp(-z))
    else:
        # Use the equivalent formula for negative inputs.
        # exp(z) <= 1, avoiding overflow from exp(-z).
        exp_z = math.exp(z)
        result = exp_z / (1 + exp_z)

    return round(result, 4)
