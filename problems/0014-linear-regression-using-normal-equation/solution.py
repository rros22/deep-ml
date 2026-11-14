import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Your code here, make sure to round
    coeffs = np.round(np.linalg.pinv(X) @ y, 4)
    return coeffs.tolist()