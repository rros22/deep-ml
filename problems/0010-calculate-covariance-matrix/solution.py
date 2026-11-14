def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    f = len(vectors)
    n = len(vectors[0])

    means = [sum(feature) / n for feature in vectors]
    cov = [[0.0] * f for _ in range(f)]

    for i in range(f):
        for j in range(i, f):
            covariance = sum(
                (x - means[i]) * (y - means[j])
                for x, y in zip(vectors[i], vectors[j])
            ) / (n - 1)

            cov[i][j] = cov[j][i] = covariance

    return cov