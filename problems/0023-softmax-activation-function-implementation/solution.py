import numpy as np

def softmax(scores: list[float]) -> list[float]:
    scores = np.asarray(scores, dtype=float)
    max_score = scores.max()

    # Shift scores for numerical stability.
    exp_scores = np.exp(scores - max_score)

    # Normalize probabilities and return a Python list.
    return (exp_scores / exp_scores.sum()).tolist()