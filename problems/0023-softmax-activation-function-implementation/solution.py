import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    scores = np.array(scores)
    max_score = scores.max()
    return np.exp(scores - max_score) / sum(np.exp(scores - max_score))
