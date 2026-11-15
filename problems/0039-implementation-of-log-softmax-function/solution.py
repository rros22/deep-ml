import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    # Your code here
    shifted = scores - scores.max()
    return shifted - np.log(np.sum(np.exp(shifted)))