import numpy as np

def precision(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Mask
    predicted_positive = y_pred == 1
    tp = np.count_nonzero((y_true == 1) & predicted_positive)
    total_positive = np.count_nonzero(predicted_positive)

    return tp / total_positive if total_positive else 0.0