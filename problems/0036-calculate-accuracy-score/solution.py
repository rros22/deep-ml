import numpy as np

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # The numerator is TP + TN
    # The denominator is just TP + FP + TN + FN == Total number of points
    return float(np.mean(y_true == y_pred))