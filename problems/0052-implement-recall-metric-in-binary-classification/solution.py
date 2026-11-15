import numpy as np

def recall(y_true, y_pred):
    """
    Calculate the recall metric for binary classification.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    
    Returns:
        Recall value as a float
    """
    # Your code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    predicted_positive = y_pred == 1
    predicted_negative = y_pred == 0
    tp = np.count_nonzero((y_true == 1) & predicted_positive)
    fn = np.count_nonzero((y_true == 1) & predicted_negative)

    return tp / (tp + fn) if tp + fn else 0.0
