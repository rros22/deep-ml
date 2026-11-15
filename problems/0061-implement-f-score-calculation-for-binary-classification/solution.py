import numpy as np

def f_score(y_true, y_pred, beta):
    """
    Calculate the F-beta score for binary classification.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        beta: Weight of recall relative to precision.
              beta > 1 favors recall; beta < 1 favors precision.

    Returns:
        F-beta score rounded to 3 decimal places.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate confusion matrix components
    tp = np.count_nonzero((y_true == 1) & (y_pred == 1))
    fp = np.count_nonzero((y_true == 0) & (y_pred == 1))
    fn = np.count_nonzero((y_true == 1) & (y_pred == 0))

    # Calculate precision and recall, avoiding division by zero
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    # Calculate F-beta score
    beta_sq = beta ** 2
    denominator = beta_sq * precision + recall

    if denominator == 0:
        return 0.0

    score = (1 + beta_sq) * precision * recall / denominator

    return round(float(score), 3)