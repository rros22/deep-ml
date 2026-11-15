import numpy as np

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    # Implement your code here
    # 1. Convert to NP arrays
    y_true = np.asarray(actual)
    y_pred = np.asarray(predicted)

    # 2. Compute confusion matrix entries
    tp = np.count_nonzero((y_true == 1) & (y_pred == 1))
    fp = np.count_nonzero((y_true == 0) & (y_pred == 1))
    tn = np.count_nonzero((y_true == 0) & (y_pred == 0))
    fn = np.count_nonzero((y_true == 1) & (y_pred == 0))

    # Construct metrix
    confusion_matrix = [[tp, fn], [fp, tn]]
    accuracy = (tp + tn) / y_pred.shape[0]
    specificity = tn / (tn + fp)
    negativePredictive = tn / (tn + fn)
    precision = tp /(tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*precision*recall / (precision + recall)
    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
