import numpy as np

def cross_val_predict(X, y, k):
    # 1. Create the folds
    X = np.asarray(X)
    y = np.asarray(y)
    indices = np.arange(X.shape[0])
    fold_indices = np.array_split(indices, k)

    # Compute predictions
    predictions = np.zeros(len(y))  # Changed
    for i, test_indices in enumerate(fold_indices):
        train_indices = np.setdiff1d(indices, test_indices)

        # Compute regression fit
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_train = np.concatenate(
            (np.ones((len(train_indices), 1)), X_train), axis=1
        ) 
        # Use pseudo inverse to solve the linear system 
        params = np.linalg.pinv(X_train) @ y_train

        # Evaluate the regression fit
        X_test = np.concatenate(
            (np.ones((len(test_indices), 1)), X[test_indices]), axis=1
        )
        pred = X_test @ params

        # Append result
        predictions[test_indices] = pred  # Changed

    return list(predictions)