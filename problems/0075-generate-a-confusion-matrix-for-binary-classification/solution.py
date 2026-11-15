def confusion_matrix(data):
    matrix = [[0, 0], [0, 0]]

    for i, j in data:
        matrix[1-i][1-j] += 1

    return matrix