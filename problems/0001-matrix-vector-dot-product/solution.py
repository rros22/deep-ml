def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Return a list where each element is the dot product of a row of 'a' with 'b'.
    # If the number of columns in 'a' does not match the length of 'b', return -1.
    rows = len(a)
    cols = len(a[0])

    if cols != len(b):
        return -1
    
    # Main loop
    result = [0 for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            result[i] += a[i][j] * b[j]
    
    return result