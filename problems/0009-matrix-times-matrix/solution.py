def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    m = len(a)
    n = len(a[0])
    l = len(b[0])

    if len(b) != n:
        return -1
    
    c = [[0 for _ in range(l)] for _ in range(m)]

    for i in range(m):
        for j in range(n):
            for k in range(l):
                c[i][k] += a[i][j] * b[j][k]

    return c