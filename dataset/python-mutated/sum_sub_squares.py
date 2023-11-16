def sum_sub_squares(matrix, k):
    if False:
        i = 10
        return i + 15
    n = len(matrix)
    result = [[0 for i in range(k)] for j in range(k)]
    if k > n:
        return
    for i in range(n - k + 1):
        l = 0
        for j in range(n - k + 1):
            sum = 0
            for p in range(i, k + i):
                for q in range(j, k + j):
                    sum += matrix[p][q]
            result[i][l] = sum
            l += 1
    return result