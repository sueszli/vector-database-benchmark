def multiply(matA: list, matB: list) -> list:
    if False:
        while True:
            i = 10
    '\n    Multiplies two square matrices matA and matB of size n x n\n    Time Complexity: O(n^3)\n    '
    n = len(matA)
    matC = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                matC[i][j] += matA[i][k] * matB[k][j]
    return matC

def identity(n: int) -> list:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns the Identity matrix of size n x n\n    Time Complexity: O(n^2)\n    '
    I = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        I[i][i] = 1
    return I

def matrix_exponentiation(mat: list, n: int) -> list:
    if False:
        while True:
            i = 10
    '\n    Calculates mat^n by repeated squaring\n    Time Complexity: O(d^3 log(n))\n                     d: dimension of the square matrix mat\n                     n: power the matrix is raised to\n    '
    if n == 0:
        return identity(len(mat))
    elif n % 2 == 1:
        return multiply(matrix_exponentiation(mat, n - 1), mat)
    else:
        tmp = matrix_exponentiation(mat, n // 2)
        return multiply(tmp, tmp)